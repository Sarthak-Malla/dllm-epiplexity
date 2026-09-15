"""Shared frozen-state diagnostics, imported by the Slurm-only experiment runner.

Run runner.py diagnose --experiment selectors (or precedence/proposals/attention_mass)
using the absolute command and environment preparation in the suite README.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import time

import torch

from examples.path_selection.experiments.artifacts import stable_hash
from examples.path_selection.experiments.configs import continuation_config, fixed_config
from dllm.core.samplers.batched_lookahead import candidate_batch_from_mask_mapping
from dllm.core.samplers.dependency_guided import (
    build_dependency_candidates,
    reconstruct_dependency_for_proposals,
    release_dependency_capture_tensors,
)
from dllm.core.samplers.parallel_candidates import build_symmetric_conflict_matrix


def candidate_records(prepared):
    """Retain only compact proposal summaries, never vocabulary distributions."""
    records = []
    pool = prepared.candidates
    for index, name in enumerate(pool.names):
        if not bool(pool.candidate_valid[index, 0]):
            continue
        positions = pool.selected_positions[index, 0]
        positions = positions[positions >= 0]
        records.append({
            "index": index, "name": name, "positions": positions.tolist(),
            "seed": int(pool.seed_anchors[index, 0]),
            "original_token_ids": prepared.x0[0, positions].tolist(),
            "confidence": prepared.confidence[0, positions].tolist(),
            "entropy": prepared.entropy[0, positions].tolist(),
            "top2_margin": prepared.top2_margin[0, positions].tolist(),
            "metadata": dict(pool.metadata[index]),
        })
    return records


def prepare_pools(prepared, *, include_confidence=False, include_mass=False):
    """Return pools, optional absolute attention, and disjoint component timings."""
    pools = {"adaptive": prepared}
    requested = torch.full_like(prepared.requested_k, 4)
    choices = [("fixed4", fixed_config())]
    if include_confidence:
        choices.append(("confidence4", fixed_config(confidence_proposals=True)))
    for name, config in choices:
        candidates, dependency, reconstruction, proposal = build_dependency_candidates(
            prepared.base_forward, active_mask=prepared.active_mask,
            requested_k=requested, anchor_state=prepared.state.anchor_state,
            response_mask=prepared.state.response_mask,
            attention_mask=prepared.state.attention_mask,
            entropy_map=prepared.entropy, confidence=prepared.confidence,
            config=config, generation_seed=prepared.step_seed,
        )
        pools[name] = replace(
            prepared, config=config, candidates=candidates, dependency=dependency,
            reconstruction_seconds=reconstruction, proposal_seconds=proposal,
            selections={},
        )
    absolute_dependency = None
    absolute_reconstruction_seconds = 0.0
    if include_mass:
        active = prepared.active_mask
        if prepared.state.anchor_state is not None:
            active = active | prepared.state.anchor_state.reliable_anchor_mask
        if prepared.x0.device.type == "cuda":
            torch.cuda.synchronize(prepared.x0.device)
        absolute_started = time.perf_counter()
        absolute_dependency = reconstruct_dependency_for_proposals(
            prepared.base_forward, active_mask=active,
            response_mask=prepared.state.response_mask,
            attention_mask=prepared.state.attention_mask,
            config=replace(prepared.config, dependency_preserve_attention_mass=True,
                           dependency_renormalize_selected_keys=False),
            region_masks={
                "all_masked": prepared.masked_active_mask & prepared.state.response_mask,
                "current_masked": prepared.active_mask,
                "available_context": prepared.state.attention_mask.bool() & ~prepared.masked_active_mask,
                "available_response": prepared.state.response_mask & ~prepared.masked_active_mask,
                "earlier_blocks": prepared.state.response_mask & (
                    torch.arange(prepared.x0.shape[1], device=prepared.x0.device)[None, :]
                    < prepared.state.prompt_lens[0] + prepared.state.block_index * prepared.config.block_size
                ),
                "anchors": (prepared.state.anchor_state.reliable_anchor_mask
                            if prepared.state.anchor_state is not None else torch.zeros_like(active)),
            },
        )
        if prepared.x0.device.type == "cuda":
            torch.cuda.synchronize(prepared.x0.device)
        absolute_reconstruction_seconds = time.perf_counter() - absolute_started
        if prepared.dependency.sink_mask is not None:
            if not torch.equal(prepared.dependency.sink_mask, absolute_dependency.sink_mask):
                raise RuntimeError("Attention comparison did not preserve reference sink identities.")
    release_dependency_capture_tensors(prepared.base_forward)
    return pools, absolute_dependency, {
        # Every pool references this same captured base pass; count it once.
        "base_forward_seconds": prepared.base_forward.base_forward_seconds,
        "reconstruction_seconds": sum(pool.reconstruction_seconds for pool in pools.values())
                                  + absolute_reconstruction_seconds,
        "absolute_reconstruction_seconds": absolute_reconstruction_seconds,
        "proposal_seconds": sum(pool.proposal_seconds for pool in pools.values()),
    }


def first_companion(prepared, candidate_index):
    """Map recorded compact construction order to absolute sequence positions."""
    seed = int(prepared.candidates.seed_anchors[candidate_index, 0])
    metadata = prepared.candidates.metadata[candidate_index]
    orders = metadata.get("construction_order_by_batch")
    if orders is None or orders[0] is None:
        raise RuntimeError("A pair control requires the candidate's actual construction order.")
    positions = [int(prepared.dependency.query_positions[0, compact]) for compact in orders[0]]
    if not positions or positions[0] != seed:
        raise RuntimeError("Recorded construction order does not start with the recorded seed.")
    return next(position for position in positions if position != seed)


def pair_prepared(prepared, candidate_index):
    """Construct the labeled seed/first-companion control in recorded growth order."""
    seed = int(prepared.candidates.seed_anchors[candidate_index, 0])
    companion = first_companion(prepared, candidate_index)
    mask = torch.zeros_like(prepared.active_mask)
    mask[0, [seed, companion]] = True
    batch = candidate_batch_from_mask_mapping({"seed_first_companion_pair": mask},
                                              eligible_mask=prepared.active_mask)
    batch = replace(batch, seed_anchors=torch.tensor([[seed]], device=mask.device))
    return replace(prepared, candidates=batch, selections={})


def directed_features(prepared, seed, companion, dependency=None):
    """Distinguish attention direction, normalized conflict, and confidence penalty."""
    dependency = prepared.dependency if dependency is None else dependency
    positions = dependency.query_positions[0]
    valid = dependency.query_valid_mask[0]
    mapping = {int(position): index for index, position in enumerate(positions.tolist()) if bool(valid[index])}
    left, right = mapping[seed], mapping[companion]
    compact_eligible = torch.zeros_like(dependency.query_valid_mask)
    compact_eligible[0, valid] = prepared.active_mask[0, positions[valid]]
    conflict = build_symmetric_conflict_matrix(
        dependency.directed, compact_eligible,
        normalization=prepared.config.dependency_conflict_normalization,
    )
    normalized = float(conflict.matrix[0, left, right])
    seed_confidence = float(prepared.confidence[0, seed])
    confidence = float(prepared.confidence[0, companion])
    risk = 1.0 - min(seed_confidence, confidence)
    record = {
        "seed": seed, "companion": companion,
        "seed_to_companion": float(dependency.directed[0, left, right]),
        "companion_to_seed": float(dependency.directed[0, right, left]),
        "symmetric_conflict": normalized,
        "conflict_scale": float(conflict.scale_by_batch[0]),
        "confidence_weight": risk,
        "penalty": prepared.config.dependency_conflict_penalty * normalized * risk,
        "confidence": confidence, "seed_confidence": seed_confidence,
        "entropy": float(prepared.entropy[0, companion]),
        "top2_margin": float(prepared.top2_margin[0, companion]),
        "selected_key_mass": float(dependency.directed[0, right].sum()),
    }
    masses = getattr(dependency, "attention_mass_by_region", None)
    if masses is not None:
        record["attention_mass_by_region"] = {
            name: float(value[0, right]) for name, value in masses.items()
        }
    return record


class DiagnosticRunner:
    """Execute isolated actions and reuse only complete matching branch identities."""

    def __init__(self, sampler, store, accounting, score_output):
        self.sampler = sampler
        self.store = store
        self.accounting = accounting
        self.score_output = score_output

    def branch(self, prepared, index, *, state_key, doc_id, mode="simultaneous", probe=None,
               reuse_label="branch"):
        mask = prepared.candidates.candidate_masks[index]
        token_ids = prepared.x0
        if probe is not None:
            token_ids = torch.as_tensor(probe["token_ids"], device=mask.device, dtype=torch.long)
        branch = prepared.state.clone(include_history=False)
        # Prior logging is not decoder state. Keep branch tracing disabled so
        # identical actions from different proposal pools have the same complete
        # continuation state and do not retain unnecessary token trajectories.
        branch.histories = None
        branch.diagnostics = None
        branch.selected_candidates = [[] for _ in branch.prompt_lens]
        # Passing base confidence deliberately prevents refreshed reliability from
        # changing anchors; one macro-step is advanced in every intervention.
        self.sampler.commit_step(branch, prepared, mask, token_ids=token_ids,
                                 confidence=prepared.confidence)
        continuation = continuation_config()
        identity = {"doc_id": doc_id, "state": branch.state_dict(), "continuation": asdict(continuation)}
        key = stable_hash(identity)
        if self.store.has("branches", key):
            self.store.record_reuse("branches", f"{state_key}-{reuse_label}", key)
            return key
        before = self.accounting.snapshot()
        started = time.perf_counter()
        with self.accounting.scope("diagnostic_continuation"):
            output = self.sampler.continue_from_state(branch, continuation)
        score = self.score_output(doc_id, output.sequences)
        positions = torch.where(mask[0])[0]
        self.store.put_json("branches", key, {
            "doc_id": doc_id, "state_key": state_key, "mode": mode,
            "positions": positions.tolist(),
            "committed_token_ids": token_ids[0, positions].tolist(),
            "anchor_confidence": prepared.confidence[0, positions].tolist(),
            "next_decision_index": prepared.state.global_step_index + 1,
            "continuation_config_hash": stable_hash(asdict(continuation)),
            "post_commit_state_hash": stable_hash(identity["state"]),
            "accounting": self.accounting.delta(before),
            "wall_seconds": time.perf_counter() - started, **score,
        })
        return key

    def probe(self, prepared, index, *, state_key, doc_id, reverse=False, reuse_label="probe"):
        positions = prepared.candidates.selected_positions[index, 0].tolist()
        positions = [position for position in positions if position >= 0]
        seed = int(prepared.candidates.seed_anchors[index, 0])
        first = next(position for position in positions if position != seed) if reverse else seed
        forward_keys = ("temperature", "cfg_scale", "cfg_keep_tokens", "suppress_tokens",
                        "begin_suppress_tokens", "right_shift_logits")
        key = stable_hash({"doc_id": doc_id, "state": prepared.state.state_dict(), "rng_after": prepared.rng_after,
                           "first": first, "first_value": int(prepared.x0[0, first]),
                           "forward_config": {name: getattr(prepared.config, name) for name in forward_keys}})
        if self.store.has("probes", key):
            self.store.record_reuse("probes", f"{state_key}-{reuse_label}", key)
            return key, self.store.get_json("probes", key)
        before = self.accounting.snapshot()
        started = time.perf_counter()
        probe_prepared, probe_index = prepared, index
        if len(positions) == 1 and int(prepared.active_mask.sum()) > 1:
            # Core deployable singleton commitment needs no refresh. The
            # diagnostic nevertheless probes its seed to measure other masked
            # values and permit correct universal-seed reuse by larger groups.
            other = next(position for position in torch.where(prepared.active_mask[0])[0].tolist()
                         if position != first)
            probe_mask = torch.zeros_like(prepared.active_mask)
            probe_mask[0, [first, other]] = True
            probe_batch = candidate_batch_from_mask_mapping({"seed_probe": probe_mask},
                                                            eligible_mask=prepared.active_mask)
            probe_batch = replace(probe_batch, seed_anchors=torch.tensor([[first]], device=probe_mask.device))
            probe_prepared = replace(prepared, candidates=probe_batch, selections={})
            probe_index = 0
        with self.accounting.scope("diagnostic_seed_probe"):
            result = self.sampler.probe_seed_first(probe_prepared, probe_index, reverse=reverse)
        # One seed-only forward refreshes every currently masked position, so the
        # compact result can be safely reused by other groups sharing that seed.
        refreshed = result.refreshed_probabilities
        refreshed_ids = result.refreshed_token_ids
        if refreshed_ids is None or refreshed_ids.shape != prepared.x0.shape:
            raise RuntimeError("Seed probes must return full logit-selected token IDs for exact commitment replay.")
        original = prepared.probabilities
        eligible = torch.where(prepared.active_mask[0])[0]
        changes = {}
        for offset in range(0, eligible.numel(), 8):
            chunk = eligible[offset:offset + 8]
            p = original[0, chunk].float()
            q = refreshed[0, chunk].float()
            midpoint = (p + q) * 0.5
            tiny = torch.finfo(p.dtype).tiny
            js = 0.5 * (
                (p * (p.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log())).sum(-1)
                + (q * (q.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log())).sum(-1)
            )
            tv = 0.5 * (p - q).abs().sum(-1)
            original_ids = prepared.x0[0, chunk]
            original_prob = p.gather(-1, original_ids[:, None]).squeeze(-1)
            refreshed_prob = q.gather(-1, original_ids[:, None]).squeeze(-1)
            for local, position in enumerate(chunk.tolist()):
                changes[str(position)] = {
                    "flipped": bool(original_ids[local] != refreshed_ids[0, position]),
                    "original_value_probability": float(original_prob[local]),
                    "refreshed_original_value_probability": float(refreshed_prob[local]),
                    "probability_loss": float(original_prob[local] - refreshed_prob[local]),
                    "total_variation": float(tv[local]), "js_divergence": float(js[local]),
                }
        # Core CommitValues is group-shaped semantically: construct universal
        # refreshed ids explicitly so reuse also serves different companions.
        token_ids = refreshed_ids.clone()
        token_ids[0, first] = prepared.x0[0, first]
        confidence = refreshed.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        confidence[0, first] = prepared.confidence[0, first]
        record = {
            "doc_id": doc_id, "state_key": state_key, "first_position": first,
            "first_token_id": int(prepared.x0[0, first]),
            "token_ids": token_ids.tolist(), "confidence": confidence.tolist(),
            "changes": changes, "accounting": self.accounting.delta(before),
            "wall_seconds": time.perf_counter() - started,
        }
        self.store.put_json("probes", key, record)
        return key, record

    def selectors(self, pools, *, state_key, doc_id):
        comparisons = []
        for name, prepared in pools.items():
            with self.accounting.scope("diagnostic_candidate"):
                entropy = self.sampler.select_step(prepared, "entropy_drop")
                cheap = self.sampler.select_step(prepared, "max_confidence")
            entropy_index, cheap_index = int(entropy.best_index[0]), int(cheap.best_index[0])
            left = set(torch.where(entropy.best_mask[0])[0].tolist())
            right = set(torch.where(cheap.best_mask[0])[0].tolist())
            masks = [tuple(record["positions"]) for record in candidate_records(prepared)]
            labels = dict(state_key=state_key, doc_id=doc_id)
            entropy_branch = self.branch(prepared, entropy_index, reuse_label=f"{name}-entropy", **labels)
            cheap_branch = self.branch(prepared, cheap_index, reuse_label=f"{name}-cheap", **labels)
            comparisons.append({
                "pool": name, "entropy_index": entropy_index, "cheap_index": cheap_index,
                "exact_agreement": left == right, "jaccard": len(left & right) / len(left | right),
                "entropy_size": len(left), "cheap_size": len(right),
                "duplicate_candidates": len(masks) - len({tuple(sorted(mask)) for mask in masks}),
                "entropy_branch": entropy_branch, "cheap_branch": cheap_branch,
                "entropy_scores": entropy.scores.tolist(), "cheap_scores": cheap.scores.tolist(),
            })
        return {"comparisons": comparisons}

    def precedence(self, pools, *, state_key, doc_id, absolute_dependency=None):
        probes, groups = [], []
        for name, prepared in pools.items():
            with self.accounting.scope("diagnostic_candidate"):
                entropy = self.sampler.select_step(prepared, "entropy_drop")
                cheap = self.sampler.select_step(prepared, "max_confidence")
            winners = {int(entropy.best_index[0]), int(cheap.best_index[0])}
            for candidate in candidate_records(prepared):
                index = candidate["index"]
                key, probe = self.probe(prepared, index, state_key=state_key, doc_id=doc_id,
                                        reuse_label=f"{name}-{index}-seed")
                seed = candidate["seed"]
                companions = [position for position in candidate["positions"] if position != seed]
                features = []
                for companion in companions:
                    record = directed_features(prepared, seed, companion)
                    record.update(probe["changes"][str(companion)])
                    record.update({"group_size": len(candidate["positions"]),
                                   "refreshed_confidence": probe["confidence"][0][companion]})
                    if absolute_dependency is not None:
                        record["absolute_attention"] = directed_features(prepared, seed, companion,
                                                                          absolute_dependency)
                    features.append(record)
                probes.append({"pool": name, "candidate_index": index, "probe_key": key,
                               "features": features, "any_flip": any(row["flipped"] for row in features),
                               "group_size": len(candidate["positions"])})
                if index not in winners:
                    continue
                variants = [("natural", prepared, index)]
                if len(candidate["positions"]) > 2:
                    variants.append(("seed_first_companion_pair", pair_prepared(prepared, index), 0))
                for label, variant, variant_index in variants:
                    kwargs = dict(state_key=state_key, doc_id=doc_id)
                    prefix = f"{name}-{index}-{label}"
                    simultaneous = self.branch(variant, variant_index, reuse_label=f"{prefix}-sim", **kwargs)
                    refreshed = self.branch(variant, variant_index, mode="seed_first", probe=probe,
                                            reuse_label=f"{prefix}-seed", **kwargs)
                    group = {"pool": name, "candidate_index": index, "label": label,
                             "group_size": int(variant.candidates.candidate_masks[variant_index].sum()),
                             "simultaneous_branch": simultaneous, "seed_first_branch": refreshed,
                             "probe_key": key}
                    kind = candidate["metadata"].get("construction_order_kind_by_batch", ["greedy"])[0]
                    group["construction_order_kind"] = kind
                    group["primary_first_companion_control"] = (
                        label == "seed_first_companion_pair" and kind == "greedy"
                    )
                    if group["group_size"] == 2:
                        reverse_key, reverse_probe = self.probe(
                            variant, variant_index, reverse=True, reuse_label=f"{prefix}-reverse", **kwargs)
                        group["reverse_probe_key"] = reverse_key
                        group["reverse_branch"] = self.branch(
                            variant, variant_index, mode="reverse", probe=reverse_probe,
                            reuse_label=f"{prefix}-reverse", **kwargs)
                    groups.append(group)
        return {"probes": probes, "groups": groups}

    def proposals(self, pools, *, state_key, doc_id):
        results = []
        for name in ("fixed4", "confidence4"):
            prepared = pools[name]
            branches = []
            for candidate in candidate_records(prepared):
                key = self.branch(prepared, candidate["index"], state_key=state_key, doc_id=doc_id,
                                  reuse_label=f"{name}-{candidate['index']}")
                branches.append({"candidate_index": candidate["index"], "branch": key})
            results.append({"pool": name, "branches": branches,
                            "unique_actions": len({row["branch"] for row in branches})})
        return {"pools": results, "proposal_pools": {
            "dependency" if row["pool"] == "fixed4" else "confidence":
            [branch["branch"] for branch in row["branches"]] for row in results
        }}
