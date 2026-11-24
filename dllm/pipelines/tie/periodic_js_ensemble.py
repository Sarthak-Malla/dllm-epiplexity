"""Run the JS pilots with the Slurm wrapper.

    sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh
    sbatch --job-name=ctca-selective-js /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh --selective
"""

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import groupby
import math
from time import perf_counter

import torch

from dllm.pipelines.tse.divergence import jensen_shannon_divergence

from .exact_text_fusion import (
    ExactTextFusion,
    MAX_CONDITIONAL_FORWARDS,
    PEER_TOP_K,
    mix_projected,
    peer_canvas,
    surrounding_text,
)
from .relay import RelayError, transfer_blocks


MODE = "periodic_js_ensemble"
SELECTIVE_MODE = "selective_js_ensemble"
ROUTING_CONFIDENCE_THRESHOLD = 0.80
ROUND_STEPS = 2
MAX_PATH_WIDTH = 2
MAX_CONDITIONAL_PER_ROUND = 4
MIN_COMMON_MASS = 0.50
MAX_NORMALIZED_JS = 0.20


@dataclass(frozen=True)
class OwnerWrite:
    slot: int
    token: int
    probabilities: torch.Tensor
    context: tuple[list[int], list[int]]


def common_support_js(base, projected):
    """Compare certified common events plus one bin for the remaining mass."""
    ids = list(projected)
    owner = base[ids].float()
    peer = base.new_tensor(list(projected.values()), dtype=torch.float32)
    mass_owner, mass_peer = owner.sum(), peer.sum()
    p = torch.cat((owner, (1 - mass_owner).clamp_min(0).view(1)))
    q = torch.cat((peer, (1 - mass_peer).clamp_min(0).view(1)))
    # The full events sum to one. The division handles overflow from roundoff.
    value = jensen_shannon_divergence((p / p.sum())[None], (q / q.sum())[None]).item()
    return min(1.0, value / math.log(2)), float(mass_owner), float(mass_peer)


def fusion_decision(write, projected, *, gated):
    """Return the mass-preserving candidate and its eligibility diagnostics."""
    divergence, owner_mass, peer_mass = common_support_js(write.probabilities, projected)
    candidate = int(mix_projected(write.probabilities, list(projected), list(projected.values())).argmax())
    if min(owner_mass, peer_mass) < MIN_COMMON_MASS:
        reason = "low_coverage"
    elif gated and divergence > MAX_NORMALIZED_JS:
        reason = "high_divergence"
    else:
        reason = "fused"
    return candidate, {"normalized_js": divergence, "owner_common_mass": owner_mass,
                       "peer_common_mass": peer_mass, "decision": reason}


class PeriodicJSEnsemble:
    """Reconcile the peer's native drafts with the main model's two-step rounds."""

    def __init__(self, sampler, owner_index=0):
        self.sampler = sampler
        self.owner_index = owner_index
        self.projection = ExactTextFusion(sampler, owner_index=owner_index)
        self.owner = self.projection.owner
        self.peer = self.projection.peer

    def peer_draft(self, template, native, spans, counts, stats):
        positions = sorted({position for span in spans.values() for position in span})
        pending = native.new_tensor(positions)
        draft = native.clone()
        for count in counts:
            if not count or not pending.numel():
                continue
            probabilities = self.projection.probabilities(template.member, draft, pending, stats)
            stats["peer_draft_forwards"] += 1
            confidence, tokens = probabilities.max(dim=1)
            ranked = confidence.argsort(descending=True, stable=True)
            take = min(count, pending.numel())
            chosen = ranked[:take]
            draft[pending[chosen]] = tokens[chosen]
            pending = pending[ranked[take:]]
            stats["peer_private_writes"] += take
        return draft

    def single_paths(self, write, peer_row, private):
        ids = {token for token, _ in self.projection.top_ids(peer_row, PEER_TOP_K)}
        owner_ids = [token for token, _ in self.projection.top_ids(write.probabilities, PEER_TOP_K)]
        for token in owner_ids:
            path = self.projection.owner_to_peer(token)
            if len(path) == 1 and path[0] < peer_row.numel():
                ids.add(path[0])
        if private[0] not in self.peer.special_ids and private[0] < peer_row.numel():
            ids.add(private[0])
        ids = sorted(ids)
        return [((token,), probability) for token, probability in zip(ids, peer_row[ids].tolist())]

    def project_round(self, template, blocks, block, proposals, writes, counts, stats, observations, gated):
        try:
            peer_ids, spans = peer_canvas(
                self.owner, self.peer, blocks, block, proposals,
                template.prompt.tolist(), self.projection.peer_marker,
            )
        except RelayError:
            stats["incompatible_rounds"] += 1
            return {}
        if not template.member.fits_context(len(peer_ids)):
            stats["overlength_rounds"] += 1
            return {}
        eligible = [write for write in writes if write.slot in spans and len(spans[write.slot]) <= MAX_PATH_WIDTH
                    and write.token not in self.owner.special_ids]
        stats["eligible_positions"] += len(eligible)
        if not eligible:
            return {}
        targets = [position for write in eligible for position in spans[write.slot]]
        if len(set(targets)) != len(targets):
            stats["overlapping_rounds"] += 1
            return {}
        native = template.prompt.new_tensor(peer_ids)
        draft = self.peer_draft(template, native, spans, counts, stats)
        private_paths = {write.slot: draft[list(spans[write.slot])].tolist() for write in eligible}
        # Retain the peer's other private writes as context. Hide the target spans
        # so a score cannot be conditioned on its own answer or another fill
        # being adjudicated at this synchronization.
        draft[targets] = self.peer.mask_id
        starts = [spans[write.slot][0] for write in eligible]
        first_rows = self.projection.probabilities(template.member, draft, starts, stats)
        stats["peer_verification_forwards"] += 1
        draft_ids = draft.tolist()
        initial = stats["conditional_forwards"]
        changes = {}
        for write, peer_row in zip(eligible, first_rows):
            span = spans[write.slot]
            private = private_paths[write.slot]
            if len(span) == 1:
                paths = self.single_paths(write, peer_row, private)
            else:
                remaining = MAX_CONDITIONAL_PER_ROUND - (stats["conditional_forwards"] - initial)
                if not remaining:
                    stats["conditional_budget_skips"] += 1
                    continue
                owner_path = self.projection.owner_to_peer(write.token)
                priority = (private, owner_path) if len(owner_path) == 2 else (private,)
                paths = self.projection.two_token_paths(
                    template.member, draft, span, peer_row, write.probabilities,
                    min(MAX_CONDITIONAL_FORWARDS, remaining), stats,
                    priority_paths=priority,
                )
            projected = self.projection.project(
                paths, write.context, surrounding_text(draft_ids, span[0], span[-1] + 1, self.peer),
                write.probabilities.numel(),
            )
            if not projected:
                stats["empty_common_support"] += 1
                continue
            candidate, evidence = fusion_decision(write, projected, gated=gated)
            reason = evidence["decision"]
            stats[reason] += 1
            stats["scored_positions"] += 1
            stats["scored_two_token_paths"] += len(span) == 2
            peer_best = max(projected, key=projected.__getitem__)
            observations.append({"round": stats["rounds"], "block": block, "slot": write.slot,
                                 "peer_width": len(span), "owner_token": write.token, "projected_peer_top": peer_best,
                                 "fused_candidate": candidate, "common_candidates": len(projected),
                                 "owner_support_for_owner_token": float(write.probabilities[write.token]),
                                 "peer_support_for_owner_token": projected.get(write.token),
                                 "owner_support_for_fused_candidate": float(write.probabilities[candidate]),
                                 "peer_support_for_fused_candidate": projected.get(candidate), **evidence})
            if reason == "fused" and candidate != write.token:
                changes[write.slot] = candidate
        return changes

    def generate(self, state, template, *, gated, confidence_threshold=None):
        if confidence_threshold is not None and not 0 <= confidence_threshold <= 1:
            raise ValueError("The routing confidence threshold must be between 0 and 1")
        stats, observations, routing = Counter(), [], []
        for block, schedule in groupby(self.sampler.schedule, key=lambda item: item[0]):
            counts = [count for _, count in schedule]
            if block == len(state.blocks) - 1:
                for count in counts:
                    state.step(block, count)
                continue
            for start in range(0, len(counts), ROUND_STEPS):
                group = counts[start:start + ROUND_STEPS]
                blocks = state.snapshot()
                proposals = {}
                writes = []
                min_confidence = 1.0

                def record(_, _block_index, offset, canvas, ranked, tokens, confidence, count, *, probabilities):
                    nonlocal min_confidence
                    if not proposals:
                        proposals.update(zip(ranked.tolist(), tokens.tolist()))
                    if confidence_threshold is not None and count:
                        min_confidence = min(min_confidence, float(confidence[count - 1]))
                    native = canvas.tolist()
                    for row, (slot, token) in enumerate(zip(ranked[:count].tolist(), tokens[:count].tolist())):
                        context = surrounding_text(native, offset + slot, offset + slot + 1, self.owner)
                        writes.append(OwnerWrite(slot, token, probabilities[row].clone(), context))
                    return ranked[:count], tokens[:count]

                for count in group:
                    state.step(block, count, record, include_probabilities=True)
                stats["rounds"] += 1
                stats["scheduled_positions"] += len(writes)
                before_peer = stats["peer_forwards"]
                before_changes = stats["token_changes"]
                consult = bool(writes) and (
                    confidence_threshold is None or min_confidence < confidence_threshold
                )
                if consult:
                    first_observation = len(observations)
                    changes = self.project_round(template, blocks, block, proposals, writes, group,
                                                 stats, observations, gated)
                    if changes:
                        synchronized = state.snapshot()
                        for slot, token in changes.items():
                            synchronized[block][slot] = token
                        try:
                            transfer_blocks(synchronized, self.owner, self.peer)
                        except RelayError:
                            stats["rejected_synchronizations"] += 1
                            changes = {}
                    for slot, token in changes.items():
                        state.blocks[block][slot] = token
                    for observation in observations[first_observation:]:
                        observation["written_token"] = changes.get(observation["slot"], observation["owner_token"])
                    stats["token_changes"] += len(changes)
                if confidence_threshold is not None:
                    stats["routed_rounds" if consult else "skipped_rounds"] += 1
                    routing.append({"round": stats["rounds"], "block": block,
                                    "min_owner_confidence": min_confidence if writes else None,
                                    "routed": consult, "peer_forwards": stats["peer_forwards"] - before_peer,
                                    "token_changes": stats["token_changes"] - before_changes})
        details = {"stats": dict(stats), "positions": observations}
        if confidence_threshold is not None:
            details.update(confidence_threshold=confidence_threshold, routing=routing)
        return asdict(state.completion(self.sampler.config.answer_marker)), details

    @torch.inference_mode()
    def sample(self, prompts, *, confidence_threshold=None):
        selective = confidence_threshold is not None
        branches, diagnostics, seconds = {}, {}, {}
        for state in self.sampler.initialize(prompts):
            start = perf_counter()
            for block, count in self.sampler.schedule:
                state.step(block, count)
            branches[state.member.name] = asdict(state.completion(self.sampler.config.answer_marker))
            seconds[state.member.name] = perf_counter() - start
        arms = (
            (("periodic_js", True, None), ("selective_js", True, confidence_threshold))
            if selective else (("periodic_fusion", False, None), ("periodic_js", True, None))
        )
        for arm, gated, threshold in arms:
            states = self.sampler.initialize(prompts)
            state, template = states[self.owner_index], states[1 - self.owner_index]
            start = perf_counter()
            branches[arm], diagnostics[arm] = self.generate(
                state, template, gated=gated, confidence_threshold=threshold,
            )
            seconds[arm] = perf_counter() - start
        result = {"branches": branches, "fusion_diagnostics": diagnostics}
        if selective:
            result["generation_seconds"] = seconds
        return result
