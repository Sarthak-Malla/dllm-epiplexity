"""Run the shared-canvas audit with python -m dllm.pipelines.tie.eval."""

from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from types import SimpleNamespace

import torch

from .commit_actions import CommitExperiment, contextual_ids, gap_candidates
from .relay import RelayError, transfer_blocks


CHECKPOINTS = (12, 44, 76)
TEXT_GAPS_PER_CHECKPOINT = 2


def alternative_positions(ranked, peer_slots, count):
    """Build an owner-only alternative with the same number of displaced slots."""
    baseline = ranked[:count]
    swaps = len(set(peer_slots) - set(baseline))
    if not swaps:
        return []
    return baseline[:count - swaps] + ranked[count:count + swaps]


def score_difference(scores, alternative, baseline):
    return sum(scores[i] for i in alternative) - sum(scores[i] for i in baseline)


def clone_state(state):
    return replace(state, blocks=[part.clone() for part in state.blocks],
                   history=deepcopy(state.history))


def fork(state, block, slots, ids):
    other = clone_state(state)
    canvas = other.blocks[block]
    canvas[canvas.new_tensor(slots)] = canvas.new_tensor(ids)
    return other


class SharedCanvasAudit:
    """Compare bounded actions on snapshots from each independent trajectory."""

    def __init__(self, sampler):
        self.sampler = sampler
        remaining = dict(enumerate(sampler.block_sizes))
        for step, (block, count) in enumerate(sampler.schedule, 1):
            if step in CHECKPOINTS and (
                remaining[block] <= count or block == len(sampler.block_sizes) - 1
            ):
                raise ValueError(
                    f"Shared-audit checkpoint {step} requires a choice of reasoning positions"
                )
            remaining[block] -= count

    @torch.inference_mode()
    def sample(self, prompts):
        owners = {}
        for index, state in enumerate(self.sampler.initialize(prompts)):
            peer = self.sampler.members[1 - index]
            helper = CommitExperiment(
                SimpleNamespace(members=(state.member, peer), config=self.sampler.config),
                "diagnostic",
            )
            helper.dream_prompt = list(prompts[1 - index])
            helper.cost = {member.name: [0, 0] for member in self.sampler.members}
            checkpoints = []
            for step, (block, count) in enumerate(self.sampler.schedule, 1):
                choose = None
                if step in CHECKPOINTS and block < len(state.blocks) - 1:
                    def choose(*args, step=step):
                        event = self.probe(helper, step, *args)
                        checkpoints.append(event)
                        _, _, _, _, slots, ids, _, count = args
                        return slots[:count], ids[:count]
                state.step(block, count, choose)
            owners[state.member.name] = {
                "baseline": asdict(state.completion(self.sampler.config.answer_marker)),
                "checkpoints": checkpoints,
                "proposal_scoring_cost": helper.cost,
            }
        return {"owners": owners}

    def probe(self, helper, step, state, block, offset, canvas,
              ranked, tokens, confidence, count):
        owner, peer = helper.sampler.members
        slots, token_ids = ranked.tolist(), tokens.tolist()
        proposed = dict(zip(slots, token_ids))
        original = slots[:count]
        event = {"step": step, "block": block, "default_slots": original,
                 "default_ids": token_ids[:count], "draft": state.snapshot(),
                 "text_checks": [], "alternatives": {}}
        try:
            native, positions = self.peer_view(helper, state, block, slots)
        except RelayError as error:
            event["unaligned"] = str(error)
            return event

        peer_logp = helper.logp(peer, native, positions)
        peer_scores, peer_ids = peer_logp.max(-1)
        if not torch.isfinite(peer_scores).all():
            event["unaligned"] = "peer did not produce finite native predictions"
            return event
        owner_scores = dict(zip(slots, confidence.clamp_min(1e-30).log().tolist()))
        peer_scores = dict(zip(slots, peer_scores.tolist()))
        peer_order = sorted(slots, key=peer_scores.__getitem__, reverse=True)[:count]
        event["peer_slots"] = peer_order
        event["aligned_mask_slots"] = slots
        event["peer_top_ids"] = peer_ids.tolist()
        event["owner_top_ids"] = token_ids
        event["common_selected_slots"] = len(set(original) & set(peer_order))
        event["order_disagrees"] = set(original) != set(peer_order)
        if event["order_disagrees"]:
            for label, positions in (
                ("peer_positions", peer_order),
                ("owner_positions", alternative_positions(slots, peer_order, count)),
            ):
                if not positions:
                    continue
                features = {
                    "owner_score": score_difference(owner_scores, positions, original),
                    "peer_score": score_difference(peer_scores, positions, original),
                }
                ids = [proposed[i] for i in positions]
                event["alternatives"][label] = self.replay(
                    state, step, block, positions, ids, features
                )

        source = owner.vocabulary
        gaps = gap_candidates(
            state.blocks[block].tolist(), slots, token_ids, source.mask_id,
            source.special_ids, count, False,
        )
        for gap in gaps[:TEXT_GAPS_PER_CHECKPOINT]:
            try:
                candidate = helper.prepare(state, block, canvas, offset, gap, proposed)
            except RelayError as error:
                event["text_checks"].append({"gap": gap, "rejected": str(error)})
                continue
            event["text_checks"].append({
                "gap": gap, "a": candidate["a"], "b": candidate["b"],
                "owner_ids_a": candidate["ids_a"], "owner_ids_b": candidate["ids_b"],
                "peer_ids_a": candidate["dream_ids_a"], "peer_ids_b": candidate["dream_ids_b"],
                "owner_margin": candidate["llada_margin"], "peer_margin": candidate["dream_margin"],
            })
            if candidate["ids_b"] == candidate["ids_a"]:
                continue
            update = dict(zip(gap, candidate["ids_b"]))
            event["alternatives"]["peer_text"] = self.replay(
                state, step, block, original, [update.get(i, proposed[i]) for i in original],
                {"owner_score": candidate["llada_margin"],
                 "peer_score": candidate["dream_margin"]},
            )
            alternative = self.owner_text(helper, state, block, offset, canvas, gap, proposed)
            if alternative is not None:
                ids, score = alternative
                update = dict(zip(gap, ids))
                event["alternatives"]["owner_text"] = self.replay(
                    state, step, block, original, [update.get(i, proposed[i]) for i in original],
                    {"owner_score": score},
                )
            break
        return event

    def peer_view(self, helper, state, block, slots):
        owner, peer = helper.sampler.members
        src, dst = owner.vocabulary, peer.vocabulary
        blocks = transfer_blocks(state.snapshot(), src, dst)
        offsets = [i for i, token in enumerate(blocks[block]) if token == dst.mask_id]
        if len(offsets) != len(slots):
            raise RelayError("transfer changed the number of active masks")
        mapping = dict(zip(sorted(slots), offsets))
        start = len(helper.dream_prompt) + sum(map(len, blocks[:block]))
        marker = dst.encode(self.sampler.config.answer_marker)
        ids = helper.dream_prompt + sum(blocks[:-1], []) + marker + blocks[-1]
        if not peer.fits_context(len(ids)):
            raise RelayError("retokenized draft exceeds peer context")
        native = torch.tensor(ids, device=peer.device, dtype=torch.long)
        return native, [start + mapping[i] for i in slots]

    def owner_text(self, helper, state, block_index, offset, canvas, gap, proposed):
        owner = state.member
        source = owner.vocabulary
        block = state.blocks[block_index].tolist()
        lo, hi = gap[0], gap[-1] + 1
        left_start, right_end = lo - 1, hi + 1
        while left_start and block[left_start - 1] not in source.special_ids:
            left_start -= 1
        while right_end < len(block) and block[right_end] not in source.special_ids:
            right_end += 1
        left, right = source.decode(block[left_start:lo]), source.decode(block[hi:right_end])
        original = [proposed[i] for i in gap]
        logp = helper.logp(owner, canvas, [offset + lo])[0]
        for token in logp.topk(min(8, logp.numel())).indices.tolist():
            if token == original[0] or token in source.special_ids:
                continue
            ids = [token, *original[1:]]
            try:
                if contextual_ids(source, left, source.decode(ids), right) != ids:
                    continue
            except RelayError:
                continue
            delta = (helper.path_score(owner, canvas, offset + lo, ids)
                     - helper.path_score(owner, canvas, offset + lo, original))
            if isfinite(delta):
                return ids, delta
        return None

    def replay(self, state, step, block, slots, ids, features):
        alternative = fork(state, block, slots, ids)
        for next_block, count in self.sampler.schedule[step:]:
            alternative.step(next_block, count)
        return {"slots": slots, "ids": ids, **features,
                "completion": asdict(alternative.completion(self.sampler.config.answer_marker))}
