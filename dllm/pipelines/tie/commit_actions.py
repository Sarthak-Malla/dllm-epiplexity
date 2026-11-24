"""Run commit-action experiments with python -m dllm.pipelines.tie.eval."""

from copy import deepcopy
from dataclasses import asdict, replace
import math

import torch

from .relay import RelayError, transfer_blocks


def gap_candidates(block, ranked, tokens, mask_id, special, count, scheduling):
    """Find complete masked runs with ordinary text on both sides."""
    limit = 2 * count if scheduling else count
    ranks = {slot: index for index, slot in enumerate(ranked[:limit])}
    proposed = dict(zip(ranked, tokens))
    gaps = []
    for start in range(1, len(block) - 1):
        if block[start] != mask_id or block[start - 1] in special:
            continue
        end = start
        while end < len(block) and block[end] == mask_id:
            end += 1
        gap = tuple(range(start, end))
        if (len(gap) > min(2, count) or end == len(block) or block[end] in special
                or any(slot not in ranks or proposed[slot] in special for slot in gap)):
            continue
        if scheduling and all(ranks[slot] < count for slot in gap):
            continue
        gaps.append(gap)
    return sorted(gaps, key=lambda gap: (-len(gap), max(ranks[i] for i in gap), gap[0]))


def contextual_ids(vocab, left, text, right):
    """Require native token boundaries and exact decoded text across the gap."""
    left_ids, ids, right_ids = map(vocab.encode, (left, text, right))
    joint = vocab.encode(left + text + right)
    if (not ids or any(token in vocab.special_ids for token in joint)
            or joint != left_ids + ids + right_ids
            or vocab.decode(joint) != left + text + right):
        raise RelayError("candidate failed contextual boundary check")
    return ids


class CommitExperiment:
    """Run one intervention and paired continuations on the LLaDA scaffold."""

    def __init__(self, sampler, mode, thresholds=None, *, prefer=None):
        self.sampler = sampler
        self.mode = mode
        self.thresholds = thresholds or {}
        self.prefer = prefer

    @torch.inference_mode()
    def sample(self, prompts, doc_id):
        self.cost = {m.name: [0, 0] for m in self.sampler.members}
        self.record = {"doc_id": doc_id, "attempts": [], "branches": {}}
        self.forks = {}
        self.attempted = False
        self.dream_prompt = list(prompts[1])
        master = self.sampler.initialize(prompts)[0]
        start = 16 + 16 * (doc_id % 4)
        for step, (block, count) in enumerate(self.sampler.schedule, 1):
            self.step_number = step
            choose = self.choose if step >= start and not self.attempted else None
            if self.mode == "ctca":
                choose = self.choose
            master.step(block, count, choose)
        base = master.completion(self.sampler.config.answer_marker)
        self.record["branches"]["keep"] = asdict(base)
        for name, state in self.forks.items():
            for block, count in self.sampler.schedule[self.intervention_step:]:
                state.step(block, count)
            self.record["branches"][name] = asdict(
                state.completion(self.sampler.config.answer_marker)
            )
        self.record["scoring_cost"] = self.cost
        return self.record

    def logp(self, member, canvas, positions):
        self.cost[member.name][0] += 1
        self.cost[member.name][1] += len(canvas)
        return member.log_probabilities(canvas, canvas.new_tensor(positions))

    def path_score(self, member, canvas, offset, ids):
        canvas = canvas.clone()
        score = 0.0
        for position, token in enumerate(ids, offset):
            score += float(self.logp(member, canvas, [position])[0, token])
            canvas[position] = token
        return score

    def prepare(self, state, block_index, canvas, offset, gap, proposed, exact=False):
        master, dream = self.sampler.members
        source, target = master.vocabulary, dream.vocabulary
        block = state.blocks[block_index].tolist()
        lo, hi = gap[0], gap[-1] + 1
        left_start, right_end = lo - 1, hi + 1
        while left_start and block[left_start - 1] not in source.special_ids:
            left_start -= 1
        while right_end < len(block) and block[right_end] not in source.special_ids:
            right_end += 1
        left_ids, right_ids = block[left_start:lo], block[hi:right_end]
        left, right = source.decode(left_ids), source.decode(right_ids)
        ids_a = [proposed[i] for i in gap]
        a = source.decode(ids_a)
        if not a or "\ufffd" in left + a + right:
            raise RelayError("candidate includes empty or incomplete text")
        if contextual_ids(source, left, a, right) != ids_a:
            raise RelayError("owner baseline is not a canonical contextual encoding")
        foreign_a = contextual_ids(target, left, a, right)
        if len(foreign_a) > 4 or (exact and len(foreign_a) != 1):
            raise RelayError("peer gap width is outside the configured budget")
        blocks = state.snapshot()
        # Split the active block at the gap. The transfer keeps all other masks.
        blocks[block_index:block_index + 1] = [block[:lo], block[lo:hi], block[hi:]]
        foreign = transfer_blocks(blocks, source, target)
        if foreign[block_index][-len(target.encode(left)):] != target.encode(left):
            raise RelayError("left anchor changed in peer canvas")
        if foreign[block_index + 2][:len(target.encode(right))] != target.encode(right):
            raise RelayError("right anchor changed in peer canvas")
        marker = target.encode(self.sampler.config.answer_marker)
        dream_offset = len(self.dream_prompt) + sum(map(len, foreign[:block_index + 1]))
        foreign[block_index + 1] = [target.mask_id] * len(foreign_a)
        dream_ids = self.dream_prompt + sum(foreign[:-1], []) + marker + foreign[-1]
        if not dream.fits_context(len(dream_ids)):
            raise RelayError("peer canvas exceeds context limit")
        native_canvas = torch.tensor(dream_ids, device=dream.device)
        if exact:
            base = self.logp(master, canvas, [offset + lo])[0].exp()
            peer = self.logp(dream, native_canvas, [dream_offset])[0].exp()
            aligned = torch.zeros_like(base)
            for token in peer.topk(min(32, peer.numel())).indices.tolist():
                if token in target.special_ids:
                    continue
                text = target.decode([token])
                try:
                    mapped = contextual_ids(source, left, text, right)
                    peer_ids = contextual_ids(target, left, text, right)
                except RelayError:
                    continue
                if peer_ids == [token] and len(mapped) == 1 and mapped[0] not in source.special_ids:
                    aligned[mapped[0]] += peer[token].to(aligned.device)
            ids_b = [int(((1 - 0.5 * aligned.sum()) * base + 0.5 * aligned).argmax())]
            return {"a": a, "b": source.decode(ids_b), "ids_a": ids_a, "ids_b": ids_b,
                    "retained_mass": float(aligned.sum())}
        draft = native_canvas.clone()
        foreign_b = []
        dream_score_b = 0.0
        for position in range(dream_offset, dream_offset + len(foreign_a)):
            logp = self.logp(dream, draft, [position])[0]
            token = int(logp.argmax())
            if token in target.special_ids:
                raise RelayError("peer proposed a special token")
            dream_score_b += float(logp[token])
            foreign_b.append(token)
            draft[position] = token
        b = target.decode(foreign_b)
        ids_b = contextual_ids(source, left, b, right)
        if len(ids_b) != len(ids_a) or contextual_ids(target, left, b, right) != foreign_b:
            raise RelayError("candidate changed native gap width or tokenization")
        dream_score_a = (None if self.mode == "llada_verifier" else
                         self.path_score(dream, native_canvas, dream_offset, foreign_a))
        master_a = self.path_score(master, canvas, offset + lo, ids_a)
        master_b = self.path_score(master, canvas, offset + lo, ids_b)
        scores = (dream_score_b, master_a, master_b)
        if not all(math.isfinite(score) for score in scores) or (
            dream_score_a is not None and not math.isfinite(dream_score_a)
        ):
            raise RelayError("non-finite native path score")
        return {"a": a, "b": b, "ids_a": ids_a, "ids_b": ids_b,
                "dream_ids_a": foreign_a, "dream_ids_b": foreign_b,
                "dream_margin": dream_score_b - dream_score_a if dream_score_a is not None else None,
                "llada_margin": master_b - master_a,
                "dream_confidence": (
                    torch.tensor(dream_score_a / len(foreign_a)).exp().item()
                    if dream_score_a is not None else None
                )}

    def choose(self, state, block_index, offset, canvas, ranked, tokens, confidence, count):
        base_slots, base_values = ranked[:count], tokens[:count]
        if self.prefer is not None:
            return self.prefer(self, state, block_index, offset, canvas, ranked, tokens, confidence, count)
        if block_index == len(state.blocks) - 1:
            return base_slots, base_values
        ranked_ids = ranked.tolist()
        proposed = dict(zip(ranked_ids, tokens.tolist()))
        scheduling = self.mode in {"schedule_diagnostic", "positions", "joint"}
        gaps = gap_candidates(state.blocks[block_index].tolist(), ranked_ids,
                              tokens.tolist(), state.member.vocabulary.mask_id,
                              state.member.vocabulary.special_ids, count, scheduling)
        if self.mode == "exact":
            gaps = [gap for gap in gaps if len(gap) == 1]
        if not gaps:
            return base_slots, base_values
        gap = gaps[0]
        self.attempted = True
        self.intervention_step = self.step_number
        event = {"step": self.step_number, "block": block_index, "gap": gap,
                 "default_slots": base_slots.tolist(), "default_ids": base_values.tolist(),
                 "prefix": state.snapshot()}
        self.record["attempts"].append(event)
        try:
            candidate = self.prepare(state, block_index, canvas, offset, gap, proposed,
                                     exact=self.mode == "exact")
        except RelayError as error:
            event["rejected"] = str(error)
            return base_slots, base_values
        event.update(candidate)
        if scheduling:
            promoted = list(gap) + [slot for slot in ranked_ids if slot not in gap][:count-len(gap)]
            displaced = set(base_slots.tolist()) - set(promoted)
            cutoff = max(float(confidence[ranked_ids.index(i)]) for i in displaced)
            event["schedule_margin"] = event["dream_confidence"] - cutoff
        else:
            promoted = base_slots.tolist()
        event["candidate_slots"] = promoted

        if self.mode == "diagnostic":
            branches = {"replace": (promoted, True)}
        elif self.mode == "schedule_diagnostic":
            branches = {"positions": (promoted, False), "joint": (promoted, True)}
        else:
            if not scheduling or event["schedule_margin"] >= self.thresholds["schedule"]:
                use_peer = self.mode == "exact" or (
                    self.mode == "llada_verifier"
                    and event["llada_margin"] >= self.thresholds["llada_only"]
                ) or (
                    self.mode in {"text", "joint"}
                    and event["dream_margin"] >= self.thresholds["dream"]
                    and event["llada_margin"] >= -self.thresholds["veto"]
                )
                branches = {self.mode: (promoted, use_peer)}
            else:
                branches = {}
        updates = dict(zip(gap, candidate["ids_b"]))
        for name, (slots, use_peer) in branches.items():
            values = [updates[i] if use_peer and i in updates else proposed[i] for i in slots]
            if slots == base_slots.tolist() and values == base_values.tolist():
                continue
            branch = replace(state, blocks=[block.clone() for block in state.blocks],
                             history=deepcopy(state.history))
            branch.blocks[block_index][ranked.new_tensor(slots)] = tokens.new_tensor(values)
            self.forks[name] = branch
            event.setdefault("writes", {})[name] = [slots, values]
        return base_slots, base_values
