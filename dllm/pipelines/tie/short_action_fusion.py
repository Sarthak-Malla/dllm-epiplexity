"""Run short-lookahead diagnostics with python -m dllm.pipelines.tie.eval --model tie."""

from dataclasses import asdict
from itertools import takewhile
import json
from pathlib import Path

import torch

from .action_fusion import (
    ACTION_CHECKPOINTS, action_ranks, numeric_answer_span, score_responses, validate_checkpoints,
)
from .shared_audit import fork


def load_full_rollouts(path, doc_ids):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc = row["doc_id"]
            if row["mode"] != "action_fusion_wide" or doc in rows:
                raise ValueError(f"Unexpected or repeated full rollout for doc {doc}")
            rows[doc] = row
    if sorted(rows) != list(doc_ids):
        raise ValueError("Full-rollout reference must match the declared split")
    return rows


class ShortActionFusionDiagnostic:
    """Compare an active-block lookahead with the saved full continuations."""

    def __init__(self, sampler):
        if [m.name for m in sampler.members] != ["llada", "dream"]:
            raise ValueError("Action fusion requires LLaDA followed by Dream")
        validate_checkpoints(sampler, ACTION_CHECKPOINTS, 2)
        self.sampler = sampler
        answer_block = len(sampler.block_sizes) - 1
        self.answer_schedule = [(block, count) for block, count in sampler.schedule
                                if block == answer_block]

    def peek(self, owner, step, block, slots, ids):
        branch = fork(owner, block, slots, ids)
        for next_block, count in takewhile(
            lambda entry: entry[0] == block, self.sampler.schedule[step:]
        ):
            branch.step(next_block, count)
        for answer_block, count in self.answer_schedule:
            branch.step(answer_block, count)
        completion = branch.completion(self.sampler.config.answer_marker, reasoning_blocks=block + 1)
        return {
            "response": completion.response,
            "forwards": branch.forward_calls - owner.forward_calls,
            "processed_tokens": branch.processed_tokens - owner.processed_tokens,
        }

    def complete_swap(self, owner, step, block, slots, ids):
        branch = fork(owner, block, slots, ids)
        for next_block, count in self.sampler.schedule[step:]:
            branch.step(next_block, count)
        return {
            "response": branch.completion(self.sampler.config.answer_marker).response,
            "forwards": branch.forward_calls - owner.forward_calls,
            "processed_tokens": branch.processed_tokens - owner.processed_tokens,
        }

    def score(self, prompts, candidates, trim, *, tentative):
        marker = self.sampler.config.answer_marker
        delimiter = marker.lstrip()
        spans = {}
        for candidate in candidates:
            text = trim(candidate["response"])
            span = numeric_answer_span(text)
            if tentative and span is not None:
                marker_index = text.rfind(delimiter)
                if marker_index < 0 or span[0] < marker_index + len(delimiter):
                    span = None
            spans[text] = span
        scores, costs = score_responses(self.sampler.members, prompts, marker, spans, answer_spans=spans)
        for candidate in candidates:
            text = trim(candidate["response"])
            span = spans[text]
            candidate["scores"] = scores[text]
            candidate["extracted_answer"] = None if span is None else text[slice(*span)]
        return costs

    @torch.inference_mode()
    def sample(self, prompts, reference, trim):
        owner = self.sampler.initialize(prompts)[0]
        events = []
        for step, (block, count) in enumerate(self.sampler.schedule, 1):
            choose = None
            if step in ACTION_CHECKPOINTS:
                def choose(state, block, offset, canvas, ranked, tokens, confidence,
                           count, step=step):
                    ranks = ranked.new_tensor(action_ranks(count, wide=True)["defer_two"])
                    keep_slots, keep_ids = ranked[:count].tolist(), tokens[:count].tolist()
                    swap_slots, swap_ids = ranked[ranks].tolist(), tokens[ranks].tolist()
                    event = {
                        "step": step, "block": block,
                        "baseline_slots": keep_slots, "baseline_ids": keep_ids,
                        "alternate_slots": swap_slots, "alternate_ids": swap_ids,
                        "short": {
                            "keep": self.peek(state, step, block, keep_slots, keep_ids),
                            "swap": self.peek(state, step, block, swap_slots, swap_ids),
                        },
                    }
                    if reference is None:
                        event["full"] = {
                            "swap": self.complete_swap(state, step, block, swap_slots, swap_ids)
                        }
                    events.append(event)
                    return ranked[:count], tokens[:count]
            owner.step(block, count, choose)

        baseline = asdict(owner.completion(self.sampler.config.answer_marker))
        if reference is not None:
            self.attach_saved_reference(baseline, events, reference)
        else:
            for event in events:
                event["full"]["keep"] = {"response": baseline["response"]}

        short_candidates = [candidate for event in events for candidate in event["short"].values()]
        full_candidates = [candidate for event in events for candidate in event["full"].values()]
        return {
            "baseline": baseline,
            "checkpoints": events,
            "scoring_cost": self.score(prompts, short_candidates, trim, tentative=True),
            "full_reference_scoring_cost": self.score(prompts, full_candidates, trim, tentative=False),
        }

    @staticmethod
    def attach_saved_reference(baseline, events, reference):
        for field in ("response", "reasoning_ids", "answer_ids"):
            recorded = reference["baseline"][field]
            actual = list(baseline[field]) if isinstance(baseline[field], tuple) else baseline[field]
            if actual != recorded:
                raise ValueError(f"Full-rollout baseline differs on {field}")

        full = {event["step"]: event for event in reference["checkpoints"]
                if event["action"] == "defer_two"}
        if sorted(full) != list(ACTION_CHECKPOINTS):
            raise ValueError("Missing defer-two full-rollout checkpoints")
        full_keep = {key: reference["baseline"][key] for key in ("response", "correct")}
        for event in events:
            old = full[event["step"]]
            fields = ("block", "baseline_slots", "baseline_ids", "alternate_slots", "alternate_ids")
            if any(event[field] != old[field] for field in fields):
                raise ValueError(f"Full-rollout action differs at step {event['step']}")
            full_swap = {key: old["alternate"][key] for key in ("response", "correct")}
            event["full"] = {"keep": full_keep, "swap": full_swap}
