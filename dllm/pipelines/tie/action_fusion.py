"""Run the action-fusion diagnostic with python -m dllm.pipelines.tie.eval."""

from dataclasses import asdict
from math import isfinite
from string import punctuation

import torch

from dllm.core.eval.running_metrics import gsm8k_answer_span

from .commit_actions import contextual_ids
from .relay import RelayError
from .shared_audit import fork


ACTION_CHECKPOINTS = (12, 60, 108)


def numeric_answer_span(text):
    span = gsm8k_answer_span(text)
    if span is not None and any(char.isdigit() for char in text[slice(*span)]):
        return span
    return None


def validate_checkpoints(sampler, checkpoints, minimum):
    remaining = list(sampler.block_sizes)
    for step, (block, count) in enumerate(sampler.schedule, 1):
        if step in checkpoints and (
            count < minimum or remaining[block] < count + minimum or block == len(remaining) - 1
        ):
            raise ValueError(f"Action-fusion checkpoint {step} needs a choice of slots")
        remaining[block] -= count
    if len(sampler.schedule) < max(checkpoints):
        raise ValueError("The schedule does not reach all action-fusion checkpoints")


def action_ranks(count, wide):
    """Use zero-based LLaDA ranks to define alternatives to the default top ranks."""
    actions = {"defer_last": (*range(count - 1), count)}
    if wide:
        actions["defer_first"] = (*range(1, count), count)
        actions["defer_two"] = (*range(count - 2), count, count + 1)
    return actions


def answer_reconstruction(member, prompt, marker, response, cost):
    """Score the candidate's native answer tokens with the answer masked."""
    prefix, delimiter, answer = response.rpartition(marker.lstrip())
    if not delimiter or not answer.strip():
        return None
    return text_reconstruction(member, prompt, prefix + delimiter, answer, "", cost)


def shared_prefix_length(first, second):
    return next((i for i, pair in enumerate(zip(first, second)) if pair[0] != pair[1]),
                min(len(first), len(second)))


def native_answer_span(vocabulary, left, answer, right):
    """Cover an extracted answer with canonical tokens, including adjacent formatting."""
    text = left + answer + right
    ids = vocabulary.encode(text)
    if any(token in vocabulary.special_ids for token in ids) or vocabulary.decode(ids) != text:
        raise RelayError("response failed native tokenization round trip")
    start = shared_prefix_length(ids, vocabulary.encode(left))
    suffix = shared_prefix_length(ids[start:][::-1], vocabulary.encode(right)[::-1])
    stop = len(ids) - suffix
    visible_left = vocabulary.decode(ids[:start])
    visible_right = vocabulary.decode(ids[stop:])
    if start >= stop or not left.startswith(visible_left) or not right.endswith(visible_right):
        raise RelayError("native token span does not cover the extracted answer")
    boundary = left[len(visible_left):] + right[:len(right) - len(visible_right)]
    if any(not char.isspace() and char not in punctuation for char in boundary):
        raise RelayError("native answer span would mask visible reasoning")
    return ids[:start], ids[start:stop], ids[stop:]


def text_reconstruction(member, prompt, left, answer, right, cost, *, extracted=False):
    """Mask a verified native encoding of a text span and score its reconstruction."""
    if "\ufffd" in left + answer + right:
        return None
    vocabulary = member.vocabulary
    try:
        if extracted:
            before, targets, after = native_answer_span(vocabulary, left, answer, right)
        else:
            targets = contextual_ids(vocabulary, left, answer, right)
            before, after = vocabulary.encode(left), vocabulary.encode(right)
    except RelayError:
        return None
    visible = list(prompt) + before
    stop = len(visible) + len(targets)
    ids = visible + [vocabulary.mask_id] * len(targets) + after
    if not member.fits_context(len(ids)):
        return None
    canvas = torch.tensor(ids, device=member.device)
    positions = torch.arange(len(visible), stop, device=member.device)
    logp = member.log_probabilities(canvas, positions)
    correct_ids = canvas.new_tensor(targets).unsqueeze(1).to(logp.device)
    score = float(logp.gather(1, correct_ids).mean())
    cost[0] += 1
    cost[1] += len(ids)
    return score if isfinite(score) else None


@torch.inference_mode()
def score_responses(members, prompts, marker, responses, *, answer_spans=None):
    """Score each distinct response text once under each verifier."""
    costs = {member.name: [0, 0] for member in members}
    scores = {response: {} for response in responses}
    for member, prompt in zip(members, prompts):
        for response, by_model in scores.items():
            cost = costs[member.name]
            if answer_spans is None:
                score = answer_reconstruction(member, prompt, marker, response, cost)
            elif (span := answer_spans[response]) is not None:
                start, stop = span
                score = text_reconstruction(
                    member, prompt, response[:start], response[start:stop], response[stop:], cost,
                    extracted=True,
                )
            else:
                score = None
            by_model[member.name] = score
    return scores, costs


class ActionFusionDiagnostic:
    """Generate paired LLaDA actions and score both texts with each verifier."""

    def __init__(self, sampler, *, wide=False):
        if [member.name for member in sampler.members] != ["llada", "dream"]:
            raise ValueError("Action fusion requires LLaDA followed by Dream")
        self.sampler = sampler
        self.wide = wide
        validate_checkpoints(sampler, ACTION_CHECKPOINTS, 2 if wide else 1)

    @torch.inference_mode()
    def sample(self, prompts):
        owner = self.sampler.initialize(prompts)[0]
        marker = self.sampler.config.answer_marker
        events = []
        for step, (block, count) in enumerate(self.sampler.schedule, 1):
            choose = None
            if step in ACTION_CHECKPOINTS:
                def choose(state, block, offset, canvas, ranked, tokens, confidence,
                           count, step=step):
                    baseline = ranked[:count]
                    for action, ranks in action_ranks(count, self.wide).items():
                        indices = ranked.new_tensor(ranks)
                        slots, values = ranked[indices].tolist(), tokens[indices].tolist()
                        branch = fork(state, block, slots, values)
                        for next_block, next_count in self.sampler.schedule[step:]:
                            branch.step(next_block, next_count)
                        events.append({
                            "step": step,
                            "block": block,
                            "action": action,
                            "baseline_slots": baseline.tolist(),
                            "baseline_ids": tokens[:count].tolist(),
                            "alternate_slots": slots,
                            "alternate_ids": values,
                            "rollout_forwards": branch.forward_calls - state.forward_calls,
                            "rollout_tokens": branch.processed_tokens - state.processed_tokens,
                            "alternate": asdict(branch.completion(marker)),
                        })
                    return baseline, tokens[:count]
            owner.step(block, count, choose)

        baseline = asdict(owner.completion(marker))
        candidates = [baseline, *(event["alternate"] for event in events)]
        scores, costs = score_responses(
            self.sampler.members, prompts, marker, [candidate["response"] for candidate in candidates]
        )
        for candidate in candidates:
            candidate["scores"] = scores[candidate["response"]]
        return {"baseline": baseline, "checkpoints": events, "scoring_cost": costs}
