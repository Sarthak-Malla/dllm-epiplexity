"""Run the fusion pilot with python -m dllm.pipelines.tie.eval --model tie."""

from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from itertools import groupby

import torch

from .relay import RelayError, transfer_blocks


MODE = "exact_text_fusion"
PEER_TOP_K = 32
OWNER_TOP_K = 8
MAX_CONDITIONAL_FORWARDS = 2
PEER_WEIGHT = 0.5


def text_boundaries(vocabulary, ids, text):
    """Map complete decoded prefixes to native token boundaries."""
    boundaries = {0: 0}
    for end in range(1, len(ids) + 1):
        prefix = vocabulary.decode(ids[:end])
        if prefix and "\ufffd" not in prefix and text.startswith(prefix):
            boundaries[len(prefix)] = end
    return boundaries


def align_draft(owner, peer, draft):
    """Retokenize a hidden draft and find exact single-owner-token spans."""
    text = owner.decode(draft)
    native = peer.encode(text)
    if (not text or "\ufffd" in text or peer.decode(native) != text
            or any(token in peer.special_ids for token in native)):
        return len(draft), {}
    src = {slot: char for char, slot in text_boundaries(owner, draft, text).items()}
    dst = text_boundaries(peer, native, text)
    spans = {}
    for slot, token in enumerate(draft):
        start, end = src.get(slot), src.get(slot + 1)
        if start not in dst or end not in dst:
            continue
        lo, hi = dst[start], dst[end]
        if lo < hi and owner.decode([token]) == peer.decode(native[lo:hi]) == text[start:end]:
            spans[slot] = (lo, hi)
    return len(native), spans


def peer_canvas(owner, peer, blocks, active, proposals, prompt, marker):
    """Keep visible text, and align hidden positions using only draft boundaries."""
    block = blocks[active]
    parts = [
        list(ids) for _, ids in groupby(block, key=lambda token: token == owner.mask_id)
    ]
    partitioned = [*blocks[:active], *parts, *blocks[active + 1:]]
    transferred = transfer_blocks(partitioned, owner, peer)
    start = len(prompt) + sum(map(len, transferred[:active]))
    relative = 0
    alignment = {}
    active_ids = []
    for part, encoded in zip(parts, transferred[active:active + len(parts)]):
        if part[0] == owner.mask_id:
            slots = range(relative, relative + len(part))
            for ordinary, items in groupby(
                slots, key=lambda slot: proposals[slot] not in owner.special_ids
            ):
                segment = list(items)
                if ordinary:
                    width, spans = align_draft(owner, peer, [proposals[i] for i in segment])
                    offset = start + len(active_ids)
                    alignment.update({
                        segment[i]: tuple(range(offset + lo, offset + hi))
                        for i, (lo, hi) in spans.items()
                    })
                else:
                    width = len(segment)
                active_ids.extend([peer.mask_id] * width)
        else:
            active_ids.extend(encoded)
        relative += len(part)
    transferred[active:active + len(parts)] = [active_ids]
    canvas = prompt + sum(transferred[:-1], []) + marker + transferred[-1]
    return canvas, alignment


def surrounding_text(ids, lo, hi, vocabulary):
    left, right = lo, hi
    while left and ids[left - 1] not in vocabulary.special_ids:
        left -= 1
    while right < len(ids) and ids[right] not in vocabulary.special_ids:
        right += 1
    return ids[left:lo], ids[hi:right]


def exact_in_context(vocabulary, context, ids, text):
    left, right = context
    return vocabulary.decode(left + ids + right) == (
        vocabulary.decode(left) + text + vocabulary.decode(right)
    )


def mix_projected(base, ids, weights):
    """Mix an exact subdistribution, leaving unmatched mass with the owner."""
    mass = sum(weights)
    # Roundoff in products of model probabilities can overshoot by a few ulps.
    if mass > 1.0 + 1e-6:
        raise RuntimeError("Exact token projection exceeded unit mass")
    fused = (1 - PEER_WEIGHT * min(1.0, mass)) * base
    fused[base.new_tensor(ids, dtype=torch.long)] += PEER_WEIGHT * base.new_tensor(weights)
    return fused


class ExactTextFusion:
    """Run repeated token and position fusion and the two isolated ablations."""

    def __init__(self, sampler, owner_index=0):
        self.sampler = sampler
        self.owner = sampler.members[owner_index].vocabulary
        self.peer = sampler.members[1 - owner_index].vocabulary
        self.peer_marker = self.peer.encode(sampler.config.answer_marker)
        self.peer_to_owner = lru_cache(maxsize=65536)(self._peer_to_owner)
        self.owner_to_peer = lru_cache(maxsize=32768)(self._owner_to_peer)

    def _peer_to_owner(self, path):
        if any(token in self.peer.special_ids for token in path):
            return None
        text = self.peer.decode(list(path))
        if not text or "\ufffd" in text:
            return None
        ids = self.owner.encode(text)
        if len(ids) == 1 and ids[0] not in self.owner.special_ids and self.owner.decode(ids) == text:
            return ids[0], text
        return None

    def _owner_to_peer(self, token):
        if token in self.owner.special_ids:
            return []
        text = self.owner.decode([token])
        if not text or "\ufffd" in text:
            return []
        ids = self.peer.encode(text)
        if self.peer.decode(ids) == text and all(i not in self.peer.special_ids for i in ids):
            return ids
        return []

    @staticmethod
    def probabilities(member, native, positions, stats):
        stats["peer_forwards"] += 1
        stats["peer_processed_tokens"] += native.numel()
        return member.probabilities(native, native.new_tensor(positions))

    def project(self, paths, owner_context, peer_context, vocabulary_size):
        projected = {}
        for path, probability in paths:
            if probability <= 0:
                continue
            mapped = self.peer_to_owner(path)
            if mapped is None:
                continue
            token, text = mapped
            if (token < vocabulary_size and exact_in_context(self.owner, owner_context, [token], text)
                    and exact_in_context(self.peer, peer_context, list(path), text)):
                projected[token] = projected.get(token, 0.0) + probability
        return projected

    @staticmethod
    def top_ids(probabilities, count):
        values, ids = probabilities.topk(min(count, probabilities.numel()))
        return list(zip(ids.tolist(), values.tolist()))

    def two_token_paths(self, peer, native, span, first, base, remaining, stats, *, priority_paths=()):
        candidates = {}
        priority = []
        for start, end in priority_paths:
            if not 0 <= start < first.numel() or not 0 <= end < first.numel():
                continue
            if start in self.peer.special_ids or end in self.peer.special_ids:
                continue
            if start not in candidates:
                priority.append(start)
            candidates.setdefault(start, set()).add(end)
        for token, _ in self.top_ids(base, OWNER_TOP_K):
            ids = self.owner_to_peer(token)
            if len(ids) == 2 and max(ids) < first.numel():
                candidates.setdefault(ids[0], set()).add(ids[1])
        best = int(first.argmax())
        prefixes = priority or list(candidates)[:1]
        if best not in self.peer.special_ids and best not in prefixes:
            prefixes.append(best)
        prefixes.extend(start for start in candidates if start not in prefixes)
        paths = []
        for start in prefixes[:remaining]:
            ends = candidates.get(start, ())
            prefix_prob = float(first[start])
            if prefix_prob <= 0:
                continue
            scratch = native.clone()
            scratch[span[0]] = start
            second = self.probabilities(peer, scratch, span[1:], stats)[0]
            stats["conditional_forwards"] += 1
            values = dict(self.top_ids(second, PEER_TOP_K))
            values.update((end, float(second[end])) for end in ends if end not in values)
            paths.extend(((start, end), prefix_prob * prob) for end, prob in values.items())
        return paths

    def choose(self, mode, template, stats, state, block, offset, canvas,
               ranked, tokens, confidence, count, *, probabilities):
        slots = ranked.tolist()
        owner_ids = tokens.tolist()
        stats["active_positions"] += len(slots)
        try:
            peer_ids, spans = peer_canvas(
                self.owner, self.peer, state.snapshot(), block, dict(zip(slots, owner_ids)),
                template.prompt.tolist(), self.peer_marker,
            )
        except RelayError:
            stats["incompatible_context_steps"] += 1
            return ranked[:count], tokens[:count]
        peer = template.member
        if not peer.fits_context(len(peer_ids)):
            stats["overlength_steps"] += 1
            return ranked[:count], tokens[:count]
        stats["same_width_positions"] += sum(len(span) == 1 for span in spans.values())
        stats["two_peer_token_positions"] += sum(len(span) == 2 for span in spans.values())
        stats["wider_peer_positions"] += sum(len(span) > 2 for span in spans.values())
        starts = [span[0] for span in spans.values() if len(span) <= 2]
        if not starts:
            return ranked[:count], tokens[:count]
        native = template.prompt.new_tensor(peer_ids)
        first = dict(zip(starts, self.probabilities(peer, native, starts, stats)))
        owner_canvas = canvas.tolist()
        scores, values = confidence.clone(), tokens.clone()
        initial = stats["conditional_forwards"]
        for row, slot in enumerate(slots):
            span = spans.get(slot)
            if span is None or span[0] not in first:
                continue
            base_row = probabilities[row]
            peer_row = first[span[0]]
            if len(span) == 2:
                budget = MAX_CONDITIONAL_FORWARDS - (stats["conditional_forwards"] - initial)
                if budget <= 0:
                    continue
                paths = self.two_token_paths(peer, native, span, peer_row, base_row, budget, stats)
            else:
                paths = (((token,), prob) for token, prob in self.top_ids(peer_row, PEER_TOP_K))
            projected = self.project(
                paths, surrounding_text(owner_canvas, offset + slot, offset + slot + 1, self.owner),
                surrounding_text(peer_ids, span[0], span[-1] + 1, self.peer),
                base_row.numel(),
            )
            if not projected:
                continue
            stats["projected_positions"] += 1
            stats["two_token_projected_positions"] += len(span) == 2
            stats["retained_mass_sum"] += sum(projected.values())
            q = mix_projected(base_row, list(projected), list(projected.values()))
            scores[row], values[row] = q.max(0)
            if mode == "support":
                scores[row] = q[tokens[row]]
        order = (
            scores.argsort(descending=True, stable=True)[:count]
            if mode in {"positions", "support", "joint"} else torch.arange(count, device=ranked.device)
        )
        selected_values = tokens[order] if mode in {"positions", "support"} else values[order]
        stats["position_changes"] += len(set(order.tolist()) - set(range(count)))
        stats["token_changes"] += int((selected_values != tokens[order]).sum())
        if mode in {"positions", "support"}:
            stats["commits_with_disagreeing_fused_top"] += int((values[order] != tokens[order]).sum())
        return ranked[order], selected_values

    def generate(self, state, template, mode, generator=None):
        stats = Counter()

        def choose(*args, **kwargs):
            return self.choose(mode, template, stats, *args, **kwargs)

        for block, count in self.sampler.schedule:
            fuse = mode != "keep" and block != len(state.blocks) - 1
            state.step(block, count, choose if fuse else None, generator=generator,
                       include_probabilities=fuse)
        return asdict(state.completion(self.sampler.config.answer_marker)), dict(stats)

    @torch.inference_mode()
    def sample(self, prompts):
        completions, diagnostics = {}, {}
        for mode in ("keep", "tokens", "positions", "joint"):
            state, template = self.sampler.initialize(prompts)
            completions[mode], diagnostics[mode] = self.generate(state, template, mode)
        return {"branches": completions, "fusion_diagnostics": diagnostics}
