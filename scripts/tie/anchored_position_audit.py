"""Run from the repository root with python -m scripts.tie.anchored_position_audit TRACE LLADA DREAM."""

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path

from dllm.pipelines.tie.commit_actions import contextual_ids
from dllm.pipelines.tie.relay import RelayError, Vocabulary
from dllm.utils import get_tokenizer


def inspect_gap(block, lo, hi, owner_ids, peer_ids, source, target):
    slots = list(range(lo, hi))
    result = {"slots": slots}
    left_start, right_end = lo, hi
    while left_start and block[left_start - 1] not in source.special_ids:
        left_start -= 1
    while right_end < len(block) and block[right_end] not in source.special_ids:
        right_end += 1
    if left_start == lo or right_end == hi:
        return {**result, "status": "missing_visible_boundary"}
    left = source.decode(block[left_start:lo])
    right = source.decode(block[hi:right_end])
    own = [owner_ids[i] for i in slots]
    peer = [peer_ids[i] for i in slots]
    a, b = source.decode(own), target.decode(peer)
    result.update(left=left, right=right, owner_text=a, peer_text=b)
    if (any(i in source.special_ids for i in own)
            or any(i in target.special_ids for i in peer)
            or "\ufffd" in left + right + a + b):
        return {**result, "status": "invalid_text"}
    try:
        if contextual_ids(source, left, a, right) != own:
            raise RelayError("owner tokens differ from contextual encoding")
        owner_text_peer_ids = contextual_ids(target, left, a, right)
    except RelayError as error:
        return {**result, "status": "owner_text_boundary_failure", "detail": str(error)}
    result.update(owner_width=len(own), peer_width_for_owner_text=len(owner_text_peer_ids))
    if len(owner_text_peer_ids) != len(own):
        return {**result, "status": "different_native_width"}
    try:
        if contextual_ids(target, left, b, right) != peer:
            raise RelayError("peer tokens differ from contextual encoding")
    except RelayError as error:
        return {**result, "status": "peer_text_boundary_failure", "detail": str(error)}
    return {**result, "status": "aligned"}


def inspect_event(event, source, target):
    block = event["draft"][event["block"]]
    slots = event["aligned_mask_slots"]
    owner_ids = dict(zip(slots, event["owner_top_ids"]))
    peer_ids = dict(zip(slots, event["peer_top_ids"]))
    baseline = set(event["default_slots"])
    selected = set(event["alternatives"]["peer_positions"]["slots"])
    promoted = selected - baseline
    gaps = []
    for is_mask, group in groupby(range(len(block)), lambda i: block[i] == source.mask_id):
        run = list(group)
        run_slots = set(run)
        if not is_mask or not run_slots & promoted:
            continue
        result = inspect_gap(block, run[0], run[-1] + 1, owner_ids, peer_ids, source, target)
        if result["status"] == "aligned" and not run_slots <= selected:
            result["status"] = "partially_selected_gap"
        gaps.append(result)
    if not promoted or not gaps:
        raise ValueError("Peer position replay must promote a masked slot")
    return {
        "step": event["step"],
        "baseline_slots": sorted(baseline),
        "peer_slots": sorted(selected),
        "promoted_gaps": gaps,
        "all_promotions_aligned": all(gap["status"] == "aligned" for gap in gaps),
    }


def audit(path, vocabularies):
    summary = {}
    changed = {}
    for owner, source in vocabularies.items():
        target = next(vocab for name, vocab in vocabularies.items() if name != owner)
        counts = Counter()
        changed[owner] = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                draft = row["owners"][owner]
                baseline = int(draft["baseline"]["correct"])
                for event in draft["checkpoints"]:
                    alternatives = event["alternatives"]
                    branch = alternatives.get("peer_positions")
                    if branch is None:
                        continue
                    result = inspect_event(event, source, target)
                    gain = int(branch["completion"]["correct"]) - baseline
                    control = alternatives["owner_positions"]["completion"]
                    control_gain = int(control["correct"]) - baseline
                    counts["peer_replays"] += 1
                    if result["all_promotions_aligned"]:
                        counts["replays_with_all_promotions_aligned"] += 1
                        counts["aligned_recoveries"] += gain == 1
                        counts["aligned_harms"] += gain == -1
                        counts["matched_control_recoveries"] += control_gain == 1
                        counts["matched_control_harms"] += control_gain == -1
                    counts.update(gap["status"] for gap in result["promoted_gaps"])
                    if gain:
                        changed[owner].append({
                            "doc_id": row["doc_id"], "gain": gain,
                            "same_model_gain": control_gain, **result,
                        })
        summary[owner] = dict(counts)
    return {"trace": str(path.resolve()), "coverage": summary, "correctness_changes": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("llada")
    parser.add_argument("dream")
    args = parser.parse_args()
    vocabularies = {
        name: Vocabulary.from_tokenizer(get_tokenizer(model_name_or_path=checkpoint))
        for name, checkpoint in (("llada", args.llada), ("dream", args.dream))
    }
    print(json.dumps(audit(args.trace, vocabularies), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
