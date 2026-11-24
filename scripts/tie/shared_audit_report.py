"""Analyze traces with python scripts/tie/shared_audit_report.py FIT_TRACE VALIDATION_TRACE."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
import random

from commit_study import SPLITS, read


OWNERS = ("llada", "dream")
KINDS = ("text", "positions")


def coverage(rows):
    summary = {}
    for member in OWNERS:
        counts = Counter()
        for row in rows.values():
            owner = row["owners"][member]
            for model, (calls, tokens) in owner.get("proposal_scoring_cost", {}).items():
                counts[f"{model}_proposal_scoring_forwards"] += calls
                counts[f"{model}_proposal_scoring_tokens"] += tokens
            for event in owner["checkpoints"]:
                counts["checkpoints"] += 1
                if "unaligned" in event:
                    counts["unaligned_canvases"] += 1
                    continue
                counts["aligned_canvases"] += 1
                if len(event["aligned_mask_slots"]) > len(event["default_slots"]):
                    counts["position_choice_checkpoints"] += 1
                else:
                    counts["forced_position_checkpoints"] += 1
                counts["position_disagreement"] += event["order_disagrees"]
                counts["common_selected_slots"] += event["common_selected_slots"]
                counts["selected_slots"] += len(event["default_slots"])
                for check in event["text_checks"]:
                    if "rejected" in check:
                        counts["text_alignment_failures"] += 1
                    elif check["a"] == check["b"]:
                        counts["text_agreement"] += 1
                    else:
                        counts["text_disagreement"] += 1
                counts.update(event["alternatives"].keys())
        summary[member] = dict(counts)
    return summary


def candidate_groups(rows, owner, kind, origin):
    groups = {}
    for doc, row in rows.items():
        state = row["owners"][owner]
        baseline = state["baseline"]
        candidates = []
        for event in state["checkpoints"]:
            alt = event["alternatives"].get(f"{origin}_{kind}")
            if alt is None:
                continue
            completion = alt["completion"]
            candidates.append({
                "step": event["step"],
                "gain": int(completion["correct"]) - int(baseline["correct"]),
                "owner_score": alt["owner_score"],
                "peer_score": alt.get("peer_score", 0.0),
                "continuation_forwards": completion["forward_calls"] - event["step"],
            })
        if candidates:
            groups[doc] = candidates
    return groups


def outcomes(candidates):
    gains = [candidate["gain"] for candidate in candidates]
    return {"selected": len(gains), "wins": gains.count(1),
            "losses": gains.count(-1), "net": sum(gains)}


def oracle(groups, total):
    options = list(groups.values())
    return {"questions": total, "eligible_questions": len(options),
            "tested_branches": sum(map(len, options)),
            "recoverable_questions": sum(any(x["gain"] == 1 for x in a) for a in options),
            "questions_with_a_harmful_action": sum(any(x["gain"] == -1 for x in a) for a in options),
            "replay_forwards": sum(x["continuation_forwards"] for a in options for x in a)}


def fit_ranks(groups):
    return {
        feature: sorted(x[feature] for actions in groups.values() for x in actions)
        for feature in ("owner_score", "peer_score")
    }


def choose(groups, ranks, weight, count):
    def score(candidate):
        own = bisect_right(ranks["owner_score"], candidate["owner_score"]) / len(ranks["owner_score"])
        peer = bisect_right(ranks["peer_score"], candidate["peer_score"]) / len(ranks["peer_score"])
        return ((1 - weight) * own + weight * peer, -candidate["step"])

    best = [max(groups[doc], key=score) for doc in sorted(groups)]
    return sorted(best, key=score, reverse=True)[:count]


def match_checkpoints(peer, owner):
    peer_matched, owner_matched = {}, {}
    for doc in peer.keys() & owner.keys():
        steps = {row["step"] for row in peer[doc]} & {row["step"] for row in owner[doc]}
        if steps:
            peer_matched[doc] = [row for row in peer[doc] if row["step"] in steps]
            owner_matched[doc] = [row for row in owner[doc] if row["step"] in steps]
    return peer_matched, owner_matched


def compare_selectors(fit, validation):
    # One action per question and the same fixed intervention count for each comparison.
    fraction = 0.25
    fit_count = max(1, round(len(fit) * fraction))
    count = max(1, round(len(validation) * fraction))
    ranks = fit_ranks(fit)
    weights = (0.25, 0.5, 0.75, 1.0)
    weight = max(weights, key=lambda w: outcomes(choose(fit, ranks, w, fit_count))["net"])
    rng = random.Random(1234)
    keys = sorted(validation)
    random_scores = sorted(
        outcomes([rng.choice(validation[doc]) for doc in rng.sample(keys, count)])["net"]
        for _ in range(200)
    )
    return {"peer_weight": weight, "validation_count": count,
            "cross_model": outcomes(choose(validation, ranks, weight, count)),
            "owner_confidence": outcomes(choose(validation, ranks, 0, count)),
            "random_mean_net": sum(random_scores) / len(random_scores),
            "random_90_percent_range": [random_scores[9], random_scores[189]]}


def report(fit_rows, validation_rows):
    comparisons = {}
    for owner in OWNERS:
        comparisons[owner] = {}
        for kind in KINDS:
            fit = candidate_groups(fit_rows, owner, kind, "peer")
            validation = candidate_groups(validation_rows, owner, kind, "peer")
            fit_oracle = oracle(fit, len(fit_rows))
            controls = candidate_groups(validation_rows, owner, kind, "owner")
            peer_matched, owner_matched = match_checkpoints(validation, controls)
            result = {
                "fit_oracle": fit_oracle,
                "validation_oracle": oracle(validation, len(validation_rows)),
                "same_checkpoint_oracles": {
                    "peer": oracle(peer_matched, len(peer_matched)),
                    "owner": oracle(owner_matched, len(owner_matched)),
                },
            }
            if fit_oracle["recoverable_questions"] and validation:
                result["metadata_selector"] = compare_selectors(fit, validation)
            comparisons[owner][kind] = result
    return {"fit_coverage": coverage(fit_rows), "validation_coverage": coverage(validation_rows),
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit")
    parser.add_argument("validation")
    args = parser.parse_args()
    print(json.dumps(report(read([args.fit], SPLITS["fit"], "shared_audit"),
                            read([args.validation], SPLITS["validation"], "shared_audit")),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
