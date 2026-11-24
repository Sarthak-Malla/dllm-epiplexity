"""Run with python scripts/tie/short_action_fusion_report.py FIT_TRACE VALIDATION_TRACE."""

import argparse
from collections import defaultdict
import json
from random import Random

from action_fusion_report import outcomes, select, signed_percentile
from commit_study import SPLITS, comparison, read


MODE = "action_fusion_short"
MODELS = ("llada", "dream")


def paired_margins(candidates):
    keep, swap = (candidates[action]["scores"] for action in ("keep", "swap"))
    if any(scores[model] is None for scores in (keep, swap) for model in MODELS):
        return None
    return {model: swap[model] - keep[model] for model in MODELS}


def events(rows):
    return [
        {
            "doc": doc, "step": event["step"],
            "gain": int(event["full"]["swap"]["correct"]) - int(event["full"]["keep"]["correct"]),
            **{horizon: paired_margins(event[horizon]) for horizon in ("short", "full")},
        }
        for doc, row in rows.items() for event in row["checkpoints"]
    ]


def fit_scales(fit):
    return {
        horizon: {
            model: sorted(abs(e[horizon][model]) for e in fit if e[horizon] and e[horizon][model])
            for model in MODELS
        }
        for horizon in ("short", "full")
    }


def random_reference(common, count):
    groups = defaultdict(list)
    for event in common:
        groups[event["doc"]].append(event["gain"])
    rng = Random(1234)
    scores = sorted(sum(rng.choice(groups[doc]) for doc in rng.sample(sorted(groups), count))
                    for _ in range(2000))
    return {"swaps": count, "mean_net": sum(scores) / len(scores),
            "net_90_percent_range": [scores[99], scores[1899]]}


def evaluate(rows, scales):
    pairs = events(rows)
    common = [event for event in pairs if event["short"] is not None and event["full"] is not None]
    selectable = [event for event in common if any(event["short"].values()) or any(event["full"].values())]
    for event in selectable:
        short, full = event["short"], event["full"]
        event["margins"] = {
            **short,
            "short_fusion": sum(signed_percentile(short[m], scales["short"][m]) for m in MODELS) / 2,
            "full_fusion": sum(signed_percentile(full[m], scales["full"][m]) for m in MODELS) / 2,
        }
    ranked = {
        rule: select([e for e in selectable if e["margins"][rule] > 0], rule)
        for rule in (*MODELS, "short_fusion", "full_fusion")
    }
    eligible = len({event["doc"] for event in selectable})
    cap = max(1, eligible // 4) if eligible else 0
    count = min(cap, *(len(selection) for selection in ranked.values()))
    selections = {rule: dict(list(items.items())[:count]) for rule, items in ranked.items()}

    def corrected(selected):
        return {doc: {"correct": int(row["baseline"]["correct"]) + selected.get(doc, {}).get("gain", 0)}
                for doc, row in rows.items()}

    baseline = corrected({})
    recovered = {rule: {doc for doc, event in selection.items() if event["gain"] == 1}
                 for rule, selection in selections.items()}
    report = {
        "questions": len(rows),
        "baseline_correct": sum(row["correct"] for row in baseline.values()),
        "pairs": len(pairs),
        "short_pairs_scored": sum(event["short"] is not None for event in pairs),
        "full_pairs_scored": sum(event["full"] is not None for event in pairs),
        "common_pairs_scored": len(common),
        "scorable_questions_with_distinct_scores": eligible,
        "recoverable_questions": len({e["doc"] for e in pairs if e["gain"] == 1}),
        "recoverable_questions_in_common_pool": len({e["doc"] for e in selectable if e["gain"] == 1}),
        "matched_swaps": count,
        "positive_questions": {rule: len(items) for rule, items in ranked.items()},
        "matched_selectors": {rule: outcomes(items) for rule, items in selections.items()},
        "random": random_reference(selectable, count),
        "short_fusion_vs": {
            label: comparison(corrected(selections["short_fusion"]), corrected(items))
            for label, items in {"baseline": {}, **selections}.items() if label != "short_fusion"
        },
        "full_fusion_recoveries_also_selected_by_short": sorted(
            recovered["full_fusion"] & recovered["short_fusion"]
        ),
        "short_recovery_docs": sorted(recovered["short_fusion"]),
        "full_recovery_docs": sorted(recovered["full_fusion"]),
        "lookahead_forwards": sum(c["forwards"] for row in rows.values() for e in row["checkpoints"]
                                  for c in e["short"].values()),
        "lookahead_tokens": sum(c["processed_tokens"] for row in rows.values() for e in row["checkpoints"]
                                for c in e["short"].values()),
        "scoring_cost": {
            key: {model: [sum(row[key][model][i] for row in rows.values()) for i in range(2)]
                  for model in MODELS}
            for key in ("scoring_cost", "full_reference_scoring_cost")
        },
    }
    reference_rollouts = [e["full"]["swap"] for row in rows.values() for e in row["checkpoints"]
                          if "forwards" in e["full"]["swap"]]
    if reference_rollouts:
        report["reference_rollout_forwards"] = sum(c["forwards"] for c in reference_rollouts)
        report["reference_rollout_tokens"] = sum(c["processed_tokens"] for c in reference_rollouts)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit")
    parser.add_argument("validation")
    parser.add_argument("--fresh", action="store_true", help="Score a fresh development trace")
    args = parser.parse_args()
    fit = read([args.fit], SPLITS["fit"], MODE)
    split = "fresh" if args.fresh else "validation"
    validation = read([args.validation], SPLITS[split], f"{MODE}_fresh" if args.fresh else MODE)
    scales = fit_scales(events(fit))
    print(json.dumps({"fit": evaluate(fit, scales), split: evaluate(validation, scales)},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
