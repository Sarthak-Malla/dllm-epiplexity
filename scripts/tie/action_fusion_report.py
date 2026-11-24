"""Run with python scripts/tie/action_fusion_report.py SPLIT TRACE."""

import argparse
from bisect import bisect_left, bisect_right
import json
from math import copysign

from commit_study import SPLITS, comparison, read


def eligible_actions(rows):
    actions = []
    for doc, row in rows.items():
        baseline = row["baseline"]
        for event in row["checkpoints"]:
            alternate = event["alternate"]
            scores = (baseline["scores"], alternate["scores"])
            alternate_gain = int(alternate["correct"]) - int(baseline["correct"])
            margins = {}
            for model in ("llada", "dream"):
                before, after = (candidate[model] for candidate in scores)
                if before is not None and after is not None:
                    margins[model] = after - before
            actions.append({
                "doc": doc,
                "step": event["step"],
                "action": event.get("action", "defer_last"),
                "gain": alternate_gain,
                "response": alternate["response"],
                "changed_response": alternate["response"] != baseline["response"],
                "margins": margins,
            })
    return actions


def select(actions, rule="llada"):
    selected = {}
    for action in sorted(actions, key=lambda a: (-a["margins"][rule], a["step"], a["doc"])):
        selected.setdefault(action["doc"], action)
    return selected


def outcomes(selected):
    gains = [action["gain"] for action in selected.values()]
    return {"swaps": len(gains), "wins": gains.count(1), "losses": gains.count(-1),
            "net": sum(gains)}


def unique_changed(actions):
    return {(a["doc"], a["response"]): a for a in actions if a["changed_response"]}.values()


def signed_percentile(value, scale):
    if not value or not scale:
        return 0.0
    rank = bisect_left(scale, abs(value)) + bisect_right(scale, abs(value))
    return copysign(rank / (2 * len(scale)), value)


def wide_comparison(actions, paired, correctness, reference):
    """Combine signed, unlabeled score percentiles across the paired pool."""
    scales = {
        model: sorted(abs(a["margins"][model]) for a in unique_changed(reference)
                      if a["margins"][model])
        for model in ("llada", "dream")
    }

    for a in paired:
        margins = a["margins"]
        margins["fusion"] = sum(signed_percentile(margins[model], scales[model])
                                for model in ("llada", "dream")) / 2
    selectors = {
        rule: select([a for a in paired if a["margins"][rule] > 0], rule)
        for rule in ("llada", "dream", "fusion")
    }
    eligible_docs = {a["doc"] for a in unique_changed(paired)}
    cap = max(1, len(eligible_docs) // 4) if eligible_docs else 0
    count = min(cap, *(len(result) for result in selectors.values()))
    matched = {rule: dict(list(selected.items())[:count]) for rule, selected in selectors.items()}
    standard_rescues = {a["doc"] for a in actions if a["action"] == "defer_last" and a["gain"] == 1}
    wide_rescues = {a["doc"] for a in actions if a["gain"] == 1}
    return {
        "by_action": {
            kind: {
                "tested": sum(a["action"] == kind for a in actions),
                "changed_responses": sum(a["action"] == kind and a["changed_response"] for a in actions),
                "recoveries": sum(a["action"] == kind and a["gain"] == 1 for a in actions),
                "harms": sum(a["action"] == kind and a["gain"] == -1 for a in actions),
            }
            for kind in sorted({a["action"] for a in actions})
        },
        "recoverable_questions_beyond_defer_last": sorted(wide_rescues - standard_rescues),
        "scored_questions_with_changed_response": len(eligible_docs),
        "nonzero_reference_margins": {model: len(scale) for model, scale in scales.items()},
        "positive_questions": {rule: len(selected) for rule, selected in selectors.items()},
        "matched_swaps": count,
        "matched_selectors": {rule: outcomes(selected) for rule, selected in matched.items()},
        "fusion_vs_each_verifier": {
            rule: comparison(correctness(matched["fusion"]), correctness(matched[rule]))
            for rule in ("llada", "dream")
        },
    }


def report(rows, *, wide=False, fit_rows=None):
    actions = eligible_actions(rows)
    paired = [action for action in actions if len(action["margins"]) == 2]
    owner_positive = [a for a in paired if a["margins"]["llada"] > 0]
    fusion = select([a for a in owner_positive if a["margins"]["dream"] > 0])
    owner = select(owner_positive)
    owner_matched = dict(list(owner.items())[:len(fusion)])
    baseline = {doc: {"correct": row["baseline"]["correct"]} for doc, row in rows.items()}

    def correctness(selected):
        return {
            doc: {"correct": int(result["correct"]) + selected.get(doc, {}).get("gain", 0)}
            for doc, result in baseline.items()
        }

    output = {
        "questions": len(rows),
        "baseline_correct": sum(row["correct"] for row in baseline.values()),
        "actions": len(actions),
        "changed_responses": sum(a["changed_response"] for a in actions),
        "scored_by_both": len(paired),
        "recoverable_questions": len({a["doc"] for a in actions if a["gain"] == 1}),
        "scored_recoverable_questions": len({a["doc"] for a in paired if a["gain"] == 1}),
        "by_checkpoint": {
            step: {
                "actions": sum(a["step"] == step for a in actions),
                "scored_by_both": sum(a["step"] == step for a in paired),
                "recoveries": sum(a["step"] == step and a["gain"] == 1 for a in paired),
                "harms": sum(a["step"] == step and a["gain"] == -1 for a in paired),
            }
            for step in sorted({a["step"] for a in actions})
        },
        "unanimous_fusion": outcomes(fusion),
        "llada_only_matched_rate": outcomes(owner_matched),
        "fusion_vs_llada_only": comparison(correctness(fusion), correctness(owner_matched)),
        "rollout_forwards": sum(e["rollout_forwards"] for row in rows.values()
                                for e in row["checkpoints"]),
        "rollout_tokens": sum(e["rollout_tokens"] for row in rows.values()
                              for e in row["checkpoints"]),
        "scoring_cost": {
            model: [sum(row["scoring_cost"][model][i] for row in rows.values()) for i in range(2)]
            for model in ("llada", "dream")
        },
    }
    if wide:
        reference = paired if fit_rows is None else [
            a for a in eligible_actions(fit_rows) if len(a["margins"]) == 2
        ]
        output["wide_comparison"] = wide_comparison(actions, paired, correctness, reference)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("pilot", "fit", "validation"))
    parser.add_argument("trace")
    parser.add_argument("--wide", action="store_true", help="Compare all three wide-pool selectors")
    parser.add_argument("--fit-trace", help="Wide-pool fit trace for the validation score scale")
    args = parser.parse_args()
    mode = "action_fusion_wide" if args.wide else "action_fusion"
    if args.wide and args.split == "validation" and not args.fit_trace:
        parser.error("Wide-pool validation requires --fit-trace")
    if args.fit_trace and (not args.wide or args.split != "validation"):
        parser.error("--fit-trace is only used for wide-pool validation")
    fit_rows = read([args.fit_trace], SPLITS["fit"], mode) if args.fit_trace else None
    rows = read([args.trace], SPLITS[args.split], mode)
    print(json.dumps(report(rows, wide=args.wide, fit_rows=fit_rows), indent=2))


if __name__ == "__main__":
    main()
