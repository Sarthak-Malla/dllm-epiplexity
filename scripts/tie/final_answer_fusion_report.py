"""Run with python scripts/tie/final_answer_fusion_report.py FIT_TRACE VALIDATION_TRACE."""

import argparse
from collections import Counter
import json

from action_fusion_report import signed_percentile
from commit_study import SPLITS, comparison, read


MODE = "final_answer_fusion"
MODELS = ("llada", "dream")


def paired_margins(row):
    llada, dream = (row["candidates"][model]["scores"] for model in MODELS)
    if any(scores[model] is None for scores in (llada, dream) for model in MODELS):
        return None
    return {model: dream[model] - llada[model] for model in MODELS}


def fit_scales(rows):
    margins = [pair for row in rows.values() if (pair := paired_margins(row)) is not None]
    return {model: sorted(abs(pair[model]) for pair in margins if pair[model]) for model in MODELS}


def evaluate(rows, scales):
    standalone = {
        model: {doc: {"correct": row["candidates"][model]["correct"]} for doc, row in rows.items()}
        for model in MODELS
    }
    baseline = standalone["llada"]
    chosen = {rule: {} for rule in (*MODELS, "fusion")}
    switches = Counter()
    missing = Counter()
    covered = 0
    for doc, row in rows.items():
        if row["generation_forwards"] != 0:
            raise ValueError(f"Expected only saved responses for doc {doc}")
        margins = paired_margins(row)
        if margins is None:
            for candidate, data in row["candidates"].items():
                missing.update(f"{model}_scoring_{candidate}" for model in MODELS
                               if data["scores"][model] is None)
            margins = dict.fromkeys(MODELS, 0.0)
        else:
            covered += 1
        fused = sum(signed_percentile(margins[model], scales[model]) for model in MODELS) / 2
        for rule, margin in {**margins, "fusion": fused}.items():
            winner = "dream" if margin > 0 else "llada"
            switches[rule] += winner == "dream"
            chosen[rule][doc] = standalone[winner][doc]
    result = {
        "questions": len(rows),
        "scored_by_both": covered,
        "missing_scores": dict(missing),
        "standalone_correct": {
            model: sum(row["correct"] for row in standalone[model].values()) for model in MODELS
        },
        "hindsight_union_correct": sum(
            any(standalone[model][doc]["correct"] for model in MODELS) for doc in rows
        ),
        "selectors": {
            rule: {
                "selected_dream": switches[rule],
                "correct": sum(item["correct"] for item in selection.values()),
                "vs_standalone_llada": comparison(selection, baseline),
            }
            for rule, selection in chosen.items()
        },
        "fusion_vs_single_verifiers": {
            model: comparison(chosen["fusion"], chosen[model]) for model in MODELS
        },
        "generation_forwards": 0,
        "scoring_cost": {
            model: [sum(row["scoring_cost"][model][i] for row in rows.values()) for i in range(2)]
            for model in MODELS
        },
    }
    if "extracted_answer" in next(iter(rows.values()))["candidates"]["llada"]:
        result["missing_extracted_answers"] = {
            model: sum(row["candidates"][model]["extracted_answer"] is None
                       for row in rows.values())
            for model in MODELS
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit")
    parser.add_argument("validation")
    parser.add_argument("--extract", action="store_true", help="Read answer-extraction traces")
    args = parser.parse_args()
    mode = f"{MODE}_extract" if args.extract else MODE
    fit = read([args.fit], SPLITS["fit"], mode)
    validation = read([args.validation], SPLITS["validation"], mode)
    scales = fit_scales(fit)
    print(json.dumps({
        "fit_nonzero_margins": {model: len(scale) for model, scale in scales.items()},
        "fit": evaluate(fit, scales),
        "validation": evaluate(validation, scales),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
