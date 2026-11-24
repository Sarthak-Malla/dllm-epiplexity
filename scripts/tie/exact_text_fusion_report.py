"""Run with python scripts/tie/exact_text_fusion_report.py PILOT_TRACE DREAM_TRACE."""

import argparse
from collections import Counter
import json

from commit_study import SPLITS, read


def paired(reference, candidate):
    wins = [doc for doc in reference if not reference[doc] and candidate[doc]]
    losses = [doc for doc in reference if reference[doc] and not candidate[doc]]
    return {"gains": len(wins), "losses": len(losses), "net": len(wins) - len(losses),
            "gained_docs": sorted(wins), "lost_docs": sorted(losses)}


def report(rows, dream):
    keep = {doc: row["branches"]["keep"]["correct"] for doc, row in rows.items()}
    dream_correct = {doc: row["correct"] for doc, row in dream.items()}
    result = {
        "questions": len(rows),
        "standalone": {"llada": sum(keep.values()), "dream": sum(dream_correct.values())},
        "arms": {},
    }
    for arm in ("keep", "tokens", "positions", "joint"):
        correct = {doc: row["branches"][arm]["correct"] for doc, row in rows.items()}
        counts = Counter()
        for row in rows.values():
            counts.update(row["fusion_diagnostics"][arm])
        for key in ("forward_calls", "processed_tokens"):
            counts[f"llada_{key}"] = sum(row["branches"][arm][key] for row in rows.values())
        projected = counts["projected_positions"]
        result["arms"][arm] = {
            "correct": sum(correct.values()),
            "vs_llada": paired(keep, correct),
            "vs_dream": paired(dream_correct, correct),
            "counts": dict(counts),
            "mean_retained_mass_per_projected_position": (
                counts["retained_mass_sum"] / projected if projected else None
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("dream_trace")
    args = parser.parse_args()
    split = SPLITS["pilot"]
    fusion = read([args.trace], split, "exact_text_fusion")
    dream = read([args.dream_trace], split, "dream")
    print(json.dumps(report(fusion, dream), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
