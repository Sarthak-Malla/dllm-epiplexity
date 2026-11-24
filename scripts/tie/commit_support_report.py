"""Run with python scripts/tie/commit_support_report.py TRACE [--replication]."""

import argparse
from collections import Counter
import json
from math import comb

from commit_study import MODEL_BILLIONS, SPLITS, comparison, read


ARMS = ("llada", "dream", "positions", "support", "dream_multi")


def paired_test(candidate, baseline):
    result = comparison(candidate, baseline)
    wins, losses = result["wins"], result["losses"]
    discordant = wins + losses
    tail = sum(comb(discordant, k) for k in range(min(wins, losses) + 1))
    return {**result, "mcnemar_exact_p": min(1.0, 2 * tail / (1 << discordant))}


def report(rows, *, replication=False):
    candidates = ("positions",) if replication else ("positions", "support")
    branches = {
        arm: {doc: row["branches"][arm] for doc, row in rows.items()}
        for arm in ARMS if not replication or arm != "support"
    }
    table = {}
    for arm, records in branches.items():
        tokens, forwards, counts = Counter(), Counter(), Counter()
        for doc, branch in records.items():
            stats = rows[doc]["fusion_diagnostics"].get(arm, {})
            counts.update(stats)
            tokens[branch["member"]] += branch["processed_tokens"]
            forwards[branch["member"]] += branch["forward_calls"]
            if arm in {"positions", "support"}:
                tokens["llada"] += stats.get("peer_processed_tokens", 0)
                forwards["llada"] += stats.get("peer_forwards", 0)
            elif arm == "dream_multi":
                dream = branches["dream"][doc]
                tokens["dream"] += dream["processed_tokens"] + stats["extra_processed_tokens"] - branch["processed_tokens"]
                forwards["dream"] += dream["forward_calls"] + stats["extra_forwards"] - branch["forward_calls"]
        table[arm] = {
            "correct": sum(row["correct"] for row in records.values()),
            "forward_calls": dict(forwards),
            "processed_tokens": dict(tokens),
            "weighted_tokens": sum(MODEL_BILLIONS[name] * count for name, count in tokens.items()),
            "diagnostics": dict(counts),
        }
    comparisons = {
        candidate: {
            arm: paired_test(branches[candidate], branches[arm]) for arm in branches if arm != candidate
        }
        for candidate in candidates
    }
    result = {
        "questions": len(rows),
        "arms": table,
        "native_oracle_correct": sum(
            branches["llada"][doc]["correct"] or branches["dream"][doc]["correct"]
            for doc in rows
        ),
        **{
            f"{candidate}_comparisons": comparisons[candidate]
            for candidate in candidates
        },
        **{
            f"{candidate}_cost_relative_to_dream_multi": (
                table[candidate]["weighted_tokens"] / table["dream_multi"]["weighted_tokens"]
            )
            for candidate in candidates
        },
    }
    if replication:
        result["beats_all_controls_at_0_05"] = all(
            pair["gain"] > 0 and pair["mcnemar_exact_p"] < .05
            for pair in comparisons["positions"].values()
        )
        result["within_two_sample_proxy_budget"] = (
            result["positions_cost_relative_to_dream_multi"] <= 1.05
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    split, mode = (
        ("replication", "fused_position_replication") if args.replication else ("support", "commit_support")
    )
    result = report(read([args.trace], SPLITS[split], mode), replication=args.replication)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
