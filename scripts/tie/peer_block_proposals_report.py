"""Run with python scripts/tie/peer_block_proposals_report.py PILOT_TRACE."""

import argparse
from collections import Counter
import json

from commit_study import SPLITS, read


MODE = "peer_block_proposals"


def report(rows):
    baseline = sum(row["baseline"]["correct"] for row in rows.values())
    outcomes = {"owner": Counter(), "peer": Counter()}
    rescues = {"owner": set(), "peer": set()}
    match = Counter()
    rejection = Counter()
    forwards, tokens = Counter(), Counter()
    continuations = 0
    for doc, row in rows.items():
        for event in row["checkpoints"]:
            match["attempted_pairs"] += 1
            forwards.update(event["proposal_forwards"])
            tokens.update(event["proposal_tokens"])
            if "rejected" in event:
                rejection[event["rejected"]] += 1
                continue
            match["eligible_pairs"] += 1
            match["different_native_width"] += event["dream_width"] != event["llada_width"]
            match["distinct_responses"] += event["owner"]["response"] != event["peer"]["response"]
            continuations += event["continuation_forwards_per_branch"]
            for rule in outcomes:
                gain = int(event[rule]["correct"]) - int(row["baseline"]["correct"])
                outcomes[rule][gain] += 1
                if gain > 0:
                    rescues[rule].add(doc)
    return {
        "questions": len(rows), "baseline_correct": baseline,
        "candidate_coverage": dict(match), "rejections": dict(rejection),
        "branches": {
            rule: {"wins": outcomes[rule][1], "harms": outcomes[rule][-1],
                   "oracle_correct": baseline + len(rescues[rule]), "recovered_docs": sorted(rescues[rule])}
            for rule in outcomes
        },
        "dream_only_recoveries_above_resampling": sorted(rescues["peer"] - rescues["owner"]),
        "llada_only_recoveries_above_dream_proposals": sorted(rescues["owner"] - rescues["peer"]),
        "joint_oracle_correct": baseline + len(rescues["owner"] | rescues["peer"]),
        "proposal_forwards": dict(forwards),
        "proposal_processed_tokens": dict(tokens),
        "continuation_forwards_per_arm": continuations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    print(json.dumps(report(read([args.trace], SPLITS["pilot"], MODE)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
