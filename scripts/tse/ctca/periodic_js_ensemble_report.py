"""Summarize a completed JS ensemble pilot on a Slurm allocation.

    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/periodic_js_ensemble_report.py \
        /absolute/run/directory/trajectory.jsonl /absolute/run/directory/split.json
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.tie.commit_study import FRESH_GSM8K_TASK, MODEL_BILLIONS, SPLITS, comparison, read


ARMS = ("llada", "dream", "periodic_fusion", "periodic_js")
SELECTIVE_ARMS = ("llada", "dream", "periodic_js", "selective_js")


def distribution_summary(rows, arm):
    counts = Counter()
    strata = defaultdict(list)
    for row in rows.values():
        diagnostics = row["fusion_diagnostics"][arm]
        counts.update(diagnostics["stats"])
        for point in diagnostics["positions"]:
            relation = "same_projected_top1" if point["owner_token"] == point["projected_peer_top"] else "different_projected_top1"
            strata[point["peer_width"], relation].append(point)
    return {
        "counts": dict(counts),
        "by_width_and_top1": [
            {"peer_width": width, "proposal_relation": relation, "positions": len(points),
             "mean_js": sum(point["normalized_js"] for point in points) / len(points),
             "mean_owner_common_mass": sum(point["owner_common_mass"] for point in points) / len(points),
             "mean_peer_common_mass": sum(point["peer_common_mass"] for point in points) / len(points),
             "decisions": dict(Counter(point["decision"] for point in points)),
             "token_changes": sum(point["written_token"] != point["owner_token"] for point in points)}
            for (width, relation), points in sorted(strata.items())
        ],
    }


def routing_summary(rows):
    rounds, routed, contacted = 0, 0, 0
    thresholds = set()
    for doc, row in rows.items():
        diagnostics = row["fusion_diagnostics"]["selective_js"]
        threshold = diagnostics["confidence_threshold"]
        thresholds.add(threshold)
        stats = diagnostics["stats"]
        decisions = diagnostics["routing"]
        if [point["round"] for point in decisions] != list(range(1, stats["rounds"] + 1)):
            raise ValueError(f"Incomplete routing trace for doc {doc}")
        for point in decisions:
            confidence = point["min_owner_confidence"]
            if point["routed"] != (confidence is not None and confidence < threshold):
                raise ValueError(f"Routing decision disagrees with the threshold for doc {doc}")
            if not point["routed"] and (point["peer_forwards"] or point["token_changes"]):
                raise ValueError(f"Skipped round performed peer work for doc {doc}")
        if sum(point["peer_forwards"] for point in decisions) != stats.get("peer_forwards", 0):
            raise ValueError(f"Routing counters disagree with the peer cost for doc {doc}")
        rounds += len(decisions)
        routed += sum(point["routed"] for point in decisions)
        contacted += sum(point["peer_forwards"] > 0 for point in decisions)
    if len(thresholds) != 1:
        raise ValueError("Expected one frozen routing confidence threshold")
    return {"confidence_threshold": thresholds.pop(), "rounds": rounds,
            "routed_rounds": routed, "skipped_rounds": rounds - routed,
            "rounds_with_peer_forwards": contacted, "routing_fraction": routed / rounds}


def pilot_branches(rows, arms, owner):
    if any(set(row["branches"]) != set(arms) for row in rows.values()):
        raise ValueError("Each question needs all four pilot arms")
    branches = {arm: {doc: row["branches"][arm] for doc, row in rows.items()} for arm in arms}
    for arm, records in branches.items():
        expected = arm if arm in MODEL_BILLIONS else owner
        if any(record["member"] != expected for record in records.values()):
            raise ValueError(f"Expected {expected} to decode arm {arm}")
    return branches


def report(trace, split, *, selective=False, owner="llada", llada_owner_trace=None):
    if llada_owner_trace is not None and (not selective or owner != "dream"):
        raise ValueError("The LLaDA-owner reference requires a Dream-owned selective pilot")
    split_name = "selective_js" if selective else "periodic_js"
    arms = SELECTIVE_ARMS if selective else ARMS
    peer = "llada" if owner == "dream" else "dream"
    ids = json.loads(split.read_text())[FRESH_GSM8K_TASK]
    if ids != SPLITS[split_name]:
        raise ValueError(f"Written split does not match the frozen {split_name} questions")
    rows = read([trace], ids, f"{split_name}_ensemble")
    branches = pilot_branches(rows, arms, owner)
    diagnostics = {arm: distribution_summary(rows, arm) for arm in arms[2:]}
    metrics = {}
    for arm, records in branches.items():
        model = arm if arm in MODEL_BILLIONS else owner
        native_forwards = sum(item["forward_calls"] for item in records.values())
        native_tokens = sum(item["processed_tokens"] for item in records.values())
        counts = diagnostics.get(arm, {}).get("counts", {})
        peer_forwards = counts.get("peer_forwards", 0)
        peer_tokens = counts.get("peer_processed_tokens", 0)
        correct = sum(item["correct"] for item in records.values())
        metrics[arm] = {
            "correct": correct, "accuracy": correct / len(ids), "forwards": native_forwards + peer_forwards,
            "processed_tokens": native_tokens + peer_tokens, "peer_forwards": peer_forwards,
            "weighted_processed_tokens": MODEL_BILLIONS[model] * native_tokens + MODEL_BILLIONS[peer] * peer_tokens,
        }
        if selective:
            metrics[arm]["generation_seconds"] = sum(row["generation_seconds"][arm] for row in rows.values())
    result = {"documents": len(rows), "owner_model": owner, "peer_model": peer, "arms": metrics,
              "paired": {arm: {base: comparison(branches[arm], branches[base]) for base in ("llada", "dream")}
                         for arm in arms[2:]},
              "fusion_diagnostics": diagnostics}
    if selective:
        result["routing"] = routing_summary(rows)
        result["selective_vs_periodic"] = comparison(branches["selective_js"], branches["periodic_js"])
        periodic_cost = metrics["periodic_js"]["peer_forwards"]
        result["routing"]["peer_forward_savings_fraction"] = (
            1 - metrics["selective_js"]["peer_forwards"] / periodic_cost if periodic_cost else None
        )
    else:
        result["gate_vs_ungated"] = comparison(branches["periodic_js"], branches["periodic_fusion"])
    if llada_owner_trace is not None:
        reference = pilot_branches(read([llada_owner_trace], ids, "selective_js_ensemble"), arms, "llada")
        result["dream_owner_vs_llada_owner"] = {
            arm: comparison(branches[arm], reference[arm]) for arm in arms[2:]
        }
        result["native_output_mismatches_between_runs"] = {
            model: sum(branches[model][doc]["response"] != reference[model][doc]["response"] for doc in ids)
            for model in MODEL_BILLIONS
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--owner", choices=tuple(MODEL_BILLIONS), default="llada")
    parser.add_argument("--llada-owner-trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.trace, args.split, selective=args.selective, owner=args.owner,
                    llada_owner_trace=args.llada_owner_trace)
    contents = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(contents)
        temporary.replace(args.output)
    print(contents, end="")


if __name__ == "__main__":
    main()
