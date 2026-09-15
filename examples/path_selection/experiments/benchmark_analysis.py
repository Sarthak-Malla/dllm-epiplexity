"""Report task-native metrics and paired accuracy for compact policy benchmarks.

On a compute node with the dllm environment, run:
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/analyze.py --output-root /absolute/path/to/benchmark
The generic analyzer dispatches here for policy_benchmark manifests.
"""

import csv
from statistics import mean

try:
    from .artifacts import atomic_json
except ImportError:
    from artifacts import atomic_json


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def build_benchmark_report(root, manifest, units, paired_summary):
    expected = set(manifest["document_ids"])
    groups = {}
    for record in units["benchmarks"].values():
        arm, doc_id = record["arm"], record["doc_id"]
        if doc_id not in expected or doc_id in groups.setdefault(arm, {}):
            raise ValueError(f"Unexpected or duplicate benchmark document: {arm}/{doc_id}")
        groups[arm][doc_id] = record
    subsets = {"all": expected}
    split = manifest["settings"].get("report_split_at", 0)
    if split:
        subsets["prefix"] = {doc_id for doc_id in expected if doc_id < split}
        subsets["remainder"] = expected - subsets["prefix"]
    results, pairs, metric_rows = {}, {}, []
    for subset, ids in subsets.items():
        if not ids:
            continue
        by_arm = {}
        for arm, records in sorted(groups.items()):
            shared = ids & set(records)
            if not shared:
                continue
            items = [records[doc] for doc in sorted(shared)]
            names = set(items[0]["metric_scores"])
            if any(set(item["metric_scores"]) != names for item in items):
                raise ValueError(f"Inconsistent task metrics within {arm}.")
            by_arm[arm] = {
                "documents": len(items), "expected_documents": len(ids), "complete": shared == ids,
                "primary_accuracy": mean(item["primary_correct"] for item in items),
                "mean_evaluated_rows": mean(item["accounting"]["evaluated_rows"] for item in items),
                "mean_model_calls": mean(item["accounting"]["model_calls"] for item in items),
                "mean_generation_seconds": mean(item["generation_seconds"] for item in items),
                "mean_actions": mean(item["action_count"] for item in items),
                "metric_means": {name: mean(item["metric_scores"][name] for item in items) for name in sorted(names)},
            }
            metric_rows.extend({"subset": subset, "arm": arm, "metric": name, "value": value,
                                "documents": len(items)} for name, value in by_arm[arm]["metric_means"].items())
        results[subset] = by_arm
        pairs[subset] = {}
        comparisons = (("first_action_entropy", "reference_cheap"),
                       ("first_action_entropy", "reference_entropy"),
                       ("reference_entropy", "reference_cheap"))
        for left, right in comparisons:
            if left not in groups or right not in groups:
                continue
            shared = sorted(ids & set(groups[left]) & set(groups[right]))
            if shared:
                result = paired_summary([{"doc_id": doc, "left": groups[left][doc]["primary_correct"],
                                          "right": groups[right][doc]["primary_correct"]} for doc in shared])
                result["complete"] = set(shared) == ids
                pairs[subset][f"{left}-minus-{right}"] = result
    summary = {"task": manifest["task"], "primary_metric": manifest["primary_metric"],
               "primary_filter": manifest["primary_filter"], "document_count": len(expected),
               "coverage": {kind: len(values) for kind, values in units.items()},
               "benchmarks": results, "paired_benchmarks": pairs, "development_nomination": None}
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    _csv(output / "metrics.csv", metric_rows)
    _csv(output / "benchmarks.csv", [{"subset": subset, "arm": arm,
                                      **{key: value for key, value in item.items() if key != "metric_means"}}
                                     for subset, arms in results.items() for arm, item in arms.items()])
    _csv(output / "paired_comparisons.csv", [{"subset": subset, "comparison": comparison, **item}
                                            for subset, comparisons in pairs.items()
                                            for comparison, item in comparisons.items()])
    lines = [f"# Selector benchmark: {manifest['task']}", "",
             f"Primary metric: `{manifest['primary_metric']},{manifest['primary_filter']}`. "
             f"Evaluation documents: {len(expected)}.", "",
             "Timing measures generation only; task filtering and correctness evaluation occur afterward.", ""]
    if split:
        lines.extend([f"The prefix contains document IDs below {split}; the remainder excludes them. "
                      "Report both alongside the full-task result when the prefix informed policy selection.", ""])
    for subset, arms in results.items():
        lines.extend([f"## {subset}", "",
                      "| Policy | Documents | Primary accuracy | Mean evaluations | Mean seconds | Complete |",
                      "|---|---:|---:|---:|---:|---|"])
        for arm, item in arms.items():
            lines.append(f"| {arm} | {item['documents']}/{item['expected_documents']} | "
                         f"{100 * item['primary_accuracy']:.2f}% | {item['mean_evaluated_rows']:.2f} | "
                         f"{item['mean_generation_seconds']:.2f} | {item['complete']} |")
        lines.extend(["", "| Comparison | Problems | Wins / losses | Difference (pp) | 95% interval (pp) | Complete |",
                      "|---|---:|---:|---:|---|---|"])
        for comparison, item in pairs[subset].items():
            low, high = item["ci95_pp"]
            lines.append(f"| {comparison} | {item['problems']} | {item['wins']} / {item['losses']} | "
                         f"{item['difference_pp']:.2f} | [{low:.2f}, {high:.2f}] | {item['complete']} |")
        lines.append("")
    lines.extend(["Intervals resample whole problems. Partial comparisons include only shared completed problems. "
                  "An observed gap within one percentage point does not establish accuracy preservation.", "",
                  f"All task-native metric/filter combinations: [metrics.csv]({output / 'metrics.csv'}).", ""])
    (output / "report.md").write_text("\n".join(lines))
    return summary
