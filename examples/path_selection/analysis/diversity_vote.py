"""Compare the seven historical table methods and vote on their saved answers.

The user runs this CPU-only analysis on a compute node, after activating dllm:
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/analysis/diversity_vote.py
Or submit /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/analysis/diversity_vote.slurm.sh.
No model is loaded, no generation is performed, and source results are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import import_module
from itertools import combinations
import json
from pathlib import Path
import re
import sys

ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
sys.path.insert(0, str(ROOT))
helpers = import_module("examples.path_selection.ablation.7_fixed_size_candidate_search.analyze_results")
from examples.path_selection.experiments.artifacts import atomic_json

SUMMARY = ROOT / "eval_results/path_selection/ablation/analysis_20260907/summary.json"
BASELINE = ROOT / "eval_results/path_selection/gsm8k_full_greedy/GSAI-ML__LLaDA-8B-Instruct/results_2026-08-28T22-34-58.832476.json"
OUTPUT = ROOT / "eval_results/path_selection/ablation/diversity_vote_table_v1"
# This order is fixed before inspecting ensemble outcomes and breaks tied votes.
METHODS = (
    ("mdlm", "Original MDLM"),
    ("fixed_k4_n4", "Fixed four tokens, n=4"),
    ("entropy2.0_k4_n4", "B=2, cap 4, n=4"),
    ("max_confidence", "Maximum-confidence selection"),
    ("entropy2.0_k64_n4", "B=2, cap 64, n=4"),
    ("entropy4.0_k64_n4", "B=4, cap 64, n=4"),
    ("entropy2.0_k64_n8", "B=2, cap 64, n=8"),
)


def normalize_answer(value, metric):
    """Apply exactly the saved exact-match normalization, without numeric coercion."""
    helpers.require(isinstance(value, str), "Extracted answers must be strings")
    if value == "[invalid]":
        return None
    for pattern in metric.get("regexes_to_ignore", []):
        value = re.sub(pattern, "", value)
    if metric.get("ignore_case", False):
        value = value.lower()
    helpers.require(not metric.get("ignore_punctuation", False)
                    and not metric.get("ignore_numbers", False), "Unsupported metric normalization")
    return value if value else None


def vote(answers):
    """Plurality with abstentions and fixed method-order ties; no targets are used."""
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return {"answer": None, "votes": 0, "tie": False, "strict_majority": False,
                "abstentions": len(answers)}
    maximum = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == maximum}
    winner = next(answer for answer in answers if answer in tied)
    return {"answer": winner, "votes": maximum, "tie": len(tied) > 1,
            "strict_majority": maximum > len(answers) / 2,
            "abstentions": sum(answer is None for answer in answers)}


def check_identity(reference, other, context):
    for field in helpers.IDENTITY_FIELDS:
        helpers.require(reference[field] == other[field], f"Mismatched {field}: {context}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    historical = {run["name"]: run for run in helpers.read_json(SUMMARY)["runs"]}
    runs, provenance, metric, extraction = {}, [], None, None
    for name, label in METHODS:
        path = BASELINE if name == "mdlm" else Path(historical[name]["result_path"])
        result = helpers.read_json(path)
        sample_path = path.with_name(path.name.replace("results_", "samples_gsm8k_cot_")).with_suffix(".jsonl")
        samples = helpers.load_samples(sample_path, result, None)
        helpers.require(len(samples[helpers.FILTERS[0]]) == 1319, f"Expected 1,319 examples: {path}")
        config = result["configs"]["gsm8k_cot"]
        metrics = [item for item in config["metric_list"] if item["metric"] == "exact_match"]
        helpers.require(len(metrics) == 1, f"Ambiguous matching rule: {path}")
        if metric is None:
            metric = metrics[0]
            extraction = config["filter_list"]
        helpers.require(metrics[0] == metric, f"Matching rules differ: {path}")
        helpers.require(config["filter_list"] == extraction, f"Extraction rules differ: {path}")
        helpers.require(config["num_fewshot"] == 5, f"Expected five-shot prompts: {path}")
        arguments = helpers.model_arguments(result["config"]["model_args"])
        helpers.require(arguments["max_new_tokens"] == 256 and arguments["block_size"] == 64,
                        f"Unexpected generation length/block size: {path}")
        if runs:
            reference = runs["mdlm"]["samples"]
            for filter_name in helpers.FILTERS:
                for doc_id in range(1319):
                    check_identity(reference[filter_name][doc_id], samples[filter_name][doc_id],
                                   f"{name}/{filter_name}/doc{doc_id}")
        runs[name] = {"label": label, "samples": samples,
                      "calls_per_example": 64.0 if name == "mdlm" else historical[name].get("model_calls_per_example")}
        provenance.append({"method": name, "result_path": str(path), "result_sha256": sha256(path),
                           "samples_path": str(sample_path), "samples_sha256": sha256(sample_path),
                           "model_args": arguments, "evaluation_config": result["config"]})
    return runs, metric, provenance


def interval(values):
    import numpy as np
    return np.quantile(values, [0.025, 0.975]).tolist()


def analyze_filter(runs, metric, filter_name, repeats, seed):
    import numpy as np
    names = [name for name, _ in METHODS]
    n = len(runs[names[0]]["samples"][filter_name])
    outcomes = np.zeros((n, len(names) + 1), dtype=np.int8)
    answers, responses, records = [], [], []
    for doc_id in range(n):
        rows = [runs[name]["samples"][filter_name][doc_id] for name in names]
        target = normalize_answer(rows[0]["target"], metric)
        helpers.require(target is not None, f"Invalid target for document {doc_id}")
        normalized = []
        for index, row in enumerate(rows):
            helpers.require(len(row["filtered_resps"]) == 1, "Expected one answer per method")
            answer = normalize_answer(row["filtered_resps"][0], metric)
            correct = answer is not None and answer == target
            helpers.require(correct == bool(row["exact_match"]),
                            f"Normalization does not reproduce saved score: {names[index]}/{filter_name}/{doc_id}")
            normalized.append(answer)
            outcomes[doc_id, index] = correct
        selected = vote(normalized)
        outcomes[doc_id, -1] = selected["answer"] is not None and selected["answer"] == target
        records.append({"doc_id": doc_id, "filter": filter_name, "target": target,
                        "answers": dict(zip(names, normalized)), "vote": selected,
                        "vote_correct": bool(outcomes[doc_id, -1]),
                        "correct_methods": [name for index, name in enumerate(names) if outcomes[doc_id, index]]})
        answers.append(normalized)
        responses.append([row["resps"] for row in rows])

    # Resample whole questions jointly, preserving pairing among all methods.
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, outcomes.shape[1]), dtype=np.float64)
    for start in range(0, repeats, 128):
        size = min(128, repeats - start)
        indices = rng.integers(0, n, size=(size, n))
        boot[start:start + size] = outcomes[indices].mean(axis=1)
    individual, paired = [], []
    counts = outcomes[:, :-1].sum(axis=1)
    for index, name in enumerate(names):
        individual.append({"method": name, "label": runs[name]["label"],
                           "correct": int(outcomes[:, index].sum()), "accuracy": float(outcomes[:, index].mean()),
                           "ci95": interval(boot[:, index]),
                           "uniquely_correct": int(((counts == 1) & (outcomes[:, index] == 1)).sum())})
    for left, right in combinations(range(len(names)), 2):
        left_wins = int(((outcomes[:, left] == 1) & (outcomes[:, right] == 0)).sum())
        right_wins = int(((outcomes[:, right] == 1) & (outcomes[:, left] == 0)).sum())
        valid_equal = sum(row[left] is not None and row[left] == row[right] for row in answers)
        paired.append({"left": names[left], "right": names[right],
                       "same_valid_answer": valid_equal,
                       "both_invalid": sum(row[left] is None and row[right] is None for row in answers),
                       "same_response_text": sum(row[left] == row[right] for row in responses),
                       "both_correct": int(((outcomes[:, left] == 1) & (outcomes[:, right] == 1)).sum()),
                       "both_wrong": int(((outcomes[:, left] == 0) & (outcomes[:, right] == 0)).sum()),
                       "left_only_correct": left_wins, "right_only_correct": right_wins,
                       "either_correct": int(((outcomes[:, left] + outcomes[:, right]) > 0).sum()),
                       "difference_pp": 100 * float((outcomes[:, left] - outcomes[:, right]).mean()),
                       "ci95_pp": interval(100 * (boot[:, left] - boot[:, right])),
                       "mcnemar_p_unadjusted": helpers.exact_mcnemar(left_wins, right_wins)})
    ensemble_comparisons = []
    for index, name in enumerate(names):
        ensemble_comparisons.append({"reference": name,
            "wins": int(((outcomes[:, -1] == 1) & (outcomes[:, index] == 0)).sum()),
            "losses": int(((outcomes[:, -1] == 0) & (outcomes[:, index] == 1)).sum()),
            "difference_pp": 100 * float((outcomes[:, -1] - outcomes[:, index]).mean()),
            "ci95_pp": interval(100 * (boot[:, -1] - boot[:, index]))})
    majority = [item for item in records if item["vote"]["strict_majority"]]
    ensemble = {"correct": int(outcomes[:, -1].sum()), "accuracy": float(outcomes[:, -1].mean()),
                "ci95": interval(boot[:, -1]), "ties": sum(row["vote"]["tie"] for row in records),
                "no_valid_votes": sum(row["vote"]["answer"] is None for row in records),
                "strict_majority_coverage": len(majority),
                "strict_majority_correct": sum(row["vote_correct"] for row in majority),
                "strict_majority_accuracy_on_covered": (sum(row["vote_correct"] for row in majority) / len(majority)
                                                        if majority else None),
                "strict_majority_accuracy_abstentions_wrong": sum(row["vote_correct"] for row in majority) / n,
                "comparisons": ensemble_comparisons}
    return {"examples": n, "individual": individual, "pairs": paired, "ensemble": ensemble,
            "oracle_any_correct": int((counts > 0).sum()), "all_correct": int((counts == len(names)).sum()),
            "none_correct": int((counts == 0).sum()),
            "correct_method_count_histogram": dict(sorted(Counter(map(int, counts)).items()))}, records


def write_reports(summary, directory):
    flex, strict = (summary["filters"][name] for name in helpers.FILTERS)
    lines = ["# Answer diversity and voting", "",
             "Seven methods from the historical table; 1,319 shared questions and identical saved prompts/targets.", "",
             "This measures extracted-answer and response-text diversity. Exact token-reveal path diversity is unavailable: "
             "the baseline has no path trace, the cheap selector has empty step records, and compact dependency traces omit selected positions.", "",
             "Voting uses the existing exact-match normalization. Invalid extractions abstain. The largest answer group wins; "
             "ties use the fixed table order (MDLM first), without consulting targets. A strict majority requires at least four of seven votes. "
             "No ensemble subset or tie rule was selected by accuracy.", "",
             "| Method | Strict correct | Flexible correct | Flexible accuracy | Sole successful method (flexible) |",
             "|---|---:|---:|---:|---:|"]
    tex = [r"\subsection{Answer Diversity and Voting}",
           r"\begin{table}[H]", r"\centering", r"\begin{tabular}{lrrr}", r"\toprule",
           r"Method & Strict correct & Flexible correct & Flexible accuracy \\", r"\midrule"]
    for index, item in enumerate(flex["individual"]):
        strict_count = strict["individual"][index]["correct"]
        lines.append(f"| {item['label']} | {strict_count} | {item['correct']} | {100 * item['accuracy']:.2f}% | {item['uniquely_correct']} |")
        tex.append(f"{item['label']} & {strict_count} & {item['correct']} & {100 * item['accuracy']:.2f}\\% " + r"\\")
    e = flex["ensemble"]
    lines += [f"| Plurality vote | {strict['ensemble']['correct']} | {e['correct']} | {100 * e['accuracy']:.2f}% | --- |", "",
              f"Flexible vote accuracy: **{100 * e['accuracy']:.2f}%**, 95% document bootstrap interval "
              f"[{100 * e['ci95'][0]:.2f}, {100 * e['ci95'][1]:.2f}]%. Tied largest groups: {e['ties']}. "
              f"No valid votes: {e['no_valid_votes']}.", "",
              f"A strict majority exists for {e['strict_majority_coverage']}/1319 questions; "
              f"{e['strict_majority_correct']} of those answers are correct. "
              f"Counting all other questions as abstentions/incorrect gives {100 * e['strict_majority_accuracy_abstentions_wrong']:.2f}% accuracy.", "",
              f"At least one method is correct on **{flex['oracle_any_correct']}/1319** questions "
              f"({100 * flex['oracle_any_correct'] / 1319:.2f}%). This is an oracle coverage bound, not a usable selector. "
              f"All seven are correct on {flex['all_correct']}; none is correct on {flex['none_correct']}.", "",
              "| Pair (left / right) | Same valid answer | Both correct | Left only correct | Right only correct | Both wrong |",
              "|---|---:|---:|---:|---:|---:|"]
    for pair in flex["pairs"]:
        lines.append(f"| {pair['left']} / {pair['right']} | {pair['same_valid_answer']} | {pair['both_correct']} | "
                     f"{pair['left_only_correct']} | {pair['right_only_correct']} | {pair['both_wrong']} |")
    lines += ["", "Vote gains and losses against each method (flexible extraction):", "",
              "| Reference | Vote wins | Vote losses | Difference (pp) | Paired 95% CI (pp) |",
              "|---|---:|---:|---:|---:|"]
    for pair in e["comparisons"]:
        lines.append(f"| {pair['reference']} | {pair['wins']} | {pair['losses']} | {pair['difference_pp']:+.2f} | "
                     f"[{pair['ci95_pp'][0]:+.2f}, {pair['ci95_pp'][1]:+.2f}] |")
    lines += ["", "Inference is exploratory across historical implementations and one generation run per method. "
              "Shared documents/prompts are checked, but this is not an isolated causal mechanism comparison.", "",
              "Voting requires generating all seven answers. The total evaluation cost is the sum across methods; "
              "the historical cheap-selector cost is missing, so the exact ensemble cost remains unavailable.", "",
              f"Machine-readable details: [summary.json]({directory / 'summary.json'}). "
              f"Auditable per-question votes: [votes.jsonl]({directory / 'votes.jsonl'})."]
    tex += [r"\midrule", f"Plurality vote & {strict['ensemble']['correct']} & {e['correct']} & "
            f"{100 * e['accuracy']:.2f}\\% " + r"\\", r"\bottomrule", r"\end{tabular}",
            r"\caption{Results on 1,319 GSM8K-CoT examples using the saved strict and flexible extraction rules.}",
            r"\end{table}",
            "The vote selects the most frequent normalized answer. Invalid extractions abstain; ties follow the fixed table order. "
            f"At least one method is correct on {flex['oracle_any_correct']} examples, while the vote is correct on {e['correct']}. "
            "This measures answer diversity, not the token-reveal order. Generating all answers incurs the combined cost of the seven methods."]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    (directory / "report.tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    helpers.require(args.bootstrap_resamples > 0, "Bootstrap resamples must be positive")
    directory = args.output_directory.resolve()
    helpers.require(not directory.exists() or not any(directory.iterdir()),
                    "Choose a new/empty analysis directory; existing reports are not overwritten")
    runs, metric, provenance = load_inputs()
    summary = {"completed": True, "method_order": [name for name, _ in METHODS], "metric": metric,
               "bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed,
               "source_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__)),
                                  str(Path(helpers.__file__)): sha256(Path(helpers.__file__))},
               "input_provenance": provenance, "filters": {},
               "calls_per_example": {name: run["calls_per_example"] for name, run in runs.items()},
               "ensemble_calls_per_example": None,
               "voting_rule": "Plurality; invalid extractions abstain; ties use fixed table order; strict majority >3 votes"}
    records = []
    for filter_name in helpers.FILTERS:
        summary["filters"][filter_name], rows = analyze_filter(runs, metric, filter_name, args.bootstrap_resamples, args.seed)
        records.extend(rows)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "votes.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_reports(summary, directory)
    atomic_json(directory / "summary.json", summary)  # Completion marker written last.
    print((directory / "report.md").read_text())
    print(f"LaTeX report: {directory / 'report.tex'}")


if __name__ == "__main__":
    main()
