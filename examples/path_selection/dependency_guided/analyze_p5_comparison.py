"""Summarize the Phase-5 end-to-end comparison and extract audit traces.

Run after the Slurm array finishes:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_p5_comparison.py \
        --input-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p5_6 \
        --output-path /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p5_6/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


EXPECTED_TASKS = ("gsm8k_cot", "humaneval_instruct")
EXPECTED_STRATEGIES = (
    "baseline_random",
    "baseline_current_mixed",
    "baseline_confidence_gumbel",
    "dependency",
)
EXPECTED_BUDGETS = (2, 4, 8)


def _finite(values: Iterable[float | int | None]) -> list[float]:
    """Return finite numeric values as floats."""
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _distribution(values: Iterable[float | int | None]) -> dict[str, object]:
    """Describe a score vector without requiring NumPy."""
    finite = sorted(_finite(values))
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "q25": None,
                "median": None, "q75": None, "max": None}

    def percentile(fraction: float) -> float:
        offset = (len(finite) - 1) * fraction
        lower = math.floor(offset)
        upper = math.ceil(offset)
        if lower == upper:
            return finite[lower]
        weight = offset - lower
        return finite[lower] * (1 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": finite[0],
        "q25": percentile(0.25),
        "median": percentile(0.5),
        "q75": percentile(0.75),
        "max": finite[-1],
    }


def _latest_result(run_directory: Path) -> Path:
    """Find the latest lm-eval aggregate, excluding custom sidecars."""
    candidates = [
        path
        for path in run_directory.rglob("results_*.json")
        if "_candidates" not in path.name
        and "_diagnostics" not in path.name
        and "_runtime" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No lm-eval result found below {run_directory}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _quality_metrics(result: dict[str, object], task: str) -> dict[str, float]:
    """Extract all scalar task-quality metrics except aliases and stderr."""
    task_result = result["results"][task]
    return {
        name: float(value)
        for name, value in task_result.items()
        if name != "alias"
        and "stderr" not in name
        and isinstance(value, (int, float))
    }


def _load_run(
    root: Path,
    verifier: str,
    task: str,
    strategy: str,
    budget: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load one aggregate and its structured per-step diagnostics."""
    run_directory = root / verifier / task / strategy / f"n{budget}"
    result_path = _latest_result(run_directory)
    diagnostics_path = run_directory / f"results.json_{verifier}_diagnostics.json"
    runtime_path = run_directory / f"results.json_{verifier}_runtime.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Missing diagnostics: {diagnostics_path}.")
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Missing runtime metrics: {runtime_path}.")
    result = json.loads(result_path.read_text())
    diagnostic_examples = json.loads(diagnostics_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    steps = [
        step
        for example in diagnostic_examples
        for step in example.get("steps", [])
    ]
    all_candidate_scores = [
        candidate.get("verifier_score")
        for step in steps
        for candidate in step.get("candidates", [])
        if candidate.get("valid")
    ]
    selected_scores = [
        step["selected_candidate"].get("verifier_score")
        for step in steps
        if step.get("selected_candidate") is not None
    ]
    all_candidate_scores_per_heldout = [
        candidate.get("verifier_score_per_heldout")
        if candidate.get("verifier_score_per_heldout") is not None
        else candidate.get("verifier_score") / max(candidate.get("heldout_count", 0), 1)
        for step in steps
        for candidate in step.get("candidates", [])
        if candidate.get("valid")
    ]
    selected_scores_per_heldout = [
        (
            step["selected_candidate"].get("verifier_score_per_heldout")
            if step["selected_candidate"].get("verifier_score_per_heldout")
            is not None
            else step["selected_candidate"].get("verifier_score")
            / max(step["selected_candidate"].get("heldout_count", 0), 1)
        )
        for step in steps
        if step.get("selected_candidate") is not None
    ]
    timing_names = (
        "base_forward",
        "attention_reconstruction",
        "candidate_proposal_generation",
        "candidate_lookahead",
    )
    timing = {
        name: _distribution(
            step.get("timing_seconds", {}).get(name) for step in steps
        )
        for name in timing_names
    }
    sink_counts = [int(step.get("sink_count", 0)) for step in steps]
    collapses = [step for step in steps if step.get("candidate_collapse")]
    close_margins = [
        step
        for step in steps
        if step.get("margin_to_bf16_reference_ratio") is not None
        and step["margin_to_bf16_reference_ratio"] < 1.0
    ]
    fallback_selections = [
        step
        for step in steps
        if step.get("selected_candidate") is not None
        and step["selected_candidate"].get("fallback") is True
    ]
    summary = {
        "verifier": verifier,
        "task": task,
        "strategy": strategy,
        "candidate_budget": budget,
        "result_path": str(result_path.resolve()),
        "diagnostics_path": str(diagnostics_path.resolve()),
        "runtime_path": str(runtime_path.resolve()),
        "quality_metrics_exploratory": _quality_metrics(result, task),
        "lm_eval_total_seconds": result.get("total_evaluation_time_seconds"),
        "generation_total_seconds": runtime.get("generation_total_seconds"),
        "cuda_peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
        "example_count": len(diagnostic_examples),
        "step_count": len(steps),
        "all_candidate_verifier_scores": _distribution(all_candidate_scores),
        "selected_verifier_scores": _distribution(selected_scores),
        "all_candidate_verifier_scores_per_heldout": _distribution(
            all_candidate_scores_per_heldout
        ),
        "selected_verifier_scores_per_heldout": _distribution(
            selected_scores_per_heldout
        ),
        "timing_seconds_per_step": timing,
        "lookahead_model_calls": _distribution(
            step.get("lookahead_model_calls") for step in steps
        ),
        "sink_count": {
            "total": sum(sink_counts),
            "steps_with_sinks": sum(count > 0 for count in sink_counts),
        },
        "candidate_collapse_count": len(collapses),
        "close_margin_count": len(close_margins),
        "fallback_selection_count": len(fallback_selections),
    }
    return summary, diagnostic_examples


def _load_native_confidence(
    root: Path,
    verifier: str,
    task: str,
) -> dict[str, object]:
    """Load the native MDLMSampler confidence-top-1 decoder control."""
    run_directory = root / verifier / task / "native_confidence" / "n1"
    result_path = _latest_result(run_directory)
    runtime_path = run_directory / "results.json_greedy_runtime.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Missing runtime metrics: {runtime_path}.")
    result = json.loads(result_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    return {
        "verifier": None,
        "task": task,
        "strategy": "native_confidence",
        "candidate_budget": 1,
        "result_path": str(result_path.resolve()),
        "runtime_path": str(runtime_path.resolve()),
        "quality_metrics_exploratory": _quality_metrics(result, task),
        "lm_eval_total_seconds": result.get("total_evaluation_time_seconds"),
        "generation_total_seconds": runtime.get("generation_total_seconds"),
        "cuda_peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
        "note": (
            "Native MDLMSampler confidence top-1; it has no candidate lookahead "
            "and is an end-to-end quality/runtime control, not an N-candidate "
            "proposal-quality control."
        ),
    }


def _comparison_key(run: dict[str, object]) -> tuple[str, str, int]:
    return run["task"], run["strategy"], run["candidate_budget"]


def analyze(root: Path, verifier: str) -> dict[str, object]:
    """Validate the complete matrix and construct the Phase-5 review payload."""
    runs = []
    trace_candidates = []
    failures = []
    for task in EXPECTED_TASKS:
        for strategy in EXPECTED_STRATEGIES:
            for budget in EXPECTED_BUDGETS:
                try:
                    run, examples = _load_run(
                        root, verifier, task, strategy, budget
                    )
                    runs.append(run)
                    if strategy == "dependency" and budget == 4 and examples:
                        trace_candidates.append(
                            {
                                "task": task,
                                "strategy": strategy,
                                "candidate_budget": budget,
                                **examples[0],
                            }
                        )
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    failures.append(
                        {
                            "task": task,
                            "strategy": strategy,
                            "candidate_budget": budget,
                            "error": str(error),
                        }
                    )
    native_runs = []
    for task in EXPECTED_TASKS:
        try:
            native_runs.append(_load_native_confidence(root, verifier, task))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            failures.append(
                {
                    "task": task,
                    "strategy": "native_confidence",
                    "candidate_budget": 1,
                    "error": str(error),
                }
            )
    by_key = {_comparison_key(run): run for run in runs}
    paired = []
    for task in EXPECTED_TASKS:
        dependency = by_key.get((task, "dependency", 4))
        confidence = by_key.get((task, "baseline_confidence_gumbel", 8))
        if dependency is not None and confidence is not None:
            native = next(
                (run for run in native_runs if run["task"] == task),
                None,
            )
            paired.append(
                {
                    "task": task,
                    "dependency_n4": dependency,
                    "confidence_gumbel_n8": confidence,
                    "native_confidence_top1": native,
                    "selected_verifier_mean_difference": (
                        dependency["selected_verifier_scores"]["mean"]
                        - confidence["selected_verifier_scores"]["mean"]
                    ),
                    "selected_verifier_per_heldout_mean_difference": (
                        dependency["selected_verifier_scores_per_heldout"]["mean"]
                        - confidence[
                            "selected_verifier_scores_per_heldout"
                        ]["mean"]
                    ),
                }
            )
    return {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "interpretation": (
            "Small exploratory end-to-end comparison; task accuracy is not a "
            "benchmark claim. Candidate verifier quality is the primary signal."
        ),
        "verifier": verifier,
        "expected_run_count": (
            len(EXPECTED_TASKS) * len(EXPECTED_STRATEGIES) * len(EXPECTED_BUDGETS)
            + len(EXPECTED_TASKS)
        ),
        "completed_run_count": len(runs) + len(native_runs),
        "failures": failures,
        "runs": runs,
        "native_confidence_runs": native_runs,
        "dependency_n4_vs_confidence_gumbel_n8": paired,
        "complete_dependency_n4_traces": trace_candidates[:2],
    }


def main() -> None:
    """Parse arguments and write an atomic-enough final JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        choices=("entropy_drop", "risk_reduction"),
        default="entropy_drop",
    )
    args = parser.parse_args()
    summary = analyze(args.input_directory.resolve(), args.verifier)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(args.output_path)
    print(json.dumps({
        "status": summary["status"],
        "completed_run_count": summary["completed_run_count"],
        "expected_run_count": summary["expected_run_count"],
        "output_path": str(args.output_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
