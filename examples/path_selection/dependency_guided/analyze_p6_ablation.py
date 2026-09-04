"""Summarize the Phase-6 fixed-k mechanism ablation and audit traces.

Run after the Slurm array finishes:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_p6_ablation.py \
        --input-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p6_7 \
        --output-path /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p6_7/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


EXPECTED_TASKS = ("gsm8k_cot", "humaneval_instruct")
EXPECTED_VARIANTS = (
    "correlated_together",
    "top_confidence",
    "hard_low_conflict",
    "anchor_support_only",
    "soft_no_anchor",
    "soft_full",
)
EXPECTED_COMMIT_K = (2, 4)


def _finite(values: Iterable[float | int | None]) -> list[float]:
    """Return finite numeric values as floats."""
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _distribution(values: Iterable[float | int | None]) -> dict[str, object]:
    """Describe a numeric vector without requiring NumPy."""
    finite = sorted(_finite(values))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

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
    """Find the latest lm-eval aggregate while excluding sampler sidecars."""
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
    """Extract scalar task-quality metrics while excluding aliases and stderr."""
    task_result = result["results"][task]
    return {
        name: float(value)
        for name, value in task_result.items()
        if name != "alias"
        and "stderr" not in name
        and isinstance(value, (int, float))
    }


def _difference(left: float | None, right: float | None) -> float | None:
    """Subtract two optional scalars."""
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _quality_differences(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Compute matched quality deltas for metrics shared by two runs."""
    return {
        name: left[name] - right[name]
        for name in sorted(left.keys() & right.keys())
    }


def _load_run(
    root: Path,
    verifier: str,
    task: str,
    variant: str,
    commit_k: int,
    candidate_budget: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load one aggregate plus its diagnostics and runtime sidecars."""
    run_directory = (
        root
        / verifier
        / task
        / variant
        / f"k{commit_k}"
        / f"n{candidate_budget}"
    )
    result_path = _latest_result(run_directory)
    diagnostics_path = run_directory / f"results.json_{verifier}_diagnostics.json"
    runtime_path = run_directory / f"results.json_{verifier}_runtime.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Missing diagnostics: {diagnostics_path}.")
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Missing runtime metrics: {runtime_path}.")
    result = json.loads(result_path.read_text())
    examples = json.loads(diagnostics_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    steps = [step for example in examples for step in example.get("steps", [])]
    selected = [
        step["selected_candidate"]
        for step in steps
        if step.get("selected_candidate") is not None
    ]
    consistency_count = sum(
        int(step.get("immediate_token_consistency_count", 0)) for step in steps
    )
    consistency_total = sum(
        int(step.get("immediate_token_consistency_total", 0)) for step in steps
    )
    next_state_steps = [
        step for step in steps if int(step.get("step_index", 0)) > 0
    ]
    step_counts = [len(example.get("steps", [])) for example in examples]
    summary = {
        "verifier": verifier,
        "task": task,
        "variant": variant,
        "commit_k": commit_k,
        "candidate_budget": candidate_budget,
        "result_path": str(result_path.resolve()),
        "diagnostics_path": str(diagnostics_path.resolve()),
        "runtime_path": str(runtime_path.resolve()),
        "quality_metrics_exploratory": _quality_metrics(result, task),
        "example_count": len(examples),
        "step_count": len(steps),
        "steps_per_example": _distribution(step_counts),
        "immediate_token_consistency": {
            "count": consistency_count,
            "total": consistency_total,
            "rate": (
                consistency_count / consistency_total
                if consistency_total
                else None
            ),
        },
        "next_state_metric_mean": _distribution(
            step.get("current_state_metric_mean") for step in next_state_steps
        ),
        "next_state_metric_sum": _distribution(
            step.get("current_state_metric_sum") for step in next_state_steps
        ),
        "selected_set_mean_conflict": _distribution(
            step.get("selected_set_mean_conflict") for step in steps
        ),
        "selected_set_max_conflict": _distribution(
            step.get("selected_set_max_conflict") for step in steps
        ),
        "selected_verifier_score": _distribution(
            candidate.get("verifier_score") for candidate in selected
        ),
        "selected_anchor_support_sum": _distribution(
            candidate.get("anchor_support_sum") for candidate in selected
        ),
        "hard_fallback_selection_count": sum(
            bool(candidate.get("hard_fallback_count")) for candidate in selected
        ),
        "generation_total_seconds": runtime.get("generation_total_seconds"),
        "lm_eval_total_seconds": result.get("total_evaluation_time_seconds"),
        "cuda_peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
    }
    return summary, examples


def _comparison(
    runs: dict[tuple[str, str, int], dict[str, object]],
    task: str,
    commit_k: int,
) -> dict[str, object] | None:
    """Build matched mechanism contrasts for one task and action width."""
    required = {
        variant: runs.get((task, variant, commit_k))
        for variant in EXPECTED_VARIANTS
    }
    if any(run is None for run in required.values()):
        return None
    correlated = required["correlated_together"]
    confidence = required["top_confidence"]
    hard = required["hard_low_conflict"]
    anchor_only = required["anchor_support_only"]
    no_anchor = required["soft_no_anchor"]
    full = required["soft_full"]
    return {
        "task": task,
        "commit_k": commit_k,
        "hard_minus_top_confidence_mean_conflict": _difference(
            hard["selected_set_mean_conflict"]["mean"],
            confidence["selected_set_mean_conflict"]["mean"],
        ),
        "hard_minus_correlated_mean_conflict": _difference(
            hard["selected_set_mean_conflict"]["mean"],
            correlated["selected_set_mean_conflict"]["mean"],
        ),
        "full_minus_correlated_mean_conflict": _difference(
            full["selected_set_mean_conflict"]["mean"],
            correlated["selected_set_mean_conflict"]["mean"],
        ),
        "full_minus_no_anchor_immediate_consistency_rate": _difference(
            full["immediate_token_consistency"]["rate"],
            no_anchor["immediate_token_consistency"]["rate"],
        ),
        "full_minus_no_anchor_next_state_metric_mean": _difference(
            full["next_state_metric_mean"]["mean"],
            no_anchor["next_state_metric_mean"]["mean"],
        ),
        "full_minus_no_anchor_quality_metrics": _quality_differences(
            full["quality_metrics_exploratory"],
            no_anchor["quality_metrics_exploratory"],
        ),
        "anchor_only_minus_no_anchor_quality_metrics": _quality_differences(
            anchor_only["quality_metrics_exploratory"],
            no_anchor["quality_metrics_exploratory"],
        ),
        "interpretation": {
            "conflict_deltas": "Negative favors the first named mechanism.",
            "consistency_delta": "Positive favors full anchor support.",
            "next_state_metric_delta": (
                "Negative favors full anchor support because entropy/risk is lower."
            ),
            "quality_deltas": "Positive favors the first named mechanism.",
        },
    }


def analyze(
    root: Path,
    verifier: str,
    candidate_budget: int,
) -> dict[str, object]:
    """Validate the complete matrix and construct the Phase-6 review payload."""
    runs = []
    traces = []
    failures = []
    for task in EXPECTED_TASKS:
        for variant in EXPECTED_VARIANTS:
            for commit_k in EXPECTED_COMMIT_K:
                try:
                    run, examples = _load_run(
                        root,
                        verifier,
                        task,
                        variant,
                        commit_k,
                        candidate_budget,
                    )
                    runs.append(run)
                    if variant == "soft_full" and examples:
                        traces.append(
                            {
                                "task": task,
                                "variant": variant,
                                "commit_k": commit_k,
                                **examples[0],
                            }
                        )
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    failures.append(
                        {
                            "task": task,
                            "variant": variant,
                            "commit_k": commit_k,
                            "candidate_budget": candidate_budget,
                            "error": str(error),
                        }
                    )
    indexed = {
        (run["task"], run["variant"], run["commit_k"]): run
        for run in runs
    }
    comparisons = [
        comparison
        for task in EXPECTED_TASKS
        for commit_k in EXPECTED_COMMIT_K
        if (comparison := _comparison(indexed, task, commit_k)) is not None
    ]
    expected_count = (
        len(EXPECTED_TASKS) * len(EXPECTED_VARIANTS) * len(EXPECTED_COMMIT_K)
    )
    return {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "interpretation": (
            "Limited-sample Phase-6 mechanism evidence; final correctness is "
            "exploratory, while matched conflict, consistency, and next-state "
            "metrics are the primary signals."
        ),
        "verifier": verifier,
        "candidate_budget": candidate_budget,
        "expected_run_count": expected_count,
        "completed_run_count": len(runs),
        "failures": failures,
        "runs": runs,
        "mechanism_comparisons": comparisons,
        "complete_soft_full_traces": traces,
    }


def main() -> None:
    """Parse arguments and atomically replace the final JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        choices=("entropy_drop", "risk_reduction"),
        default="entropy_drop",
    )
    parser.add_argument("--candidate-budget", type=int, default=4)
    args = parser.parse_args()
    if args.candidate_budget <= 0:
        parser.error("--candidate-budget must be positive")
    summary = analyze(
        args.input_directory.resolve(),
        args.verifier,
        args.candidate_budget,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(args.output_path)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "completed_run_count": summary["completed_run_count"],
                "expected_run_count": summary["expected_run_count"],
                "output_path": str(args.output_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
