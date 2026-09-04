"""Summarize Phase-7 calibration, held-out, or confirmation comparisons.

Run after a Phase-7 Slurm stage finishes:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_p7_cardinality.py \
        --input-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p7/main/calibration \
        --output-path /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p7/main/calibration/summary.json \
        --stage calibration
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
import math
from pathlib import Path
import random
import statistics


TASKS = ("gsm8k_cot", "humaneval_instruct")
CALIBRATION_POLICIES = (
    "fixed_k2",
    "scheduler",
    "marginal_tau0",
    "marginal_tau0.25",
    "marginal_tau0.5",
    "joint_per_token",
    "joint_immediate_cost",
    "joint_size_penalty",
    "entropy_budget0.5",
    "entropy_budget1.0",
    "entropy_budget2.0",
)
COMPARISON_POLICIES = (
    "fixed_k2",
    "scheduler",
    "marginal_utility",
    "joint_k",
    "entropy_budget",
)
CONFIRMATION_POLICIES = (
    "fixed_k2",
    "fixed_k4",
    "entropy_budget",
)
MASK_STAGES = ("early", "middle_early", "middle_late", "late")
PRIMARY_SAMPLE_FIELDS = {
    "gsm8k_cot": ("flexible-extract", "exact_match"),
    "humaneval_instruct": ("create_test", "pass@1"),
}


def _finite(values: Iterable[float | int | None]) -> list[float]:
    """Return finite numeric values as floats."""
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    """Describe a numeric vector without NumPy."""
    finite = sorted(_finite(values))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": finite[0],
        "median": statistics.median(finite),
        "max": finite[-1],
    }


def _mask_stage(mask_ratio: float) -> str:
    """Assign one of four descending-mask-ratio stages."""
    if mask_ratio > 0.75:
        return "early"
    if mask_ratio > 0.5:
        return "middle_early"
    if mask_ratio > 0.25:
        return "middle_late"
    return "late"


def _latest_result(run_directory: Path) -> Path:
    """Find the newest lm-eval aggregate below one policy directory."""
    candidates = [
        path
        for path in run_directory.rglob("results_*.json")
        if "_diagnostics" not in path.name
        and "_runtime" not in path.name
        and "_candidates" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No lm-eval result found below {run_directory}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _latest_samples(run_directory: Path, task: str) -> Path:
    """Find the newest per-example lm-eval sample file for one task."""
    candidates = list(run_directory.glob(f"samples_{task}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No lm-eval samples found below {run_directory}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _primary_outcomes(samples_path: Path, task: str) -> dict[int, float]:
    """Load one primary metric outcome per document from an lm-eval JSONL."""
    filter_name, metric_name = PRIMARY_SAMPLE_FIELDS[task]
    outcomes: dict[int, float] = {}
    for line in samples_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("filter") != filter_name:
            continue
        if "doc_id" not in record or metric_name not in record:
            raise KeyError(
                f"Sample in {samples_path} lacks doc_id or {metric_name!r}."
            )
        doc_id = int(record["doc_id"])
        if doc_id in outcomes:
            raise ValueError(f"Duplicate primary outcome for doc_id={doc_id}.")
        outcomes[doc_id] = float(record[metric_name])
    if not outcomes:
        raise ValueError(
            f"No {filter_name!r}/{metric_name!r} outcomes in {samples_path}."
        )
    return outcomes


def _quality_metrics(result: dict[str, object], task: str) -> dict[str, float]:
    """Extract task metrics while excluding aliases and standard errors."""
    task_result = result["results"][task]
    return {
        name: float(value)
        for name, value in task_result.items()
        if name != "alias"
        and "stderr" not in name
        and isinstance(value, (int, float))
    }


def _primary_quality(metrics: dict[str, float], task: str) -> float:
    """Return the predeclared exploratory quality coordinate for a task."""
    preferred = (
        "exact_match,flexible-extract"
        if task == "gsm8k_cot"
        else "pass@1,create_test"
    )
    if preferred not in metrics:
        raise KeyError(f"Missing primary quality metric {preferred!r}.")
    return metrics[preferred]


def _load_run(
    root: Path,
    verifier: str,
    task: str,
    policy: str,
    candidate_budget: int,
    require_samples: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load one aggregate and derive cardinality/efficiency audit metrics."""
    run_directory = root / verifier / task / policy / f"n{candidate_budget}"
    result_path = _latest_result(run_directory)
    diagnostics_path = run_directory / f"results.json_{verifier}_diagnostics.json"
    runtime_path = run_directory / f"results.json_{verifier}_runtime.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Missing diagnostics: {diagnostics_path}.")
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Missing runtime: {runtime_path}.")
    result = json.loads(result_path.read_text())
    examples = json.loads(diagnostics_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    samples_path = None
    outcomes: dict[int, float] = {}
    try:
        samples_path = _latest_samples(run_directory, task)
        outcomes = _primary_outcomes(samples_path, task)
    except FileNotFoundError:
        if require_samples:
            raise
    steps = [step for example in examples for step in example.get("steps", [])]
    selected = [
        step["selected_candidate"]
        for step in steps
        if isinstance(step.get("selected_candidate"), dict)
    ]
    action_sizes = [int(step.get("commit_k", 0)) for step in steps]
    size_counts = Counter(action_sizes)
    stage_sizes: dict[str, list[int]] = {stage: [] for stage in MASK_STAGES}
    for step in steps:
        stage_sizes[_mask_stage(float(step.get("mask_ratio", 0.0)))].append(
            int(step.get("commit_k", 0))
        )

    large_early_consistent = 0
    large_early_total = 0
    large_early_actions = 0
    for example in examples:
        example_steps = example.get("steps", [])
        for index, step in enumerate(example_steps):
            if int(step.get("commit_k", 0)) >= 4 and float(
                step.get("mask_ratio", 0.0)
            ) >= 0.5:
                large_early_actions += 1
                if index + 1 < len(example_steps):
                    successor = example_steps[index + 1]
                    large_early_consistent += int(
                        successor.get("immediate_token_consistency_count", 0)
                    )
                    large_early_total += int(
                        successor.get("immediate_token_consistency_total", 0)
                    )

    quality = _quality_metrics(result, task)
    step_counts = [len(example.get("steps", [])) for example in examples]
    cumulative_metrics = [
        sum(
            float(step.get("current_state_metric_sum", 0.0))
            for step in example.get("steps", [])
        )
        for example in examples
    ]
    invalid_selection_count = sum(
        not isinstance(step.get("selected_candidate"), dict)
        or step["selected_candidate"].get("valid") is not True
        for step in steps
    )
    wrong_action_width_count = sum(
        isinstance(step.get("selected_candidate"), dict)
        and len(step["selected_candidate"].get("positions", []))
        != int(step.get("commit_k", 0))
        for step in steps
    )
    summary = {
        "verifier": verifier,
        "task": task,
        "policy": policy,
        "candidate_budget": candidate_budget,
        "result_path": str(result_path.resolve()),
        "diagnostics_path": str(diagnostics_path.resolve()),
        "runtime_path": str(runtime_path.resolve()),
        "samples_path": (
            str(samples_path.resolve()) if samples_path is not None else None
        ),
        "primary_outcomes_by_doc_id": {
            str(doc_id): outcome for doc_id, outcome in sorted(outcomes.items())
        },
        "quality_metrics_exploratory": quality,
        "primary_quality_exploratory": _primary_quality(quality, task),
        "example_count": len(examples),
        "step_count": len(steps),
        "steps_per_example": _distribution(step_counts),
        "action_size": _distribution(action_sizes),
        "action_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "action_size_by_mask_stage": {
            stage: _distribution(stage_sizes[stage]) for stage in MASK_STAGES
        },
        "base_forward_count": sum(
            int(step.get("captured_base_forward_count", 0)) for step in steps
        ),
        "lookahead_forward_count": sum(
            int(step.get("lookahead_model_calls", 0)) for step in steps
        ),
        "total_model_forward_count": sum(
            int(step.get("captured_base_forward_count", 0))
            + int(step.get("lookahead_model_calls", 0))
            for step in steps
        ),
        "cumulative_current_metric_sum_per_example": _distribution(
            cumulative_metrics
        ),
        "selected_raw_verifier_score": _distribution(
            candidate.get("raw_verifier_score") for candidate in selected
        ),
        "selected_size_aware_verifier_score": _distribution(
            candidate.get("size_aware_verifier_score") for candidate in selected
        ),
        "selected_immediate_action_cost": _distribution(
            candidate.get("immediate_action_cost") for candidate in selected
        ),
        "large_early_action_count": large_early_actions,
        "large_early_token_failure_rate": (
            1.0 - large_early_consistent / large_early_total
            if large_early_total
            else None
        ),
        "large_early_token_failure_count": (
            large_early_total - large_early_consistent
        ),
        "large_early_token_total": large_early_total,
        "candidate_collapse_count": sum(
            bool(step.get("candidate_collapse")) for step in steps
        ),
        "invalid_selection_count": invalid_selection_count,
        "wrong_action_width_count": wrong_action_width_count,
        "generation_total_seconds": runtime.get("generation_total_seconds"),
        "lm_eval_total_seconds": result.get("total_evaluation_time_seconds"),
        "cuda_peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
        "configuration": (
            {
                name: steps[0].get(name)
                for name in (
                    "cardinality_strategy",
                    "maximum_action_size",
                    "allowed_action_sizes",
                    "size_scoring_rule",
                    "utility_threshold",
                    "entropy_budget",
                    "immediate_cost_weight",
                    "size_penalty",
                )
            }
            if steps
            else {}
        ),
    }
    return summary, examples


def _paired_bootstrap_interval(
    differences: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> list[float] | None:
    """Return a deterministic percentile interval for a paired mean difference."""
    if not differences:
        return None
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        statistics.fmean(differences[generator.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    return [
        estimates[int(0.025 * (resamples - 1))],
        estimates[int(0.975 * (resamples - 1))],
    ]


def _paired_quality_comparisons(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare entropy budget with fixed controls on shared document IDs."""
    by_task_policy = {
        (str(run["task"]), str(run["policy"])): run for run in runs
    }
    comparisons = []
    comparison_index = 0
    for task in TASKS:
        challenger = by_task_policy.get((task, "entropy_budget"))
        if challenger is None:
            continue
        challenger_outcomes = {
            int(doc_id): float(value)
            for doc_id, value in challenger["primary_outcomes_by_doc_id"].items()
        }
        for reference_policy in ("fixed_k2", "fixed_k4"):
            reference = by_task_policy.get((task, reference_policy))
            if reference is None:
                continue
            reference_outcomes = {
                int(doc_id): float(value)
                for doc_id, value in reference["primary_outcomes_by_doc_id"].items()
            }
            if set(challenger_outcomes) != set(reference_outcomes):
                raise ValueError(
                    f"Paired document IDs differ for {task}: entropy_budget versus "
                    f"{reference_policy}."
                )
            doc_ids = sorted(challenger_outcomes)
            differences = [
                challenger_outcomes[doc_id] - reference_outcomes[doc_id]
                for doc_id in doc_ids
            ]
            comparisons.append(
                {
                    "task": task,
                    "challenger": "entropy_budget",
                    "reference": reference_policy,
                    "example_count": len(doc_ids),
                    "challenger_correct": sum(challenger_outcomes.values()),
                    "reference_correct": sum(reference_outcomes.values()),
                    "accuracy_difference": statistics.fmean(differences),
                    "challenger_only_wins": sum(value > 0 for value in differences),
                    "reference_only_wins": sum(value < 0 for value in differences),
                    "paired_ties": sum(value == 0 for value in differences),
                    "paired_bootstrap_ci95": _paired_bootstrap_interval(
                        differences,
                        seed=42 + comparison_index,
                    ),
                    "bootstrap_resamples": 10_000,
                }
            )
            comparison_index += 1
    return comparisons


def _frontier(runs: list[dict[str, object]], task: str) -> list[str]:
    """Return policies not dominated on exploratory quality and generation time."""
    task_runs = [run for run in runs if run["task"] == task]
    frontier = []
    for candidate in task_runs:
        quality = float(candidate["primary_quality_exploratory"])
        seconds = float(candidate["generation_total_seconds"])
        dominated = any(
            float(other["primary_quality_exploratory"]) >= quality
            and float(other["generation_total_seconds"]) <= seconds
            and (
                float(other["primary_quality_exploratory"]) > quality
                or float(other["generation_total_seconds"]) < seconds
            )
            for other in task_runs
            if other is not candidate
        )
        if not dominated:
            frontier.append(str(candidate["policy"]))
    return frontier


def analyze(
    root: Path,
    *,
    stage: str,
    verifier: str,
    candidate_budget: int,
) -> dict[str, object]:
    """Validate an expected stage matrix and build the FG7 review payload."""
    if stage == "calibration":
        policies = CALIBRATION_POLICIES
    elif stage == "comparison":
        policies = COMPARISON_POLICIES
    elif stage == "confirmation":
        policies = CONFIRMATION_POLICIES
    else:
        raise ValueError("stage must be calibration, comparison, or confirmation.")
    runs = []
    traces = []
    failures = []
    for task in TASKS:
        for policy in policies:
            try:
                run, examples = _load_run(
                    root,
                    verifier,
                    task,
                    policy,
                    candidate_budget,
                    require_samples=stage == "confirmation",
                )
                runs.append(run)
                if examples:
                    traces.append(
                        {
                            "task": task,
                            "policy": policy,
                            **examples[0],
                        }
                    )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "task": task,
                        "policy": policy,
                        "error": str(error),
                    }
                )
    expected_count = len(TASKS) * len(policies)
    return {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "stage": stage,
        "verifier": verifier,
        "candidate_budget": candidate_budget,
        "expected_run_count": expected_count,
        "completed_run_count": len(runs),
        "failures": failures,
        "interpretation": (
            (
                "The fresh paired confirmation evaluates fixed k=2, fixed k=4, "
                "and the calibration-frozen entropy-budget policy. It can resolve "
                "G7 provisionally but is not the multi-task Phase-8 benchmark."
            )
            if stage == "confirmation"
            else (
                "Calibration and eight-example quality are exploratory. Action size, "
                "stage, NFE, cumulative uncertainty, and large-early consistency are "
                "the mechanism signals; G7 requires a held-out quality-efficiency gain."
            )
        ),
        "runs": runs,
        "quality_efficiency_frontier_exploratory": {
            task: _frontier(runs, task) for task in TASKS
        },
        "paired_quality_comparisons": (
            _paired_quality_comparisons(runs) if stage == "confirmation" else []
        ),
        "complete_first_example_traces": traces,
    }


def main() -> None:
    """Parse arguments and atomically replace the requested JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("calibration", "comparison", "confirmation"),
        required=True,
    )
    parser.add_argument("--verifier", choices=("entropy_drop",), default="entropy_drop")
    parser.add_argument("--candidate-budget", type=int, default=4)
    args = parser.parse_args()
    if args.candidate_budget <= 0:
        parser.error("--candidate-budget must be positive")
    summary = analyze(
        args.input_directory.resolve(),
        stage=args.stage,
        verifier=args.verifier,
        candidate_budget=args.candidate_budget,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_path)
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
