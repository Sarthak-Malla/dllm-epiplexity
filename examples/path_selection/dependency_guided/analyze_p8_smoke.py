"""Validate the frozen Phase-8 smoke matrix and estimate primary-run cost.

Run after the smoke Slurm job finishes:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_p8_smoke.py \
        --input-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p8/frozen_v1/smoke \
        --output-path /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p8/frozen_v1/smoke/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_PLAN = Path(
    "/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/"
    "dependency_guided/p8_evaluation_plan.json"
)


def _latest_result(run_directory: Path) -> Path:
    """Return the newest aggregate lm-eval result in one run directory."""
    candidates = [
        path
        for path in run_directory.glob("results_*.json")
        if "_diagnostics" not in path.name
        and "_runtime" not in path.name
        and "_candidates" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No lm-eval result below {run_directory}.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _quality_metrics(result: dict[str, object], task: str) -> dict[str, float]:
    """Extract scalar task metrics without aliases or standard errors."""
    task_result = result["results"][task]
    return {
        name: float(value)
        for name, value in task_result.items()
        if name != "alias"
        and "stderr" not in name
        and isinstance(value, (int, float))
    }


def _load_cell(
    root: Path,
    *,
    task: str,
    method: dict[str, object],
    seed: int,
) -> dict[str, object]:
    """Load and validate one smoke cell."""
    method_name = str(method["name"])
    sampler = str(method["sampler"])
    run_directory = root / task / method_name / f"seed{seed}" / "shard0000"
    result_path = _latest_result(run_directory)
    runtime_path = run_directory / f"results.json_{sampler}_runtime.json"
    diagnostics_path = run_directory / f"results.json_{sampler}_diagnostics.json"
    manifest_path = run_directory / "samples.json"
    for required in (runtime_path, diagnostics_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing Phase-8 artifact: {required}.")

    result = json.loads(result_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if len(manifest.get(task, [])) != 1:
        raise ValueError(f"Smoke manifest must contain one {task} example.")
    if len(diagnostics) != 1:
        raise ValueError(f"Smoke diagnostics must contain one {task} example.")

    steps = diagnostics[0].get("steps", [])
    invalid_selections = sum(
        not isinstance(step.get("selected_candidate"), dict)
        or step["selected_candidate"].get("valid") is not True
        for step in steps
    )
    candidate_collapses = sum(bool(step.get("candidate_collapse")) for step in steps)
    capture_lifecycle_failures = sum(
        bool(step.get("capture_active_before_lookahead"))
        or not bool(step.get("capture_tensors_released_before_lookahead", True))
        or int(step.get("lookahead_capture_forward_count", 0)) != 0
        for step in steps
    )
    if method_name != "native_confidence" and not steps:
        raise ValueError(f"Candidate method {method_name} has no diagnostics.")
    if invalid_selections or candidate_collapses or capture_lifecycle_failures:
        raise ValueError(
            f"Invariant failure for {task}/{method_name}: invalid="
            f"{invalid_selections}, collapse={candidate_collapses}, capture="
            f"{capture_lifecycle_failures}."
        )

    generation_seconds = float(runtime["generation_total_seconds"])
    if not math.isfinite(generation_seconds) or generation_seconds <= 0:
        raise ValueError(f"Invalid generation time for {task}/{method_name}.")
    return {
        "task": task,
        "method": method_name,
        "sample_id": int(manifest[task][0]),
        "quality_metrics_smoke_only": _quality_metrics(result, task),
        "generation_seconds_per_example": generation_seconds,
        "lm_eval_total_seconds": result.get("total_evaluation_time_seconds"),
        "cuda_peak_allocated_bytes": runtime.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": runtime.get("cuda_peak_reserved_bytes"),
        "action_count": len(steps),
        "model_forward_count": (
            sum(
                int(step.get("captured_base_forward_count", 0))
                + int(step.get("lookahead_model_calls", 0))
                for step in steps
            )
            if steps
            else None
        ),
        "candidate_collapse_count": candidate_collapses,
        "invalid_selection_count": invalid_selections,
        "capture_lifecycle_failure_count": capture_lifecycle_failures,
        "result_path": str(result_path.resolve()),
        "runtime_path": str(runtime_path.resolve()),
        "diagnostics_path": str(diagnostics_path.resolve()),
    }


def analyze(root: Path, plan_path: Path) -> dict[str, object]:
    """Validate every expected smoke cell and project primary GPU time."""
    plan = json.loads(plan_path.read_text())
    tasks = plan["tasks"]
    methods = plan["primary_methods"]
    seed = int(plan["shared_generation"]["seed"])
    shard_size = int(plan["sample_stages"]["primary"]["shard_size"])
    runs = []
    failures = []
    for task in tasks:
        for method in methods:
            try:
                runs.append(
                    _load_cell(
                        root,
                        task=task,
                        method=method,
                        seed=seed,
                    )
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "task": task,
                        "method": method["name"],
                        "error": str(error),
                    }
                )

    projections = []
    for run in runs:
        task_config = tasks[run["task"]]
        example_count = int(task_config["primary_stop"]) - int(
            task_config["primary_start"]
        )
        seconds = float(run["generation_seconds_per_example"])
        projections.append(
            {
                "task": run["task"],
                "method": run["method"],
                "primary_example_count": example_count,
                "projected_generation_gpu_hours": seconds * example_count / 3600,
                "projected_full_shard_seconds": seconds
                * min(shard_size, example_count),
                "fits_three_hour_limit_with_20_percent_margin": (
                    seconds * min(shard_size, example_count) <= 0.8 * 3 * 3600
                ),
            }
        )
    expected_count = len(tasks) * len(methods)
    return {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "interpretation": (
            "This is a one-example runtime and invariant smoke, not an accuracy "
            "result. Primary shard size remains accepted only if every projected "
            "cell fits the three-hour limit with a 20% margin."
        ),
        "expected_run_count": expected_count,
        "completed_run_count": len(runs),
        "failures": failures,
        "runs": runs,
        "primary_runtime_projections": projections,
        "all_projected_shards_fit": bool(projections)
        and len(projections) == expected_count
        and all(
            row["fits_three_hour_limit_with_20_percent_margin"]
            for row in projections
        ),
        "projected_total_generation_gpu_hours": sum(
            float(row["projected_generation_gpu_hours"])
            for row in projections
        ),
    }


def main() -> None:
    """Parse arguments and atomically write the smoke summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    summary = analyze(args.input_directory.resolve(), args.plan.resolve())
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
                "all_projected_shards_fit": summary["all_projected_shards_fit"],
                "projected_total_generation_gpu_hours": summary[
                    "projected_total_generation_gpu_hours"
                ],
                "output_path": str(args.output_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
