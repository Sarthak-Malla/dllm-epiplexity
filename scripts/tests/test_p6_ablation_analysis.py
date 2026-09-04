"""Test Phase-6 ablation aggregation with constructed sampler sidecars.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p6_ablation_analysis.py -v
"""

import json
from pathlib import Path
import sys

import pytest


ANALYSIS_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "path_selection"
    / "dependency_guided"
)
sys.path.insert(0, str(ANALYSIS_DIRECTORY))

from analyze_p6_ablation import (  # noqa: E402
    EXPECTED_COMMIT_K,
    EXPECTED_TASKS,
    EXPECTED_VARIANTS,
    analyze,
)


def _write_json(path: Path, value: object) -> None:
    """Write one constructed JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _step(
    *,
    step_index: int,
    conflict: float,
    consistency_count: int,
    consistency_total: int,
    state_metric: float,
) -> dict[str, object]:
    """Create the Phase-6 diagnostic fields consumed by the analyzer."""
    selected = {
        "verifier_score": 0.5,
        "anchor_support_sum": 0.25,
        "hard_fallback_count": 0,
    }
    return {
        "step_index": step_index,
        "selected_candidate": selected,
        "selected_set_mean_conflict": conflict,
        "selected_set_max_conflict": conflict + 0.1,
        "immediate_token_consistency_count": consistency_count,
        "immediate_token_consistency_total": consistency_total,
        "current_state_metric_mean": state_metric,
        "current_state_metric_sum": state_metric * 4,
    }


def test_analysis_summarizes_complete_matched_mechanism_matrix(tmp_path):
    verifier = "entropy_drop"
    budget = 4
    conflicts = {
        "correlated_together": 0.8,
        "top_confidence": 0.6,
        "hard_low_conflict": 0.1,
        "anchor_support_only": 0.3,
        "soft_no_anchor": 0.25,
        "soft_full": 0.2,
    }
    for task in EXPECTED_TASKS:
        metric_name = (
            "exact_match,flexible-extract"
            if task == "gsm8k_cot"
            else "pass@1,create_test"
        )
        for variant_index, variant in enumerate(EXPECTED_VARIANTS):
            for commit_k in EXPECTED_COMMIT_K:
                run = (
                    tmp_path
                    / verifier
                    / task
                    / variant
                    / f"k{commit_k}"
                    / f"n{budget}"
                )
                _write_json(
                    run / "results_2026.json",
                    {
                        "results": {
                            task: {
                                "alias": task,
                                metric_name: variant_index / 10,
                            }
                        },
                        "total_evaluation_time_seconds": 12.0,
                    },
                )
                consistency = 2 if variant == "soft_full" else 1
                steps = [
                    _step(
                        step_index=0,
                        conflict=conflicts[variant],
                        consistency_count=0,
                        consistency_total=0,
                        state_metric=0.8,
                    ),
                    _step(
                        step_index=1,
                        conflict=conflicts[variant],
                        consistency_count=consistency,
                        consistency_total=2,
                        state_metric=(
                            0.3 if variant == "soft_full" else 0.4
                        ),
                    ),
                ]
                _write_json(
                    run / "results.json_entropy_drop_diagnostics.json",
                    [{"example_index": 0, "steps": steps}],
                )
                _write_json(
                    run / "results.json_entropy_drop_runtime.json",
                    {
                        "generation_total_seconds": 4.0,
                        "cuda_peak_allocated_bytes": 100,
                        "cuda_peak_reserved_bytes": 200,
                    },
                )

    summary = analyze(tmp_path, verifier, budget)

    assert summary["status"] == "complete"
    assert summary["completed_run_count"] == 24
    assert summary["expected_run_count"] == 24
    assert len(summary["mechanism_comparisons"]) == 4
    assert len(summary["complete_soft_full_traces"]) == 4
    comparison = summary["mechanism_comparisons"][0]
    assert comparison["hard_minus_top_confidence_mean_conflict"] == pytest.approx(-0.5)
    assert comparison["hard_minus_correlated_mean_conflict"] == pytest.approx(-0.7)
    assert comparison["full_minus_no_anchor_immediate_consistency_rate"] == pytest.approx(0.5)
    assert comparison["full_minus_no_anchor_next_state_metric_mean"] == pytest.approx(-0.1)
    full_run = next(
        run
        for run in summary["runs"]
        if run["task"] == "gsm8k_cot"
        and run["variant"] == "soft_full"
        and run["commit_k"] == 2
    )
    assert full_run["steps_per_example"]["mean"] == 2
    assert full_run["immediate_token_consistency"]["rate"] == 1.0
    assert full_run["selected_set_mean_conflict"]["mean"] == 0.2


def test_analysis_reports_missing_cells(tmp_path):
    summary = analyze(tmp_path, "entropy_drop", 4)

    assert summary["status"] == "incomplete"
    assert summary["completed_run_count"] == 0
    assert len(summary["failures"]) == 24
    assert summary["mechanism_comparisons"] == []
