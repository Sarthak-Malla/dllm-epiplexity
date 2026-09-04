"""Test Phase-7 cardinality aggregation with constructed sidecars.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p7_cardinality_analysis.py -v
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

from analyze_p7_cardinality import (  # noqa: E402
    CALIBRATION_POLICIES,
    CONFIRMATION_POLICIES,
    TASKS,
    _frontier,
    analyze,
)


def _write_json(path: Path, value: object) -> None:
    """Write one constructed JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, values: list[object]) -> None:
    """Write constructed JSONL artifacts consumed by paired analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _step(step_index: int, action_size: int) -> dict[str, object]:
    """Build one diagnostic record consumed by the analyzer."""
    positions = list(range(action_size))
    return {
        "step_index": step_index,
        "mask_ratio": 1.0 if step_index == 0 else 0.4,
        "commit_k": action_size,
        "captured_base_forward_count": 1,
        "lookahead_model_calls": 3,
        "current_state_metric_sum": 2.0 - step_index,
        "candidate_collapse": False,
        "immediate_token_consistency_count": 0 if step_index == 0 else 3,
        "immediate_token_consistency_total": 0 if step_index == 0 else 4,
        "selected_candidate": {
            "valid": True,
            "positions": positions,
            "raw_verifier_score": 2.0,
            "size_aware_verifier_score": 0.5,
            "immediate_action_cost": 0.2,
        },
        "cardinality_strategy": "joint_k",
        "maximum_action_size": 4,
        "allowed_action_sizes": [1, 2, 4],
        "size_scoring_rule": "per_token",
        "utility_threshold": 0.25,
        "entropy_budget": 1.0,
        "immediate_cost_weight": 1.0,
        "size_penalty": 0.25,
    }


def test_analysis_reports_stage_k_nfe_and_large_early_failures(tmp_path):
    verifier = "entropy_drop"
    budget = 4
    for task in TASKS:
        metric_name = (
            "exact_match,flexible-extract"
            if task == "gsm8k_cot"
            else "pass@1,create_test"
        )
        for policy_index, policy in enumerate(CALIBRATION_POLICIES):
            run = tmp_path / verifier / task / policy / f"n{budget}"
            _write_json(
                run / "results_2026.json",
                {
                    "results": {task: {metric_name: policy_index / 10}},
                    "total_evaluation_time_seconds": 12.0,
                },
            )
            _write_json(
                run / "results.json_entropy_drop_diagnostics.json",
                [{"example_index": 0, "steps": [_step(0, 4), _step(1, 2)]}],
            )
            _write_json(
                run / "results.json_entropy_drop_runtime.json",
                {
                    "generation_total_seconds": 20.0 - policy_index,
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 200,
                },
            )

    summary = analyze(
        tmp_path,
        stage="calibration",
        verifier=verifier,
        candidate_budget=budget,
    )

    assert summary["status"] == "complete"
    assert summary["completed_run_count"] == 22
    run = summary["runs"][0]
    assert run["action_size"]["mean"] == 3.0
    assert run["action_size_counts"] == {"2": 1, "4": 1}
    assert run["action_size_by_mask_stage"]["early"]["mean"] == 4.0
    assert run["action_size_by_mask_stage"]["middle_late"]["mean"] == 2.0
    assert run["base_forward_count"] == 2
    assert run["lookahead_forward_count"] == 6
    assert run["total_model_forward_count"] == 8
    assert run["large_early_action_count"] == 1
    assert run["large_early_token_failure_rate"] == pytest.approx(0.25)
    assert run["candidate_collapse_count"] == 0
    assert run["wrong_action_width_count"] == 0
    assert len(summary["complete_first_example_traces"]) == 22


def test_frontier_removes_slower_policy_with_no_quality_gain():
    runs = [
        {
            "task": "gsm8k_cot",
            "policy": "dominated",
            "primary_quality_exploratory": 0.5,
            "generation_total_seconds": 20.0,
        },
        {
            "task": "gsm8k_cot",
            "policy": "frontier",
            "primary_quality_exploratory": 0.5,
            "generation_total_seconds": 10.0,
        },
    ]

    assert _frontier(runs, "gsm8k_cot") == ["frontier"]


def test_confirmation_stage_requires_six_frozen_policy_cells(tmp_path):
    verifier = "entropy_drop"
    budget = 4
    for task in TASKS:
        metric_name = (
            "exact_match,flexible-extract"
            if task == "gsm8k_cot"
            else "pass@1,create_test"
        )
        for policy in CONFIRMATION_POLICIES:
            run = tmp_path / verifier / task / policy / f"n{budget}"
            _write_json(
                run / "results_2026.json",
                {
                    "results": {task: {metric_name: 0.5}},
                    "total_evaluation_time_seconds": 12.0,
                },
            )
            _write_json(
                run / "results.json_entropy_drop_diagnostics.json",
                [{"example_index": 0, "steps": [_step(0, 2)]}],
            )
            _write_json(
                run / "results.json_entropy_drop_runtime.json",
                {
                    "generation_total_seconds": 10.0,
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 200,
                },
            )
            filter_name = (
                "flexible-extract" if task == "gsm8k_cot" else "create_test"
            )
            sample_metric_name = (
                "exact_match" if task == "gsm8k_cot" else "pass@1"
            )
            _write_jsonl(
                run / f"samples_{task}_2026.jsonl",
                [
                    {
                        "doc_id": 16,
                        "filter": filter_name,
                        sample_metric_name: (
                            1.0 if policy == "entropy_budget" else 0.0
                        ),
                    },
                    {
                        "doc_id": 17,
                        "filter": filter_name,
                        sample_metric_name: 0.0,
                    },
                ],
            )

    summary = analyze(
        tmp_path,
        stage="confirmation",
        verifier=verifier,
        candidate_budget=budget,
    )

    assert summary["status"] == "complete"
    assert summary["expected_run_count"] == 6
    assert summary["completed_run_count"] == 6
    assert {run["policy"] for run in summary["runs"]} == set(
        CONFIRMATION_POLICIES
    )
    assert "fresh paired confirmation" in summary["interpretation"].lower()
    assert len(summary["paired_quality_comparisons"]) == 4
    paired = summary["paired_quality_comparisons"][0]
    assert paired["example_count"] == 2
    assert paired["challenger_correct"] == 1.0
    assert paired["reference_correct"] == 0.0
    assert paired["accuracy_difference"] == 0.5
    assert paired["challenger_only_wins"] == 1
    assert paired["reference_only_wins"] == 0
    assert paired["paired_ties"] == 1
    assert paired["paired_bootstrap_ci95"] == [0.0, 1.0]
