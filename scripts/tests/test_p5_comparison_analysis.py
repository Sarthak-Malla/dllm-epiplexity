"""Test Phase-5 comparison aggregation with constructed result sidecars.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p5_comparison_analysis.py -v
"""

import json
from pathlib import Path
import sys

ANALYSIS_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "path_selection"
    / "dependency_guided"
)
sys.path.insert(0, str(ANALYSIS_DIRECTORY))

from analyze_p5_comparison import (  # noqa: E402
    EXPECTED_BUDGETS,
    EXPECTED_STRATEGIES,
    EXPECTED_TASKS,
    analyze,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_analysis_requires_and_summarizes_complete_matrix(tmp_path):
    verifier = "entropy_drop"
    for task in EXPECTED_TASKS:
        metric_name = (
            "exact_match,flexible-extract"
            if task == "gsm8k_cot"
            else "pass@1,create_test"
        )
        for strategy in EXPECTED_STRATEGIES:
            for budget in EXPECTED_BUDGETS:
                run = tmp_path / verifier / task / strategy / f"n{budget}"
                _write_json(
                    run / "results_2026.json",
                    {
                        "results": {
                            task: {"alias": task, metric_name: budget / 10}
                        },
                        "total_evaluation_time_seconds": 12.0,
                    },
                )
                step = {
                    "candidates": [
                        {
                            "valid": True,
                            "verifier_score": float(index),
                            "heldout_count": 1,
                        }
                        for index in range(budget)
                    ],
                    "selected_candidate": {
                        "verifier_score": float(budget - 1),
                        "heldout_count": 1,
                        "fallback": False,
                    },
                    "timing_seconds": {
                        "base_forward": 1.0,
                        "attention_reconstruction": 0.1,
                        "candidate_proposal_generation": 0.01,
                        "candidate_lookahead": 2.0,
                    },
                    "lookahead_model_calls": 1,
                    "sink_count": 1 if strategy == "dependency" else 0,
                    "candidate_collapse": False,
                    "margin_to_bf16_reference_ratio": 2.0,
                }
                _write_json(
                    run / "results.json_entropy_drop_diagnostics.json",
                    [
                        {"example_index": 0, "steps": [step]},
                        {"example_index": 1, "steps": [step]},
                    ],
                )
                _write_json(
                    run / "results.json_entropy_drop_runtime.json",
                    {
                        "generation_total_seconds": 4.0,
                        "cuda_peak_allocated_bytes": 100,
                        "cuda_peak_reserved_bytes": 200,
                    },
                )
        native = tmp_path / verifier / task / "native_confidence" / "n1"
        _write_json(
            native / "results_2026.json",
            {
                "results": {task: {"alias": task, metric_name: 0.5}},
                "total_evaluation_time_seconds": 3.0,
            },
        )
        _write_json(
            native / "results.json_greedy_runtime.json",
            {
                "generation_total_seconds": 1.0,
                "cuda_peak_allocated_bytes": 50,
                "cuda_peak_reserved_bytes": 75,
            },
        )

    summary = analyze(tmp_path, verifier)

    assert summary["status"] == "complete"
    assert summary["completed_run_count"] == 26
    assert summary["expected_run_count"] == 26
    assert len(summary["dependency_n4_vs_confidence_gumbel_n8"]) == 2
    assert len(summary["complete_dependency_n4_traces"]) == 2
    dependency_run = next(
        run
        for run in summary["runs"]
        if run["task"] == "gsm8k_cot"
        and run["strategy"] == "dependency"
        and run["candidate_budget"] == 4
    )
    assert dependency_run["selected_verifier_scores"]["mean"] == 3.0
    assert dependency_run["sink_count"]["total"] == 2
