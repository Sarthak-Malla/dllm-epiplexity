"""Test optional W&B progress telemetry without making network calls.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_wandb_monitor.py -v
"""

from types import SimpleNamespace
import sys

import pytest

from dllm.core.eval.wandb_monitor import (
    log_generation_progress,
    summarize_diagnostic_batch,
)


def _diagnostics():
    """Return two steps with conflict, anchors, and immediate consistency."""
    return [
        [
            {
                "step_index": 0,
                "selected_set_mean_conflict": 0.4,
                "selected_set_max_conflict": 0.7,
                "current_state_metric_mean": 0.8,
                "commit_k": 1,
                "immediate_token_consistency_count": 0,
                "immediate_token_consistency_total": 0,
                "selected_candidate": {
                    "anchor_support_sum": 0.0,
                    "hard_fallback_count": 0,
                    "raw_verifier_score": 0.4,
                    "size_aware_verifier_score": 0.4,
                    "immediate_action_cost": 0.2,
                },
            },
            {
                "step_index": 1,
                "selected_set_mean_conflict": 0.2,
                "selected_set_max_conflict": 0.5,
                "current_state_metric_mean": 0.3,
                "commit_k": 2,
                "immediate_token_consistency_count": 2,
                "immediate_token_consistency_total": 2,
                "selected_candidate": {
                    "anchor_support_sum": 0.6,
                    "hard_fallback_count": 1,
                    "raw_verifier_score": 0.8,
                    "size_aware_verifier_score": 0.35,
                    "immediate_action_cost": 0.6,
                },
            },
        ]
    ]


def test_diagnostic_batch_summary_matches_direct_aggregation():
    summary = summarize_diagnostic_batch(_diagnostics())

    assert summary["phase6/steps_in_batch"] == 2
    assert summary["phase6/mean_steps_per_example"] == 2
    assert summary["phase6/selected_set_mean_conflict"] == pytest.approx(0.3)
    assert summary["phase6/selected_set_max_conflict"] == pytest.approx(0.6)
    assert summary["phase6/next_state_metric_mean"] == pytest.approx(0.3)
    assert summary["phase6/selected_anchor_support_mean"] == pytest.approx(0.3)
    assert summary["phase6/immediate_token_consistency_rate"] == 1.0
    assert summary["phase6/hard_fallback_selection_count"] == 1
    assert summary["phase7/mean_action_size"] == 1.5
    assert summary["phase7/action_size_1_count"] == 1
    assert summary["phase7/action_size_2_count"] == 1
    assert summary["phase7/action_size_4_count"] == 0
    assert summary["phase7/selected_raw_verifier_score"] == pytest.approx(0.6)
    assert summary["phase7/selected_size_aware_verifier_score"] == pytest.approx(
        0.375
    )


def test_progress_logging_uses_active_run_and_includes_cuda(monkeypatch):
    payloads = []
    fake_run = SimpleNamespace(log=payloads.append)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=fake_run))

    logged = log_generation_progress(
        examples_completed=2,
        examples_total=8,
        batch_seconds=1.5,
        cumulative_seconds=3.0,
        diagnostics_by_example=_diagnostics(),
        cuda_memory={"peak_allocated_bytes": 123},
    )

    assert logged is True
    assert len(payloads) == 1
    assert payloads[0]["generation/progress_fraction"] == 0.25
    assert payloads[0]["generation/batch_seconds"] == 1.5
    assert payloads[0]["cuda/peak_allocated_bytes"] == 123
    assert payloads[0]["phase6/selected_set_mean_conflict"] == pytest.approx(0.3)


def test_progress_logging_is_a_noop_without_an_active_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))

    assert not log_generation_progress(
        examples_completed=1,
        examples_total=1,
        batch_seconds=1.0,
        cumulative_seconds=1.0,
        diagnostics_by_example=[],
    )


def test_monitoring_failure_does_not_escape(monkeypatch, caplog):
    def fail(_payload):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(run=SimpleNamespace(log=fail)),
    )

    assert not log_generation_progress(
        examples_completed=1,
        examples_total=1,
        batch_seconds=1.0,
        cumulative_seconds=1.0,
        diagnostics_by_example=[],
    )
    assert "telemetry unavailable" in caplog.text
