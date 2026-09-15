"""Test W&B telemetry locally using a fake client and synthetic artifacts.

Source /home/sarthak.malla/.zshrc, activate dllm, then run on a compute node:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_telemetry.py -q
These tests never contact W&B, load a checkpoint, or launch a compute job.
"""

import json
from types import SimpleNamespace
import sys

import pytest

from examples.path_selection.experiments.artifacts import atomic_json
from examples.path_selection.experiments.telemetry import ExperimentLogger


class FakeRun:
    """Capture scalar payloads and final status without network access."""

    def __init__(self):
        self.logs = []
        self.summary = {}
        self.exit_codes = []

    def log(self, values):
        self.logs.append(dict(values))

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


def _client(monkeypatch):
    calls, runs = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        runs.append(FakeRun())
        return runs[-1]

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=initialize, Settings=lambda **kwargs: kwargs,
    ))
    return calls, runs


def _write(root, kind, key, record):
    atomic_json(root / kind / f"{key}.json", record)
    return record


def test_resume_rebuilds_completed_totals_without_double_counting(monkeypatch, tmp_path):
    calls, runs = _client(monkeypatch)
    configuration = {"suite": "test", "document_ids": [0, 1], "prompt": "PRIVATE PROMPT"}
    record = {
        "arm": "cheap", "doc_id": 0, "flexible_correct": True, "strict_correct": False,
        "generation_seconds": 2.5, "accounting": {"model_calls": 8, "evaluated_rows": 8},
        "actions": [{"size": 2, "committed_token_ids": [10, 20]}],
        "response": "PRIVATE RESPONSE", "generated_token_ids": [10, 20],
    }
    ledger = {"phase": "benchmark_cheap", "doc_id": 0, "completed": True,
              "accounting": {"model_calls": 8, "evaluated_rows": 8}, "wall_seconds": 2.5}
    with ExperimentLogger(tmp_path, configuration, "benchmark_cheap", mode="offline") as logger:
        _write(tmp_path, "benchmarks", "cheap-doc0", record)
        assert logger.log_unit("benchmarks", record, unit_id="cheap-doc0")
        log_count = len(runs[0].logs)
        assert not logger.log_unit("benchmarks", record, unit_id="cheap-doc0")
        assert len(runs[0].logs) == log_count
        _write(tmp_path, "ledger", "ledger0", ledger)
        logger.log_unit("ledger", ledger, unit_id="ledger0")
    first_id = calls[0]["id"]
    assert runs[0].summary["work/model_calls"] == 8
    assert runs[0].summary["accuracy/flexible_rate"] == 1
    assert runs[0].summary["actions/mean_size"] == 2
    assert runs[0].summary["progress/documents"] == 1
    with ExperimentLogger(tmp_path, configuration, "benchmark_cheap", mode="offline") as logger:
        assert not logger.log_unit("benchmarks", record, unit_id="cheap-doc0")
    assert calls[1]["id"] == first_id
    assert calls[1]["resume"] == "allow"
    assert runs[1].summary["work/model_calls"] == 8
    assert runs[1].summary["progress/documents"] == 1
    exported = json.dumps({"config": calls[0]["config"], "logs": runs[0].logs, "summary": runs[0].summary})
    assert "PRIVATE" not in exported
    assert "generated_token_ids" not in exported
    assert "committed_token_ids" not in exported
    assert runs[0].exit_codes == [0]


def test_existing_units_and_paired_outcomes_are_logged_as_scalar_counts(monkeypatch, tmp_path):
    _, runs = _client(monkeypatch)
    _write(tmp_path, "branches", "wrong", {"flexible_correct": False})
    _write(tmp_path, "branches", "right", {"flexible_correct": True})
    record = {
        "experiment": "precedence", "state_key": "doc0-r0", "doc_id": 0,
        "probes": [{"pool": "adaptive", "any_flip": True,
                    "features": [{"flipped": True, "probability_loss": .3},
                                 {"flipped": False, "probability_loss": -.1}]}],
        "groups": [{"pool": "adaptive", "label": "natural",
                    "simultaneous_branch": "wrong", "seed_first_branch": "right",
                    "reverse_branch": "wrong"}],
    }
    _write(tmp_path, "diagnostics", "precedence-doc0-r0", record)
    # Artifacts from other stages share a worker directory but not this run's totals.
    _write(tmp_path, "benchmarks", "other-doc0", {"arm": "other", "doc_id": 0, "flexible_correct": False})
    with ExperimentLogger(tmp_path, {}, "diagnose_precedence", mode="offline"):
        pass
    summary = runs[0].summary
    assert summary["progress/states"] == 1
    assert summary["progress/unique_documents"] == 1
    assert summary["seed/adaptive/flip_rate"] == .5
    assert summary["seed/adaptive/mean_probability_loss"] == pytest.approx(.1)
    assert summary["precedence/adaptive/natural/seed_first/wins"] == 1
    assert summary["precedence/adaptive/natural/reverse/both_wrong"] == 1
    assert "accuracy/flexible_count" not in summary


def test_selector_agreement_and_cheap_wins_have_explicit_direction(monkeypatch, tmp_path):
    _, runs = _client(monkeypatch)
    _write(tmp_path, "branches", "entropy", {"flexible_correct": False})
    _write(tmp_path, "branches", "cheap", {"flexible_correct": True})
    record = {"experiment": "selectors", "doc_id": 0, "comparisons": [{
        "pool": "fixed4", "exact_agreement": False, "jaccard": .6,
        "entropy_size": 4, "cheap_size": 4,
        "entropy_branch": "entropy", "cheap_branch": "cheap",
    }]}
    _write(tmp_path, "diagnostics", "selectors-doc0", record)
    with ExperimentLogger(tmp_path, {}, "diagnose_selectors", mode="offline"):
        pass
    assert runs[0].summary["selectors/fixed4/wins"] == 1
    assert runs[0].summary["selectors/fixed4/agreement_rate"] == 0
    assert runs[0].summary["selectors/fixed4/mean_jaccard"] == .6


def test_disabled_mode_never_initializes_wandb_and_worker_ids_are_distinct(monkeypatch, tmp_path):
    calls, _ = _client(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    with ExperimentLogger(tmp_path, {}, "collect", worker_index=0, worker_count=2) as first:
        first_id = first.run_id
    with ExperimentLogger(tmp_path, {}, "collect", worker_index=1, worker_count=2) as second:
        assert second.run_id != first_id
    assert not calls


def test_initialization_error_is_actionable_and_keeps_run_identity(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise RuntimeError("synthetic authentication error")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail, Settings=lambda **kwargs: kwargs))
    with pytest.raises(RuntimeError, match="authentication/network.*offline"):
        with ExperimentLogger(tmp_path, {}, "collect", mode="online"):
            pass
    assert (tmp_path / "telemetry" / "collect-worker00.json").is_file()


def test_exception_finishes_run_once_without_exporting_exception_text(monkeypatch, tmp_path):
    _, runs = _client(monkeypatch)
    with pytest.raises(ValueError, match="PRIVATE"):
        with ExperimentLogger(tmp_path, {}, "collect", mode="offline"):
            raise ValueError("PRIVATE PROMPT IN EXCEPTION")
    assert runs[0].exit_codes == [1]
    assert runs[0].summary["invocation/error_type"] == "ValueError"
    assert "PRIVATE" not in json.dumps(runs[0].summary)
