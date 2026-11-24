"""Run: pytest /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_master_gate.py -q."""

import importlib.util
import csv
import json
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "tse/sandbox/ctca_master_gate_sandbox.py"
SPEC = importlib.util.spec_from_file_location("master_gate_sandbox", PATH)
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)


@pytest.mark.parametrize("metric", ["js", "overlap"])
def test_agreement_extremes_and_uncertainty(metric):
    p = torch.tensor([[1., 0, 0, 0]])
    q = torch.tensor([[0., 1, 0, 0]])
    assert sandbox.measures(p, p)[metric].item() == pytest.approx(1)
    assert sandbox.measures(p, q)[metric].item() == pytest.approx(0)
    uniform = torch.full((1, 4), .25)
    assert sandbox.measures(uniform, uniform)["master_uncertainty"].item() == pytest.approx(1)
    assert sandbox.gate(uniform, uniform, metric)[1].item() is False


def test_both_and_one_rules_and_boundary():
    p = torch.tensor([[1., 0, 0, 0]])
    q = torch.full((1, 4), .25)
    assert not sandbox.gate(p, q, rule="both", agreement_threshold=0)[1].item()
    assert sandbox.gate(p, q, rule="one", agreement_threshold=0)[1].item()
    assert sandbox.gate(p, p, agreement_threshold=1, uncertainty_threshold=0)[1].item()
    assert sandbox.gate(p, q, rule="both", agreement_threshold=0,
                        uncertainty_threshold=1)[1].item()


def test_master_fallback_empty_and_normalization():
    p = torch.tensor([[.8, .2, 0, 0]])
    q = torch.tensor([[0., .95, .05, 0]])
    output, accepted, _ = sandbox.gate(p, q, agreement_threshold=1)
    assert not accepted.item()
    assert torch.equal(output, p)
    output, _, _ = sandbox.gate(p, q, agreement_threshold=0, uncertainty_threshold=1)
    assert output.sum().item() == pytest.approx(1)
    empty, decisions, _ = sandbox.gate(torch.empty(0, 4), torch.empty(0, 4))
    assert empty.shape == (0, 4) and decisions.numel() == 0


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        sandbox.gate(torch.full((1, 4), .25), torch.full((1, 4), .25),
                     agreement_threshold=threshold)


def test_grid_projection_and_fixed_budget():
    grid = sandbox.simplex()
    assert grid.shape == (286, 4)
    assert torch.allclose(grid.sum(-1), torch.ones(286))
    projected = torch.full((1, 4), .25) @ sandbox.projection_maps()["collapsed"]
    assert sandbox.entropy(projected).item() == 0
    report = sandbox.canvas_report()
    assert report["accepted"] == [False, True, True]
    assert len(report["ungated"]["selected_tokens"]) == len(report["gated"]["selected_tokens"]) == 2
    assert report["ungated"]["proposals"] != report["gated"]["proposals"]
    assert report["ungated"]["selected_positions"] != report["gated"]["selected_positions"]
    assert sandbox.tied(torch.full((1, 4), .25)).item()


def test_labelled_examples_are_deterministic_and_exclude_ties():
    report = sandbox.example_report()
    assert report == sandbox.example_report()
    assert all(row["correct"] is None for row in report if row["argmax_tie"])


def test_correlated_answers_are_balanced_and_collapse_is_not_always_correct():
    cases = sandbox.correlated_cases()
    assert len(cases) == 144
    assert cases == sandbox.correlated_cases()
    answers = torch.tensor([case["answer"] for case in cases])
    assert torch.bincount(answers).tolist() == [36, 36, 36, 36]
    for name in {case["case"] for case in cases}:
        labels = [case["answer"] for case in cases if case["case"] == name]
        assert [labels.count(index) for index in range(4)] == [6, 6, 6, 6]
    native = torch.tensor([case["auxiliary"] for case in cases])
    collapsed = native @ sandbox.projection_maps()["collapsed"]
    assert int((collapsed.argmax(-1) == answers).sum()) == 36


def test_harm_prevention_and_rejected_corrections_have_fixed_denominator():
    cases = sandbox.correlated_cases()
    p = torch.tensor([case["master"] for case in cases])
    q = torch.tensor([case["auxiliary"] for case in cases])
    answers = torch.tensor([case["answer"] for case in cases])
    blend = sandbox.fuse_probabilities(p, q, .5)
    output, accepted, _ = sandbox.gate(p, q)
    counts = sandbox.outcome_counts(p, blend, output, accepted, answers)
    assert counts["labelled_comparable_cases"] == 144
    assert counts["labelled_master_correct"] == 72
    assert counts["labelled_ungated_correct"] == 96
    assert counts["labelled_prevented_harm"] == 24
    assert counts["labelled_rejected_corrections"] == 48
    assert counts["labelled_gated_correct"] == 72


def test_report_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "simplex", lambda: torch.eye(4))
    sandbox.run(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["device"] == "cpu" and report["pairs_per_mapping"] == 16
    assert report["version"] == 2 and len(report["correlated_cases"]) == 144
    with (tmp_path / "thresholds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2898
    assert {row["rule"] for row in rows} == {"both", "one", "agreement_only"}
    assert {int(row["labelled_comparable_cases"]) for row in rows if row["mapping"] == "identity"} == {144}
    for name in sandbox.projection_maps():
        assert (tmp_path / f"{name}_summary.json").exists()
