"""Run: python -m pytest /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_replay_diagnostics.py -q."""

import json
import sys

import pytest
import torch

from scripts.tse.ctca import benchmark_replay, projection_temperature_sweep


def test_temperature_sweep_selects_only_full_anchor_baseline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(projection_temperature_sweep, "ROOT", tmp_path)
    directory = tmp_path / ".cache" / "ctca"
    directory.mkdir(parents=True)
    baseline = {
        "auxiliary_id": "aux", "master_id": "master",
        "anchor_temperature": 0.01, "projection_temperature": 0.05,
        "projection_top_k": 64, "selected_anchors": 5000, "total_anchors": 5000,
    }
    for index, changes in enumerate((
        {"anchor_temperature": 0.03}, {"projection_temperature": 0.1},
        {"selected_anchors": 2048}, {"projection_top_k": 32},
    )):
        torch.save({"metadata": {**baseline, **changes}},
                   directory / f"relative_sparse_topk_wrong{index}.pt")
    with pytest.raises(FileNotFoundError, match="baseline projection"):
        projection_temperature_sweep.load_cache("aux", "master")
    torch.save({"metadata": baseline}, directory / "relative_sparse_topk_baseline.pt")
    assert projection_temperature_sweep.load_cache("aux", "master")["metadata"] == baseline
    torch.save({"metadata": baseline}, directory / "relative_sparse_topk_duplicate.pt")
    with pytest.raises(ValueError, match="Ambiguous"):
        projection_temperature_sweep.load_cache("aux", "master")


def test_replay_preserves_artifacts_for_different_anchor_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_replay, "ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    sample = {"doc_hash": "same", "arguments": {"gen_args_0": {"arg_0": "prompt"}},
              "doc": {"question": "question"}, "target": "8", "resps": [["answer"]]}
    monkeypatch.setattr(benchmark_replay, "samples", lambda job: {23: sample})

    def run(command, **kwargs):
        trace = command[command.index("--trace-jsonl") + 1]
        with open(trace, "w") as stream:
            stream.write(json.dumps({"event": "question_result", "response": "answer"}))

    monkeypatch.setattr(benchmark_replay.subprocess, "run", run)
    for anchors in (2048, 4096, "auto"):
        monkeypatch.setattr(sys, "argv", [
            "benchmark_replay.py", "--study-id", "existing-study", "--case", "23:b",
            "--condition", "ensemble", "--projection-mode", "relative_distribution",
            "--num-anchors", str(anchors),
        ])
        benchmark_replay.main()
    study = tmp_path / ".logs" / "existing-study"
    for anchors in (2048, 4096, "auto"):
        suffix = f"-relative_distribution-{anchors}"
        manifest = json.loads((study / f"doc23{suffix}.json").read_text())
        assert manifest["num_anchors"] == anchors
        assert manifest["reproduction"][0]["matches_original_response"]
        assert (tmp_path / manifest["traces"][0].lstrip("/")).is_file()
        assert (study / "test" / f"doc23{suffix}-reproduction.json").is_file()
