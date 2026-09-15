"""Check experiment resume identities and actual-forward accounting.

Source /home/sarthak.malla/.zshrc and activate the dllm conda environment, then run:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:20:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_artifacts.py
These synthetic tests require no checkpoint or GPU; the user runs them on compute.
"""

import json

import pytest
import torch

from examples.path_selection.experiments.accounting import ForwardAccounting
from examples.path_selection.experiments.artifacts import RunStore, stable_hash


def test_resume_rejects_changed_source_and_keeps_completed_units(tmp_path):
    manifest = {"documents": [0, 1], "source_hashes": {"sampler": "first"}}
    store = RunStore(tmp_path / "run", manifest)
    store.put_json("branches", "first", {"tokens": [3, 4], "correct": False})
    resumed = RunStore(tmp_path / "run", manifest, resume=True)
    assert resumed.get_json("branches", "first")["tokens"] == [3, 4]
    with pytest.raises(FileExistsError):
        RunStore(tmp_path / "run", manifest)
    with pytest.raises(ValueError, match="Resume rejected"):
        RunStore(tmp_path / "run", {**manifest, "source_hashes": {"sampler": "second"}}, resume=True)
    with pytest.raises(ValueError, match="overwrite"):
        resumed.put_json("branches", "first", {"tokens": [3, 5], "correct": True})


def test_complete_snapshot_identity_includes_rng_anchors_and_values(tmp_path):
    state = {"x": torch.tensor([[1, 2, 0]]), "rng": torch.arange(12, dtype=torch.uint8),
             "anchors": {"confidence": torch.tensor([[0.9, 0.8, 0.0]])}, "step": 2}
    store = RunStore(tmp_path, {"checkpoint": "pinned"})
    store.put_snapshot("doc0-state0", state)
    restored = store.get_snapshot("doc0-state0")
    assert stable_hash(restored) == stable_hash(state)
    changed = {**state, "step": 3}
    assert stable_hash(changed) != stable_hash(state)
    with pytest.raises(ValueError, match="collision"):
        store.put_snapshot("doc0-state0", changed)
    # A corrupted snapshot cannot silently become a valid resume point.
    torch.save(changed, tmp_path / "snapshots" / "doc0-state0.pt")
    with pytest.raises(ValueError, match="payload"):
        store.get_snapshot("doc0-state0")


def test_bfloat16_and_scalar_identity_and_record_provenance(tmp_path):
    assert stable_hash(torch.tensor(1.0, dtype=torch.bfloat16)) != stable_hash(torch.tensor(2.0, dtype=torch.bfloat16))
    assert stable_hash(torch.tensor([1])) != stable_hash(torch.tensor([[1]]))
    store = RunStore(tmp_path, {"checkpoint": "pinned"})
    store.put_json("probes", "p0", {"flips": [False, True]})
    path = tmp_path / "probes" / "p0.json"
    record = json.loads(path.read_text())
    record["_manifest_hash"] = "another-run"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="another manifest"):
        store.get_json("probes", "p0")
    with pytest.raises(ValueError, match="component"):
        store.has("../outside", "p0")


class CountingModel(torch.nn.Module):
    """A deterministic batched model that also exercises exception accounting."""

    def forward(self, input_ids, fail=False):
        if fail:
            raise RuntimeError("designed failure")
        return input_ids.float() + 1


def test_calls_rows_and_nested_scopes_count_physical_work_once():
    model = CountingModel()
    with ForwardAccounting(model) as accounting:
        with accounting.scope("base"):
            model(torch.zeros((1, 7), dtype=torch.long))
            with accounting.scope("candidate"):
                # Four rows still count as four evaluations in one invocation.
                model(input_ids=torch.zeros((4, 7), dtype=torch.long))
            before = accounting.snapshot()
            model(input_ids=torch.zeros((2, 7), dtype=torch.long))
        delta = accounting.delta(before)
    total = accounting.snapshot()
    assert total["model_calls"] == 3
    assert total["evaluated_rows"] == 7
    assert total["input_tokens"] == 49
    assert total["by_label"]["candidate"]["evaluated_rows"] == 4
    assert delta["model_calls"] == 1
    assert delta["evaluated_rows"] == 2
    assert total["model_seconds"] >= 0
    model(torch.zeros((1, 7), dtype=torch.long))
    assert accounting.snapshot() == total  # Hook removal prevents later contamination.


def test_failed_forward_is_counted_and_hooks_are_removed():
    model = CountingModel()
    accounting = ForwardAccounting(model)
    with pytest.raises(RuntimeError, match="designed failure"):
        with accounting:
            model(input_ids=torch.zeros((3, 5), dtype=torch.long), fail=True)
    assert accounting.snapshot()["evaluated_rows"] == 3
    assert not accounting._handles
