"""Test the experiment protocol on a compute node after preparing dllm.

Use the user's Slurm workflow with:
    python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_runner.py -v
The fixtures require no model checkpoint and never submit a job.
"""

from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch

from examples.path_selection.experiments import runner
from examples.path_selection.experiments import launch
from examples.path_selection.experiments import diagnostics as diagnostic_module
from examples.path_selection.experiments.artifacts import RunStore
from examples.path_selection.experiments.configs import (
    DOCUMENT_IDS, SNAPSHOT_THRESHOLDS, benchmark_configs, fixed_config, reference_config,
)
from examples.path_selection.experiments.diagnostics import (
    DiagnosticRunner, first_companion, pair_prepared,
)
from dllm.core.samplers.batched_lookahead import candidate_batch_from_mask_mapping
from dllm.core.samplers.decoding_state import PreparedStep


def test_fixed_protocol_and_one_mechanism_arms():
    assert DOCUMENT_IDS == tuple(range(100))
    assert SNAPSHOT_THRESHOLDS == (0, 85, 170)
    reference = reference_config()
    assert reference.dependency_confidence_exponent == 0
    assert reference.dependency_seed_strategy == "legacy"
    assert reference.dependency_commit_k == 1
    assert fixed_config().dependency_commit_k == 4
    assert fixed_config().dependency_seed_strategy == "legacy"
    arms = benchmark_configs()
    left, right = asdict(arms["reference_cheap"]), asdict(arms["affordable_companions"])
    assert {key for key in left if left[key] != right[key]} == {"dependency_budget_search"}
    for threshold in ("0.80", "0.90", "0.95"):
        left = asdict(arms[f"threshold_{threshold}_block64_confidence"])
        right = asdict(arms[f"threshold_{threshold}_block256_confidence"])
        assert {key for key in left if left[key] != right[key]} == {"block_size"}
        incoming = asdict(arms[f"threshold_{threshold}_block64_incoming"])
        assert {key for key in left if left[key] != incoming[key]} == {"confidence_ranking"}
    assert all(not config.diagnostic_metadata and not config.return_dict for config in arms.values())


def test_dry_run_has_no_model_load_or_output(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "must-not-exist"
    monkeypatch.setattr("sys.argv", ["runner.py", "--output-root", str(destination), "--dry-run", "collect"])
    monkeypatch.setattr(runner, "source_hashes", lambda: {})

    def forbidden():
        raise AssertionError("Dry-run tried to load a model.")

    monkeypatch.setattr(runner, "EvaluationContext", forbidden)
    runner.main()
    assert not destination.exists()
    assert '"document_work_range"' in capsys.readouterr().out


def test_two_workers_cover_exactly_one_hundred_problems():
    left = set(runner.assigned_document_ids(0, 2))
    right = set(runner.assigned_document_ids(1, 2))
    assert len(left) == len(right) == 50
    assert not left & right
    assert left | right == set(range(100))


def test_launcher_uses_separate_devices_and_shared_logging_group(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, wandb_mode="online", wandb_project="test",
                           wandb_group="shared", runner_args=["--resume", "collect"])
    commands = launch.worker_commands(args)
    for index, command in enumerate(commands):
        assert command[command.index("--device") + 1] == f"cuda:{index}"
        assert command[command.index("--output-root") + 1] == str(tmp_path / "workers" / f"worker{index}")
        assert command[command.index("--worker-index") + 1] == str(index)
        assert command[command.index("--worker-count") + 1] == "2"
        assert command[command.index("--wandb-group") + 1] == "shared"
        assert command[-2:] == ["--resume", "collect"]


def test_independent_workers_drop_inherited_collective_ranks(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PMI_RANK", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.setenv(name, "2")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    environment = launch.independent_worker_environment()
    assert environment["SLURM_JOB_ID"] == "1234"
    assert environment["OMP_NUM_THREADS"] == "12"
    assert not any(name in environment for name in
                   ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PMI_RANK", "OMPI_COMM_WORLD_SIZE"))


def test_launcher_dry_run_does_not_create_or_start_workers(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "not-created"
    monkeypatch.setattr("sys.argv", ["launch.py", "--output-root", str(destination), "--dry-run",
                                     "--", "--resume", "collect"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run started a worker.")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    launch.main()
    assert not destination.exists()
    assert '"worker_count": 2' in capsys.readouterr().out


def _prepared(positions, *, seed=1, construction_order=None):
    active = torch.tensor([[False, True, True, True, True]])
    mask = torch.zeros_like(active)
    mask[0, positions] = True
    candidates = candidate_batch_from_mask_mapping({"candidate": mask}, eligible_mask=active)
    metadata = dict(candidates.metadata[0])
    if construction_order is not None:
        metadata["construction_order_by_batch"] = (construction_order,)
    candidates = replace(candidates, seed_anchors=torch.tensor([[seed]]), metadata=(metadata,))
    state = SimpleNamespace(state_dict=lambda: {"tokens": torch.tensor([[9, 8, 8, 8, 8]])})
    probabilities = torch.tensor([[[0.8, 0.1, 0.1]] * 5])
    dependency = SimpleNamespace(query_positions=torch.tensor([[1, 2, 3, 4]]))
    return PreparedStep(
        state=state, config=reference_config(), base_forward=None,
        x0=torch.zeros((1, 5), dtype=torch.long), confidence=probabilities[..., 0],
        entropy=torch.ones((1, 5)), top2_margin=torch.full((1, 5), 0.7),
        probabilities=probabilities, candidates=candidates, dependency=dependency,
        active_mask=active, masked_active_mask=active, requested_k=torch.tensor([1]),
        step_seed=42, reconstruction_seconds=0.0, proposal_seconds=0.0, rng_after=42,
    )


def test_pair_uses_growth_order_and_keeps_canonical_positions():
    # Absolute order is [4, 3, 1]; sorting the set would incorrectly choose 1.
    prepared = _prepared([1, 3, 4], seed=4, construction_order=(3, 2, 0))
    assert first_companion(prepared, 0) == 3
    pair = pair_prepared(prepared, 0)
    assert pair.candidates.selected_positions[0, 0].tolist() == [3, 4]
    assert int(pair.candidates.seed_anchors[0, 0]) == 4


def test_singleton_probe_really_refreshes_before_cross_pool_reuse(tmp_path):
    class Accounting:
        def snapshot(self):
            return {}

        def delta(self, before):
            return {}

        def scope(self, label):
            return nullcontext()

    class Sampler:
        calls = 0

        def probe_seed_first(self, prepared, index, reverse=False):
            self.calls += 1
            assert int(prepared.candidates.candidate_masks[index].sum()) >= 2
            probabilities = prepared.probabilities.clone()
            probabilities[0, 2] = torch.tensor([0.1, 0.8, 0.1])
            return SimpleNamespace(refreshed_probabilities=probabilities,
                                   refreshed_token_ids=probabilities.argmax(-1))

    sampler = Sampler()
    store = RunStore(tmp_path / "artifacts", {"test": "universal_probe"})
    diagnostic = DiagnosticRunner(sampler, store, Accounting(), None)
    singleton = _prepared([1])
    first_key, first = diagnostic.probe(singleton, 0, state_key="doc00000-r000", doc_id=0)
    group = replace(singleton, candidates=_prepared([1, 2, 3]).candidates, config=fixed_config())
    second_key, second = diagnostic.probe(group, 0, state_key="doc00000-r000", doc_id=0)
    assert first_key == second_key
    assert sampler.calls == 1
    assert first["changes"]["2"]["flipped"] is True
    assert second["token_ids"][0][2] == 1
    assert len(list((store.root / "reuse").glob("*.json"))) == 1


def test_probe_commitment_preserves_logit_winner_when_bf16_probabilities_tie(tmp_path):
    class Accounting:
        def snapshot(self):
            return {}

        def delta(self, before):
            return {}

        def scope(self, label):
            return nullcontext()

    class Sampler:
        def probe_seed_first(self, prepared, index, reverse=False):
            probabilities = prepared.probabilities.clone().to(torch.bfloat16)
            # A rounded softmax can tie while the underlying logits select ID1.
            probabilities[0, 2] = torch.tensor([0.5, 0.5, 0.0], dtype=torch.bfloat16)
            ids = prepared.x0.clone()
            ids[0, 2] = 1
            return SimpleNamespace(refreshed_probabilities=probabilities, refreshed_token_ids=ids)

    store = RunStore(tmp_path / "artifacts", {"test": "logit_winner"})
    diagnostic = DiagnosticRunner(Sampler(), store, Accounting(), None)
    _, record = diagnostic.probe(_prepared([1, 2]), 0, state_key="doc00000-r000", doc_id=0)
    assert record["token_ids"][0][2] == 1
    assert record["changes"]["2"]["flipped"] is True


def test_pool_timings_count_one_shared_base_and_extra_absolute_reconstruction(monkeypatch):
    prepared = _prepared([1, 2])
    prepared.state.anchor_state = None
    prepared.state.response_mask = prepared.active_mask
    prepared.state.attention_mask = torch.ones_like(prepared.active_mask, dtype=torch.long)
    prepared.state.prompt_lens = [1]
    prepared.state.block_index = 0
    prepared.dependency.sink_mask = None
    prepared.base_forward = SimpleNamespace(base_forward_seconds=5.0)
    prepared.reconstruction_seconds = 1.0
    prepared.proposal_seconds = 2.0
    monkeypatch.setattr(diagnostic_module, "build_dependency_candidates",
                        lambda *args, **kwargs: (prepared.candidates, prepared.dependency, 3.0, 4.0))
    monkeypatch.setattr(diagnostic_module, "reconstruct_dependency_for_proposals",
                        lambda *args, **kwargs: prepared.dependency)
    monkeypatch.setattr(diagnostic_module, "release_dependency_capture_tensors", lambda base: None)
    clock_values = iter((10.0, 12.0))
    monkeypatch.setattr(diagnostic_module.time, "perf_counter", lambda: next(clock_values))
    pools, _, timings = diagnostic_module.prepare_pools(prepared, include_mass=True)
    assert len(pools) == 2
    assert timings["base_forward_seconds"] == 5.0
    assert timings["absolute_reconstruction_seconds"] == 2.0
    assert timings["reconstruction_seconds"] == 6.0
    assert timings["proposal_seconds"] == 6.0
