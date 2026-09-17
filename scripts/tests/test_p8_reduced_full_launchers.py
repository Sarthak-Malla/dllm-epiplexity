"""Validate Phase-8 launch configuration and path-selection evaluation controls.

Run on a compute node after preparing the environment:
    source /home/sarthak.malla/.zshrc
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_reduced_full_launchers.py -v
"""

import json
import importlib.util
from pathlib import Path

import pytest

from dllm.core.samplers.dependency_guided import validate_dependency_guided_config
from dllm.core.samplers.dependency_non_lookahead import (
    DependencyNonLookaheadSampler,
    DependencyNonLookaheadSamplerConfig,
)


REPO = Path("/home/sarthak.malla/dllm-selection-ensemble")
PLAN = REPO / "examples/path_selection/dependency_guided/p8_reduced_full_plan.json"
LAUNCHERS = {
    REPO / "examples/path_selection/run_gsm8k_full_dependency_fixed_k4.slurm.sh": (
        "gsm8k_cot",
        "dependency_fixed_k4_n4",
    ),
    REPO
    / "examples/path_selection/run_humaneval_full_dependency_fixed_k4.slurm.sh": (
        "humaneval_instruct",
        "dependency_fixed_k4_n4",
    ),
    REPO
    / "examples/path_selection/run_gsm8k_full_dependency_entropy_budget.slurm.sh": (
        "gsm8k_cot",
        "dependency_entropy_budget_n4",
    ),
    REPO
    / "examples/path_selection/run_humaneval_full_dependency_entropy_budget.slurm.sh": (
        "humaneval_instruct",
        "dependency_entropy_budget_n4",
    ),
}
SHARED_LAUNCHER = REPO / "examples/path_selection/run_p8_full_task.sh"
EVAL_PATH = REPO / "examples/path_selection/eval.py"
EVAL_SPEC = importlib.util.spec_from_file_location("p8_path_selection_eval", EVAL_PATH)
if EVAL_SPEC is None or EVAL_SPEC.loader is None:
    raise RuntimeError(f"Could not load path-selection evaluator from {EVAL_PATH}.")
EVAL_MODULE = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVAL_MODULE)


@pytest.fixture
def harness_without_model(monkeypatch):
    """Keep actual sampler configuration merging while skipping model loading."""
    def initialize(self, *, eval_config, sampler_config, sampler_cls, **kwargs):
        self.sampler_config = self._build_config(
            type(sampler_config), sampler_config, kwargs
        )
        self.sampler_cls = sampler_cls
        self._rank = 0
        self._world_size = 1
        self.accelerator = None

    monkeypatch.setattr(EVAL_MODULE.MDLMEvalHarness, "__init__", initialize)
    monkeypatch.setattr(EVAL_MODULE.torch.cuda, "is_available", lambda: False)
    return EVAL_MODULE.LLaDAPathSelectionEvalHarness


@pytest.mark.parametrize("candidate_budget", [4, 8])
def test_max_confidence_alias_preserves_corrected_adaptive_configuration(
    harness_without_model, candidate_budget, tmp_path
):
    harness = harness_without_model(
        sampler_type="max_confidence",
        candidate_budget=candidate_budget,
        dependency_cardinality_strategy="entropy_budget",
        dependency_entropy_budget=2.0,
        dependency_max_action_size=64,
        dependency_seed_strategy="incoming",
        dependency_seed_entropy_weight=1.0,
        dependency_confidence_exponent=0.0,
        dependency_direction="outgoing",
        dependency_generation_seed=42,
        block_size=64,
        temperature=0.0,
    )

    config = harness.sampler_config
    assert harness.sampler_cls is DependencyNonLookaheadSampler
    assert isinstance(config, DependencyNonLookaheadSamplerConfig)
    assert config.dependency_candidate_selector == "max_confidence"
    assert config.diagnostic_metadata is False
    assert config.proposal_strategy == "dependency"
    assert config.dependency_cardinality_strategy == "entropy_budget"
    assert config.dependency_entropy_budget == 2.0
    assert config.dependency_max_action_size == config.block_size == 64
    assert config.candidate_budget == candidate_budget
    assert config.dependency_seed_strategy == "incoming"
    assert config.dependency_seed_entropy_weight == 1.0
    assert config.dependency_confidence_exponent == 0.0
    assert config.dependency_direction == "outgoing"
    assert config.dependency_generation_seed == 42
    assert config.dependency_size_scoring == "per_token"
    validate_dependency_guided_config(config)

    harness.save_selected_candidates(str(tmp_path / "results.json"))
    runtime = json.loads(
        (tmp_path / "results.json_max_confidence_runtime.json").read_text()
    )
    assert runtime["sampler_type"] == "max_confidence"
    assert (tmp_path / "results.json_max_confidence_candidates.json").is_file()
    assert not (tmp_path / "results.json_entropy_drop_runtime.json").exists()


@pytest.mark.parametrize("selector", ["entropy_drop", "min_entropy"])
def test_max_confidence_alias_rejects_conflicting_selector(monkeypatch, selector):
    def unexpected_model_initialization(*args, **kwargs):
        pytest.fail("Conflicting selector must fail before model initialization.")

    monkeypatch.setattr(
        EVAL_MODULE.MDLMEvalHarness, "__init__", unexpected_model_initialization
    )
    with pytest.raises(ValueError, match="sampler_type='max_confidence' requires"):
        EVAL_MODULE.LLaDAPathSelectionEvalHarness(
            sampler_type="max_confidence", dependency_candidate_selector=selector
        )


def test_dependency_non_lookahead_alias_keeps_selector_override(harness_without_model):
    harness = harness_without_model(
        sampler_type="dependency_non_lookahead",
        dependency_candidate_selector="min_entropy",
    )

    assert harness.sampler_cls is DependencyNonLookaheadSampler
    assert harness.sampler_config.dependency_candidate_selector == "min_entropy"
    assert harness.sampler_type == "dependency_non_lookahead"


def test_diagnostic_call_summaries_survive_distributed_save(harness_without_model, tmp_path):
    from dllm.core.eval.decoding_summary import summarize_decoding_steps

    step = {
        "captured_base_forward_count": 1, "lookahead_model_calls": 0,
        "selected_candidate": {"index": 0, "action_size": 1},
    }
    for rank in (1, 0):
        harness = harness_without_model(sampler_type="max_confidence", diagnostic_metadata=True)
        assert harness.sampler_config.diagnostic_metadata is True
        harness._rank = rank
        harness._world_size = 2
        # Same request on both ranks represents distributed padding.
        harness.decoding_per_example = [{
            "rank": rank, "example_index": 0, "task_name": "gsm8k_cot",
            "doc_id": 7, "request_index": 0, "prompt_sha256": "same-prompt",
            "summary": summarize_decoding_steps([step, step]),
        }]
        harness.save_selected_candidates(str(tmp_path / "results.json"))
    runtime = json.loads((tmp_path / "results.json_max_confidence_runtime.json").read_text())
    summary = runtime["decoding_summary"]
    assert summary["unique_measured_examples"] == 1
    assert summary["duplicate_measured_records"] == 1
    assert summary["mean_model_calls_per_example"] == 2
    assert summary["model_calls_including_repeated_examples"] == 4
    manifest = json.loads((tmp_path / "results.json_max_confidence_decoding_manifest.json").read_text())
    assert all(Path(path).is_file() for path in manifest["shards"])


def test_reduced_plan_contains_only_the_four_new_full_cells():
    plan = json.loads(PLAN.read_text())

    assert set(plan["tasks"]) == {"gsm8k_cot", "humaneval_instruct"}
    assert {method["name"] for method in plan["methods"]} == {
        "dependency_fixed_k4_n4",
        "dependency_entropy_budget_n4",
    }
    assert plan["execution"]["job_count"] == 4
    assert plan["execution"]["gpus_per_job"] == 2
    entropy_method = next(
        method
        for method in plan["methods"]
        if method["name"] == "dependency_entropy_budget_n4"
    )
    assert entropy_method["possible_action_sizes"] == [1, 2, 3, 4]
    assert plan["tasks"]["gsm8k_cot"] | {
        "total_examples": 1319,
        "max_new_tokens": 256,
        "steps": 64,
        "block_size": 64,
    } == plan["tasks"]["gsm8k_cot"]
    assert plan["tasks"]["humaneval_instruct"] | {
        "total_examples": 164,
        "max_new_tokens": 1024,
        "steps": 256,
        "block_size": 256,
    } == plan["tasks"]["humaneval_instruct"]


def test_each_cell_has_a_two_gpu_24_hour_slurm_entrypoint():
    for launcher, (task, method) in LAUNCHERS.items():
        contents = launcher.read_text()
        assert "#SBATCH --time=24:00:00" in contents
        assert "#SBATCH --gres=gpu:2" in contents
        assert "#SBATCH --ntasks-per-node=1" in contents
        assert f"export P8_TASK={task}" in contents
        assert f"export P8_METHOD={method}" in contents


def test_shared_launcher_uses_two_accelerate_ranks_with_wandb():
    contents = SHARED_LAUNCHER.read_text()

    assert "P8_WANDB_MODE:-online" in contents
    assert "num_gpu=2" in contents
    assert 'accelerate launch \\' in contents
    assert '--num_processes "${num_gpu}"' in contents
    assert "--wandb_args" in contents
    assert "--wandb_config_args" in contents


def test_humaneval_uses_short_tmpdir_for_multiprocessing_sockets():
    contents = SHARED_LAUNCHER.read_text()

    assert 'if [ "${task}" = humaneval_instruct ]; then' in contents
    assert 'export TMPDIR=/tmp/p8-${SLURM_JOB_ID:-manual}' in contents


def test_only_main_accelerate_rank_keeps_wandb_arguments():
    arguments = [
        "eval.py",
        "--tasks",
        "gsm8k_cot",
        "--wandb_args",
        "project=test",
        "--wandb_config_args",
        "phase=8",
    ]

    assert EVAL_MODULE._strip_wandb_arguments_for_non_main_rank(
        arguments,
        local_rank=0,
    ) == arguments
    assert EVAL_MODULE._strip_wandb_arguments_for_non_main_rank(
        arguments,
        local_rank=2,
    ) == ["eval.py", "--tasks", "gsm8k_cot"]


def test_distributed_auxiliary_outputs_are_sharded_without_overwrite(
    tmp_path,
    monkeypatch,
):
    class FakeAccelerator:
        def wait_for_everyone(self):
            return None

    monkeypatch.setattr(EVAL_MODULE.torch.cuda, "is_available", lambda: False)
    output_path = tmp_path / "results.json"
    harness_class = EVAL_MODULE.LLaDAPathSelectionEvalHarness

    for rank in (1, 0):
        harness = object.__new__(harness_class)
        harness._rank = rank
        harness._world_size = 2
        harness.accelerator = FakeAccelerator()
        harness.sampler_type = "entropy_drop"
        harness.selected_candidates_per_example = [{"example_index": 0}]
        harness.diagnostics_per_example = [{"example_index": 0, "steps": []}]
        harness.generation_batch_seconds = [float(rank + 1)]
        harness.save_selected_candidates(str(output_path))

    completion = json.loads(
        (tmp_path / "results.json_entropy_drop_runtime.json").read_text()
    )
    diagnostics_manifest = json.loads(
        (
            tmp_path / "results.json_entropy_drop_diagnostics_manifest.json"
        ).read_text()
    )
    assert completion["distributed"] is True
    assert completion["world_size"] == 2
    assert completion["generation_total_seconds"] == 2.0
    assert completion["generation_work_seconds"] == 3.0
    assert len(diagnostics_manifest["shards"]) == 2
    assert all(Path(path).is_file() for path in diagnostics_manifest["shards"])
