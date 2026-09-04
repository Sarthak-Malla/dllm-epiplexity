"""Validate the reduced Phase-8 full-evaluation launch configuration.

Run from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_reduced_full_launchers.py -v
"""

import json
import importlib.util
from pathlib import Path


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


def test_reduced_plan_contains_only_the_four_new_full_cells():
    plan = json.loads(PLAN.read_text())

    assert set(plan["tasks"]) == {"gsm8k_cot", "humaneval_instruct"}
    assert {method["name"] for method in plan["methods"]} == {
        "dependency_fixed_k4_n4",
        "dependency_entropy_budget_n4",
    }
    assert plan["execution"]["job_count"] == 4
    assert plan["execution"]["gpus_per_job"] == 2
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
