"""Validate configuration-only path-selection ablation launchers.

Run from any directory with:
    source /home/sarthak.malla/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_ablation_launchers.py -v
"""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
SHARED_RUNNER = (
    ROOT
    / "examples/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "run_common.sh"
)
ABLATION2_LAUNCHERS = (
    ROOT
    / "examples/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "run_gsm8k_entropy_budget1_uncapped.slurm.sh",
    ROOT
    / "examples/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "run_gsm8k_entropy_budget2_uncapped.slurm.sh",
    ROOT
    / "examples/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "run_gsm8k_entropy_budget4_uncapped.slurm.sh",
)
ABLATION3_LAUNCHER = (
    ROOT
    / "examples/path_selection/ablation/3_candidate_budget"
    / "run_gsm8k_candidate_budget8.slurm.sh"
)
ABLATION4_LAUNCHER = (
    ROOT
    / "examples/path_selection/ablation/4_marginal_utility"
    / "run_gsm8k_marginal_utility.slurm.sh"
)


def _dry_run(launcher, **overrides):
    """Resolve real shell arguments without starting evaluation or writing files."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ABLATION", "PATH_ABLATION_"))
    }
    env.update(ABLATION2_DRY_RUN="1", **overrides)
    return subprocess.run(
        ["bash", str(launcher)], env=env, text=True, capture_output=True
    )


def test_shared_runner_keeps_four_candidates_as_the_default():
    contents = SHARED_RUNNER.read_text()

    assert "candidate_budget=${PATH_ABLATION_CANDIDATE_BUDGET:-4}" in contents
    assert "candidate_budget=${candidate_budget}" in contents
    assert "candidate_chunk_size=1" in contents


def test_existing_ablation2_launchers_do_not_override_candidate_budget():
    for launcher in ABLATION2_LAUNCHERS:
        contents = launcher.read_text()

        assert "PATH_ABLATION_CANDIDATE_BUDGET" not in contents
        assert str(SHARED_RUNNER) in contents


def test_ablation3_changes_only_the_matched_configuration_dimensions():
    contents = ABLATION3_LAUNCHER.read_text()

    assert "export PATH_ABLATION_NUMBER=3" in contents
    assert "export PATH_ABLATION_OUTPUT_SUBDIRECTORY=3_candidate_budget" in contents
    assert "export PATH_ABLATION_CANDIDATE_BUDGET=8" in contents
    assert "export ABLATION2_ENTROPY_BUDGET=2.0" in contents
    assert "export ABLATION2_MAXIMUM_ACTION_SIZE=64" in contents
    assert str(SHARED_RUNNER) in contents


def test_ablation3_retains_the_entropy_drop_dependency_algorithm():
    result = _dry_run(ABLATION3_LAUNCHER)
    assert result.returncode == 0, result.stderr
    contents = result.stdout

    required_model_arguments = (
        "sampler_type=entropy_drop",
        "proposal_strategy=dependency",
        "dependency_cardinality_strategy=entropy_budget",
        "dependency_size_scoring=per_token",
        "dependency_parallel_variant=soft_full",
        "candidate_chunk_size=1",
    )
    for argument in required_model_arguments:
        assert argument in contents


@pytest.mark.parametrize(
    "launcher,budget", zip(ABLATION2_LAUNCHERS, ("1.0", "2.0", "4.0"))
)
@pytest.mark.parametrize("maximum_size", ("8", "64"))
def test_entropy_budget_arguments_and_output_paths_are_preserved(
    launcher, budget, maximum_size
):
    result = _dry_run(launcher, ABLATION2_MAXIMUM_ACTION_SIZE=maximum_size)
    assert result.returncode == 0, result.stderr
    contents = result.stdout
    assert "dependency_cardinality_strategy=entropy_budget" in contents
    assert f"dependency_entropy_budget={budget}" in contents
    assert "candidate_budget=4," in contents
    assert "dependency_utility_threshold=" not in contents
    cap = "" if maximum_size == "64" else "max8/"
    assert f"gsm8k_cot/{cap}entropy_budget{budget}/seed42/results.json" in contents
    assert f"entropy-budget{budget}-s42" in contents


@pytest.mark.parametrize("threshold", ("0", "0.0", "0.25", "0.5"))
def test_marginal_utility_threshold_reaches_model_paths_and_wandb(threshold):
    result = _dry_run(ABLATION4_LAUNCHER, ABLATION4_UTILITY_THRESHOLD=threshold)
    assert result.returncode == 0, result.stderr
    contents = result.stdout
    tau = "0.0" if threshold == "0" else threshold
    for argument in (
        "dependency_cardinality_strategy=marginal_utility",
        f"dependency_utility_threshold={tau}",
        "dependency_max_action_size=64",
        "candidate_budget=4,",
        "dependency_parallel_variant=soft_full",
        "sampler_type=entropy_drop",
        "dependency_size_scoring=per_token",
        "dependency_generation_seed=42",
    ):
        assert argument in contents
    assert "dependency_entropy_budget=" not in contents
    directory = (
        ROOT
        / "eval_results/path_selection/ablation/4_marginal_utility"
        / "marginal_utility_v1/gsm8k_cot"
        / f"marginal_utility_tau{tau}/seed42"
    )
    assert f"Output: {directory}" in contents
    assert f"Response cache: {directory}/responses.cache" in contents
    assert (
        f"W&B name: ablation-4-n4-gsm8k_cot-uncapped-marginal-utility-tau{tau}-s42"
        in contents
    )
    assert f"cardinality_strategy=marginal_utility,utility_threshold={tau}" in contents
    assert "gpu_count=2" in contents


@pytest.mark.parametrize("threshold", ("-1", "nan", "0.3", "0.25,other=1"))
def test_marginal_utility_rejects_unsupported_thresholds(threshold):
    result = _dry_run(ABLATION4_LAUNCHER, ABLATION4_UTILITY_THRESHOLD=threshold)
    assert result.returncode == 2
    assert "UTILITY_THRESHOLD must be" in result.stderr


def test_marginal_launcher_matches_reference_slurm_resources():
    def resources(path):
        return [
            line for line in path.read_text().splitlines()
            if line.startswith("#SBATCH") and "--job-name" not in line
        ]

    assert resources(ABLATION4_LAUNCHER) == resources(ABLATION2_LAUNCHERS[1])
