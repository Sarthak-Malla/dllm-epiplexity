"""Validate configuration-only path-selection ablation launchers.

Run on a compute node with:
    source /home/sarthak.malla/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_ablation_launchers.py -v
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
    assert "candidate_chunk_size=${PATH_ABLATION_CANDIDATE_CHUNK_SIZE:-1}" in contents
    assert "candidate_chunk_size=${candidate_chunk_size}" in contents


def test_ie_maxconf_diagnostics_launcher_keeps_algorithm_and_isolates_outputs():
    launcher = SHARED_RUNNER.parent / "run_gsm8k_ie_maxconf_n4_diagnostics.slurm.sh"
    result = _dry_run(launcher)
    assert result.returncode == 0, result.stderr
    for argument in (
        "sampler_type=max_confidence", "candidate_budget=4,",
        "dependency_seed_strategy=incoming", "dependency_seed_entropy_weight=1.0",
        "dependency_confidence_exponent=0.0", "dependency_entropy_budget=2.0",
        "dependency_max_action_size=64", "temperature=0.0", "block_size=64",
        "diagnostic_metadata=true", "diagnostic_retention=full",
    ):
        assert argument in result.stdout
    assert "max_confidence_ie_diagnostics_v1/gsm8k_cot/max_confidence/diagnostics/" in result.stdout
    assert "Response caching disabled" in result.stdout
    assert "Evaluation limit: full" in result.stdout
    limited = _dry_run(launcher, PATH_ABLATION_LIMIT="8")
    assert limited.returncode == 0, limited.stderr
    assert "/limit8/" in limited.stdout


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


@pytest.mark.parametrize("launcher,count", [(ABLATION2_LAUNCHERS[1], 4), (ABLATION3_LAUNCHER, 8)])
@pytest.mark.parametrize("strategy,weight", [("incoming", "0.0"), ("incoming", "1.0"), ("confidence", "0.0")])
def test_corrected_seed_parallel_runs_isolate_caches_and_preserve_other_model_settings(
    launcher, count, strategy, weight
):
    overrides = {"PATH_ABLATION_CANDIDATE_CHUNK_SIZE": str(count)}
    legacy = _dry_run(launcher, **overrides)
    corrected = _dry_run(
        launcher,
        **overrides,
        PATH_ABLATION_SEED_STRATEGY=strategy,
        PATH_ABLATION_SEED_ENTROPY_WEIGHT=weight,
    )
    assert legacy.returncode == 0, legacy.stderr
    assert corrected.returncode == 0, corrected.stderr

    def model_arguments(output):
        line = next(line for line in output.splitlines() if line.startswith("Model arguments: "))
        return dict(argument.split("=", 1) for argument in line.removeprefix("Model arguments: ").split(","))

    expected = model_arguments(legacy.stdout)
    assert expected["dependency_seed_strategy"] == "legacy"
    assert expected["dependency_seed_entropy_weight"] == "0.0"
    expected.update(dependency_seed_strategy=strategy, dependency_seed_entropy_weight=weight)
    assert model_arguments(corrected.stdout) == expected
    assert expected["candidate_budget"] == expected["candidate_chunk_size"] == str(count)
    assert expected["dependency_entropy_budget"] == "2.0"
    assert expected["dependency_max_action_size"] == "64"
    assert expected["dependency_confidence_exponent"] == "0.0"

    path_suffix = (
        f"gsm8k_cot/seed_{strategy}_entropy{weight}/entropy_budget2.0"
        f"/candidates{count}/chunk{count}/seed42"
    )
    assert f"{path_suffix}/results.json" in corrected.stdout
    assert f"{path_suffix}/responses.cache" in corrected.stdout
    assert path_suffix not in legacy.stdout
    assert f"-seed-{strategy}-entropy{weight}-entropy-budget2.0-n{count}-chunk{count}-s42" in corrected.stdout
    wandb_config = next(line for line in corrected.stdout.splitlines() if line.startswith("W&B configuration: "))
    assert f"dependency_seed_strategy={strategy},dependency_seed_entropy_weight={weight}" in wandb_config


@pytest.mark.parametrize("strategy,weight", [
    ("unknown", "0.0"), ("legacy", "1.0"), ("confidence", "1.0"),
    ("incoming", "-1"), ("incoming", "nan"), ("incoming", "1.0,other=1"),
])
def test_corrected_seed_launcher_rejects_invalid_settings(strategy, weight):
    result = _dry_run(
        ABLATION2_LAUNCHERS[1],
        PATH_ABLATION_SEED_STRATEGY=strategy,
        PATH_ABLATION_SEED_ENTROPY_WEIGHT=weight,
    )
    assert result.returncode == 2
    assert "PATH_ABLATION_SEED_" in result.stderr


def test_corrected_sequential_candidate_counts_use_separate_response_caches():
    for count in (4, 8):
        result = _dry_run(
            ABLATION2_LAUNCHERS[1],
            PATH_ABLATION_CANDIDATE_BUDGET=str(count),
            PATH_ABLATION_CANDIDATE_CHUNK_SIZE="1",
            PATH_ABLATION_SEED_STRATEGY="incoming",
            PATH_ABLATION_SEED_ENTROPY_WEIGHT="1.0",
        )
        assert result.returncode == 0, result.stderr
        assert f"/candidates{count}/chunk1/seed42/responses.cache" in result.stdout
        assert f"-n{count}-chunk1-s42" in result.stdout


@pytest.mark.parametrize("launcher,count", [(ABLATION2_LAUNCHERS[1], 4), (ABLATION3_LAUNCHER, 8)])
def test_max_confidence_uses_no_lookahead_and_preserves_ie_entropy_budget(launcher, count):
    result = _dry_run(
        launcher,
        PATH_ABLATION_SAMPLER_TYPE="max_confidence",
        PATH_ABLATION_SEED_STRATEGY="incoming",
        PATH_ABLATION_SEED_ENTROPY_WEIGHT="1.0",
        # A previous parallel-run override must not create false batching labels.
        PATH_ABLATION_CANDIDATE_CHUNK_SIZE=str(count),
    )
    assert result.returncode == 0, result.stderr
    for argument in (
        "sampler_type=max_confidence",
        "dependency_candidate_selector=max_confidence",
        "diagnostic_metadata=false",
        "diagnostic_retention=none",
        f"candidate_budget={count},candidate_chunk_size=1",
        "dependency_seed_strategy=incoming,dependency_seed_entropy_weight=1.0",
        "dependency_cardinality_strategy=entropy_budget",
        "dependency_entropy_budget=2.0",
        "dependency_max_action_size=64",
        "dependency_direction=outgoing",
        "dependency_confidence_exponent=0.0",
    ):
        assert argument in result.stdout
    path_suffix = (
        "gsm8k_cot/max_confidence/seed_incoming_entropy1.0"
        f"/entropy_budget2.0/candidates{count}/seed42"
    )
    assert f"{path_suffix}/responses.cache" in result.stdout
    assert f"{path_suffix}/results.json_max_confidence_runtime.json" in result.stdout
    assert "selector=max_confidence,lookahead=false" in result.stdout
    assert "Candidate chunk size: not used (no lookahead)" in result.stdout
    assert "/chunk" not in result.stdout
    assert "-max-confidence-uncapped-seed-incoming-entropy1.0-entropy-budget2.0" in result.stdout
    assert f"-n{count}-s42" in result.stdout
    assert "_entropy_drop_runtime.json" not in result.stdout


@pytest.mark.parametrize("overrides", [
    {"PATH_ABLATION_SAMPLER_TYPE": "unknown"},
    {
        "PATH_ABLATION_SAMPLER_TYPE": "max_confidence",
        "PATH_ABLATION_CARDINALITY_STRATEGY": "marginal_utility",
    },
])
def test_shared_runner_rejects_unsupported_sampler_settings(overrides):
    result = _dry_run(ABLATION2_LAUNCHERS[1], **overrides)
    assert result.returncode == 2
    assert "PATH_ABLATION_SAMPLER_TYPE" in result.stderr


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
