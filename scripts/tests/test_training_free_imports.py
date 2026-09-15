"""Check experiment entrypoint imports against a competing installed examples package.

Source /home/sarthak.malla/.zshrc and activate the dllm environment, then run on compute:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:20:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_imports.py -v
Only fresh-process --help and --dry-run paths are exercised; no model, GPU, or
experiment worker is started. The tests themselves must use the user's compute workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
ENTRYPOINT_DIRECTORY = ROOT / "examples/path_selection/experiments"


@pytest.mark.parametrize("repository_first", (False, True), ids=("competitor-first", "repository-first"))
@pytest.mark.parametrize("entrypoint", ("launch.py", "runner.py", "benchmark.py"),
                         ids=("launcher", "worker", "benchmark"))
@pytest.mark.parametrize("mode", ("help", "dry-run"))
def test_fresh_entrypoint_resolves_local_examples_despite_installed_collision(
    tmp_path, repository_first, entrypoint, mode,
):
    """Both package identity and path priority are required in a fresh interpreter."""
    competing_root = tmp_path / "third_party"
    competing_package = competing_root / "examples"
    competing_package.mkdir(parents=True)
    (competing_package / "__init__.py").write_text(
        '"""A regular third-party package with no path_selection subpackage."""\n'
        'PACKAGE_ORIGIN = "competing-third-party"\n'
    )
    path_entries = [str(ROOT), str(competing_root)] if repository_first else [str(competing_root), str(ROOT)]
    path_entries.append(str(ROOT / "lm-evaluation-harness"))
    environment = os.environ.copy()
    environment.update({
        # The repository is deliberately already present. An `if not in
        # sys.path` bootstrap does not move it ahead of a competing package.
        "PYTHONPATH": os.pathsep.join(path_entries),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
    })
    # A broken dry-run must fail the workflow guard, never start GPU inference
    # merely because pytest itself is running in the required Slurm allocation.
    environment.pop("SLURM_JOB_ID", None)
    output_root = tmp_path / "must_not_be_created"
    arguments = [sys.executable, str(ENTRYPOINT_DIRECTORY / entrypoint),
                 "--output-root", str(output_root)]
    if entrypoint == "benchmark.py":
        config = tmp_path / "task.json"
        config.write_text('{"task": "synthetic", "primary_metric": "score", "primary_filter": "native"}')
        arguments.extend(("--config", str(config), "--arm", "first_action_entropy"))
    if mode == "help":
        arguments.append("--help")
    elif entrypoint == "launch.py":
        arguments.extend(("--dry-run", "--", "--resume", "collect"))
    elif entrypoint == "benchmark.py":
        arguments.append("--dry-run")
    else:
        arguments.extend(("--dry-run", "collect"))

    result = subprocess.run(
        arguments, cwd=tmp_path, env=environment, capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert result.returncode == 0, (
        f"{entrypoint} {mode} failed with an existing repository PYTHONPATH entry.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not output_root.exists(), "Import preflight unexpectedly wrote experiment artifacts."
    assert "ModuleNotFoundError" not in result.stderr
    if mode == "help":
        assert "usage:" in result.stdout.lower()
    elif entrypoint == "launch.py":
        assert '"worker_count": 2' in result.stdout
        assert '"commands"' in result.stdout
    elif entrypoint == "benchmark.py":
        assert '"selector_schedule"' in result.stdout
        assert '"document_work_range"' in result.stdout
    else:
        assert '"requested_stage": "collect"' in result.stdout
        assert '"document_work_range"' in result.stdout
