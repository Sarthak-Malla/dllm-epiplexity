"""Reduce saved path-selection diagnostics without changing online telemetry.

Run its focused tests from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_evaluation.py -v
"""

from __future__ import annotations


DIAGNOSTIC_RETENTION_MODES = ("full", "compact", "none")


def retain_diagnostics(diagnostics, mode: str):
    """Retain full, compact, or no per-step diagnostic metadata."""
    if mode not in DIAGNOSTIC_RETENTION_MODES:
        raise ValueError(
            f"diagnostic_retention must be one of {DIAGNOSTIC_RETENTION_MODES}, "
            f"got {mode!r}."
        )
    if mode == "full":
        return diagnostics
    if mode == "none":
        return [[] for _ in diagnostics]

    compact_examples = []
    for example_steps in diagnostics:
        compact_steps = []
        for step in example_steps:
            compact_step = {
                name: value
                for name, value in step.items()
                if name != "candidates"
            }
            selected = compact_step.get("selected_candidate")
            if isinstance(selected, dict):
                compact_step["selected_candidate"] = {
                    name: value
                    for name, value in selected.items()
                    if name != "positions"
                }
            compact_steps.append(compact_step)
        compact_examples.append(compact_steps)
    return compact_examples
