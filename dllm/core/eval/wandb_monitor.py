"""Log optional path-selection progress to an active Weights & Biases run.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_wandb_monitor.py -v
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import logging
import math
from typing import Any


LOGGER = logging.getLogger(__name__)


def _finite_mean(values: Iterable[object]) -> float | None:
    """Return the mean of finite numeric values, or None for an empty input."""
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def summarize_diagnostic_batch(
    diagnostics_by_example: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, int | float]:
    """Aggregate path-selection diagnostics for one generated request batch."""
    steps = [step for example in diagnostics_by_example for step in example]
    if not steps:
        return {}
    selected = [
        step["selected_candidate"]
        for step in steps
        if isinstance(step.get("selected_candidate"), Mapping)
    ]
    summary: dict[str, int | float] = {
        "phase6/steps_in_batch": len(steps),
        "phase6/mean_steps_per_example": len(steps)
        / max(len(diagnostics_by_example), 1),
        "phase6/hard_fallback_selection_count": sum(
            bool(candidate.get("hard_fallback_count"))
            for candidate in selected
        ),
    }
    optional_means = {
        "phase6/selected_set_mean_conflict": _finite_mean(
            step.get("selected_set_mean_conflict") for step in steps
        ),
        "phase6/selected_set_max_conflict": _finite_mean(
            step.get("selected_set_max_conflict") for step in steps
        ),
        "phase6/next_state_metric_mean": _finite_mean(
            step.get("current_state_metric_mean")
            for step in steps
            if int(step.get("step_index", 0)) > 0
        ),
        "phase6/selected_anchor_support_mean": _finite_mean(
            candidate.get("anchor_support_sum") for candidate in selected
        ),
    }
    summary.update(
        {
            name: value
            for name, value in optional_means.items()
            if value is not None
        }
    )
    consistency_count = sum(
        int(step.get("immediate_token_consistency_count", 0)) for step in steps
    )
    consistency_total = sum(
        int(step.get("immediate_token_consistency_total", 0)) for step in steps
    )
    if consistency_total:
        summary["phase6/immediate_token_consistency_rate"] = (
            consistency_count / consistency_total
        )
        summary["phase6/immediate_token_consistency_total"] = consistency_total
    action_sizes = [
        int(step.get("commit_k", 0))
        for step in steps
        if int(step.get("commit_k", 0)) > 0
    ]
    if action_sizes:
        summary["phase7/mean_action_size"] = sum(action_sizes) / len(action_sizes)
        summary["phase7/max_action_size"] = max(action_sizes)
        for action_size in (1, 2, 4):
            summary[f"phase7/action_size_{action_size}_count"] = action_sizes.count(
                action_size
            )
    phase7_optional_means = {
        "phase7/selected_raw_verifier_score": _finite_mean(
            candidate.get("raw_verifier_score") for candidate in selected
        ),
        "phase7/selected_size_aware_verifier_score": _finite_mean(
            candidate.get("size_aware_verifier_score") for candidate in selected
        ),
        "phase7/selected_immediate_action_cost": _finite_mean(
            candidate.get("immediate_action_cost") for candidate in selected
        ),
    }
    summary.update(
        {
            name: value
            for name, value in phase7_optional_means.items()
            if value is not None
        }
    )
    return summary


def log_generation_progress(
    *,
    examples_completed: int,
    examples_total: int,
    batch_seconds: float,
    cumulative_seconds: float,
    diagnostics_by_example: Sequence[Sequence[Mapping[str, Any]]],
    cuda_memory: Mapping[str, int] | None = None,
) -> bool:
    """Log progress to lm-eval's active W&B run without affecting evaluation."""
    try:
        import wandb

        run = wandb.run
        if run is None:
            return False
        payload: dict[str, int | float] = {
            "generation/examples_completed": examples_completed,
            "generation/examples_total": examples_total,
            "generation/progress_fraction": (
                examples_completed / examples_total if examples_total else 0.0
            ),
            "generation/batch_seconds": float(batch_seconds),
            "generation/cumulative_seconds": float(cumulative_seconds),
        }
        payload.update(summarize_diagnostic_batch(diagnostics_by_example))
        if cuda_memory is not None:
            payload.update(
                {
                    f"cuda/{name}": int(value)
                    for name, value in cuda_memory.items()
                }
            )
        run.log(payload)
        return True
    except Exception as error:  # Monitoring must never invalidate an evaluation.
        LOGGER.warning("W&B progress logging failed: %s", error)
        return False
