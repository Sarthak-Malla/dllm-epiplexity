"""Score dependency candidates without an additional model forward.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_non_lookahead.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dllm.core.samplers.candidates import CandidateBatch


NON_LOOKAHEAD_SELECTORS = (
    "max_confidence",
    "min_entropy",
    "min_top2_margin",
)


@dataclass(frozen=True)
class NonLookaheadSelectionOutput:
    """Per-candidate token statistics and the selected action."""

    candidates: CandidateBatch
    selector: str
    candidate_means: torch.Tensor
    selection_scores: torch.Tensor
    best_index: torch.Tensor
    best_score: torch.Tensor
    best_mask: torch.Tensor
    best_names: tuple[str | None, ...]


def top2_probability_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """Return ``p(top1) - p(top2)`` for each position."""
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 2:
        raise ValueError("probabilities must have shape [..., vocabulary].")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point.")
    if probabilities.shape[-1] < 2:
        raise ValueError("top-two margin requires at least two vocabulary entries.")
    if not torch.isfinite(probabilities).all() or torch.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1].")
    top_two = torch.topk(probabilities.float(), k=2, dim=-1).values
    return top_two[..., 0] - top_two[..., 1]


def _validate_position_map(
    values: torch.Tensor,
    *,
    name: str,
    candidates: CandidateBatch,
) -> torch.Tensor:
    """Validate one finite, nonnegative [B,T] token statistic."""
    expected = tuple(candidates.eligible_mask.shape)
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}.")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if values.device != candidates.candidate_masks.device:
        raise ValueError(f"{name} must share the candidate device.")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return values.float()


def select_candidates_without_lookahead(
    candidates: CandidateBatch,
    *,
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    top2_margin: torch.Tensor | None,
    selector: str,
) -> NonLookaheadSelectionOutput:
    """Choose one candidate from base-pass token statistics only.

    All selectors first average their statistic over the positions in a
    candidate. Confidence is maximized; entropy and top-two margin are minimized.
    Equal scores retain the earliest dependency candidate.
    """
    if not isinstance(candidates, CandidateBatch):
        raise TypeError("candidates must be a CandidateBatch.")
    if selector not in NON_LOOKAHEAD_SELECTORS:
        raise ValueError(
            f"selector must be one of {NON_LOOKAHEAD_SELECTORS}, got {selector!r}."
        )
    confidence = _validate_position_map(
        confidence,
        name="confidence",
        candidates=candidates,
    )
    entropy = _validate_position_map(
        entropy,
        name="entropy",
        candidates=candidates,
    )
    if selector == "max_confidence":
        token_values = confidence
        maximize = True
    elif selector == "min_entropy":
        token_values = entropy
        maximize = False
    else:
        if top2_margin is None:
            raise ValueError("min_top2_margin requires top2_margin.")
        token_values = _validate_position_map(
            top2_margin,
            name="top2_margin",
            candidates=candidates,
        )
        maximize = False

    masks = candidates.candidate_masks
    action_sizes = masks.sum(dim=-1, dtype=torch.long)
    candidate_sums = torch.where(
        masks,
        token_values.unsqueeze(0),
        torch.zeros_like(masks, dtype=torch.float32),
    ).sum(dim=-1)
    candidate_means = candidate_sums / action_sizes.clamp_min(1).float()
    selection_scores = candidate_means if maximize else -candidate_means
    selection_scores = torch.where(
        candidates.candidate_valid,
        selection_scores,
        torch.full_like(selection_scores, -torch.inf),
    )

    batch_size, sequence_length = candidates.eligible_mask.shape
    best_score = torch.full(
        (batch_size,),
        -torch.inf,
        device=selection_scores.device,
        dtype=torch.float32,
    )
    best_index = torch.full(
        (batch_size,),
        -1,
        device=selection_scores.device,
        dtype=torch.long,
    )
    for candidate_index in range(selection_scores.shape[0]):
        improved = (
            candidates.candidate_valid[candidate_index]
            & (selection_scores[candidate_index] > best_score)
        )
        best_score = torch.where(
            improved,
            selection_scores[candidate_index],
            best_score,
        )
        best_index = torch.where(
            improved,
            torch.full_like(best_index, candidate_index),
            best_index,
        )

    best_mask = torch.zeros(
        (batch_size, sequence_length),
        device=masks.device,
        dtype=torch.bool,
    )
    best_names: list[str | None] = []
    for batch_index, candidate_index in enumerate(best_index.tolist()):
        if candidate_index < 0:
            best_names.append(None)
            continue
        best_mask[batch_index] = masks[candidate_index, batch_index]
        best_names.append(candidates.names[candidate_index])

    return NonLookaheadSelectionOutput(
        candidates=candidates,
        selector=selector,
        candidate_means=candidate_means,
        selection_scores=selection_scores,
        best_index=best_index,
        best_score=best_score,
        best_mask=best_mask,
        best_names=tuple(best_names),
    )
