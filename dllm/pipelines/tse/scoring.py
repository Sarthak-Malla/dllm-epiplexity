"""Confidence and consensus scoring for TSE position selection."""

import torch


def fused_confidence(
    fused_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return maximum probability and token ID for each active position."""
    if fused_probabilities.ndim < 1:
        raise ValueError("fused_probabilities must have a vocabulary dimension")
    if (fused_probabilities < 0).any() or not torch.isfinite(fused_probabilities).all():
        raise ValueError("fused probabilities must be finite and non-negative")
    return torch.max(fused_probabilities.float(), dim=-1)


def consensus_scores(
    confidence: torch.Tensor,
    agreement: torch.Tensor,
) -> torch.Tensor:
    """Combine fused confidence and cross-model agreement per position."""
    if confidence.shape != agreement.shape:
        raise ValueError(
            "confidence and agreement must have identical shapes: "
            f"{tuple(confidence.shape)} != {tuple(agreement.shape)}"
        )
    if not torch.isfinite(confidence).all() or not torch.isfinite(agreement).all():
        raise ValueError("confidence and agreement must be finite")
    return confidence.float() * agreement.float()