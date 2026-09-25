"""Probability conversion and fusion utilities for TSE."""

import torch


def logits_to_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert logits to float32 probabilities with temperature scaling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("temperature scaling produced non-finite probabilities")
    return probabilities


def fuse_probabilities(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Blend two aligned probability tensors without changing their shape."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError(
            "probability tensors must have identical shapes: "
            f"{tuple(probabilities_a.shape)} != {tuple(probabilities_b.shape)}"
        )
    if not torch.isfinite(probabilities_a).all() or not torch.isfinite(
        probabilities_b
    ).all():
        raise ValueError("probabilities must contain only finite values")

    fused = alpha * probabilities_a.float() + (1 - alpha) * probabilities_b.float()
    if not torch.isfinite(fused).all():
        raise ValueError("probability fusion produced non-finite values")
    return fused