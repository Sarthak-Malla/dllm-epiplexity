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
    alpha: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Blend two aligned probability tensors with scalar or local weights."""
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError(
            "probability tensors must have identical shapes: "
            f"{tuple(probabilities_a.shape)} != {tuple(probabilities_b.shape)}"
        )
    if not torch.isfinite(probabilities_a).all() or not torch.isfinite(
        probabilities_b
    ).all():
        raise ValueError("probabilities must contain only finite values")

    weight_a = torch.as_tensor(
        alpha, dtype=torch.float32, device=probabilities_a.device
    )
    if weight_a.ndim == 1 and probabilities_a.ndim == 2:
        weight_a = weight_a.unsqueeze(-1)
    if weight_a.ndim not in {0, probabilities_a.ndim}:
        raise ValueError("alpha must be scalar or broadcastable over probabilities")
    if (weight_a < 0).any() or (weight_a > 1).any():
        raise ValueError("alpha must be between 0 and 1")

    fused = weight_a * probabilities_a.float() + (1 - weight_a) * probabilities_b.float()
    if not torch.isfinite(fused).all():
        raise ValueError("probability fusion produced non-finite values")
    return fused