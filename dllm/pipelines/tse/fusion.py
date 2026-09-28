"""Probability conversion and fusion utilities for TSE.

Run the focused tests with:
    pytest scripts/tests/test_tse_fusion.py -v
"""

from collections.abc import Sequence

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
    weight_a = torch.as_tensor(
        alpha, dtype=torch.float32, device=probabilities_a.device
    )
    if weight_a.ndim == 1 and probabilities_a.ndim == 2:
        weight_a = weight_a.unsqueeze(-1)
    if weight_a.ndim not in {0, probabilities_a.ndim}:
        raise ValueError("alpha must be scalar or broadcastable over probabilities")
    if (weight_a < 0).any() or (weight_a > 1).any():
        raise ValueError("alpha must be between 0 and 1")

    if weight_a.ndim == 0:
        weights = torch.stack([weight_a, 1 - weight_a])
    else:
        weights = torch.cat([weight_a, 1 - weight_a], dim=-1)
    return fuse_distributions([probabilities_a, probabilities_b], weights)


def fuse_distributions(
    probabilities: Sequence[torch.Tensor],
    weights: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Blend aligned model distributions using global or position-local weights."""
    if not probabilities:
        raise ValueError("at least one probability tensor is required")
    reference_shape = probabilities[0].shape
    if any(probability.shape != reference_shape for probability in probabilities[1:]):
        raise ValueError("probability tensors must have identical shapes")
    if any(
        not torch.isfinite(probability).all() or (probability < 0).any()
        for probability in probabilities
    ):
        raise ValueError("probabilities must be finite and non-negative")

    device = probabilities[0].device
    model_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    model_count = len(probabilities)
    if model_weights.ndim == 1:
        if model_weights.shape[0] != model_count:
            raise ValueError("global weights must have one entry per model")
        broadcast_weights = model_weights.view(
            *((1,) * (probabilities[0].ndim - 1)), model_count, 1
        )
    elif model_weights.ndim == 2 and probabilities[0].ndim == 2:
        if model_weights.shape != (reference_shape[0], model_count):
            raise ValueError("local weights must have shape [N, num_models]")
        broadcast_weights = model_weights.unsqueeze(-1)
    else:
        raise ValueError("weights must be global [M] or local [N, M]")
    if (model_weights < 0).any() or not torch.isfinite(model_weights).all():
        raise ValueError("weights must be finite and non-negative")
    if not torch.allclose(
        model_weights.sum(dim=-1),
        torch.ones_like(model_weights.sum(dim=-1)),
        atol=1e-6,
    ):
        raise ValueError("weights must sum to one")

    stacked = torch.stack(
        [probability.to(device).float() for probability in probabilities],
        dim=-2,
    )
    fused = (stacked * broadcast_weights).sum(dim=-2)
    if not torch.isfinite(fused).all():
        raise ValueError("probability fusion produced non-finite values")
    return fused
