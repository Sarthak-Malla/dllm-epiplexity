"""Divergence metrics for TSE probability distributions.

Run the focused tests with:
    pytest scripts/tests/test_tse_components.py -v
"""

from collections.abc import Sequence

import torch


def _validate_probabilities(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor | None = None,
) -> None:
    if probabilities_b is not None and probabilities_a.shape != probabilities_b.shape:
        raise ValueError(
            "probability tensors must have identical shapes: "
            f"{tuple(probabilities_a.shape)} != {tuple(probabilities_b.shape)}"
        )
    if (probabilities_a < 0).any() or not torch.isfinite(probabilities_a).all():
        raise ValueError("probabilities must be finite and non-negative")
    if probabilities_b is not None and (
        (probabilities_b < 0).any() or not torch.isfinite(probabilities_b).all()
    ):
        raise ValueError("probabilities must be finite and non-negative")


def entropy(probabilities: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """Compute categorical entropy over the final vocabulary dimension."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _validate_probabilities(probabilities)
    probabilities = probabilities.float()
    return -(
        probabilities * probabilities.clamp_min(epsilon).log()
    ).sum(dim=-1)


def jensen_shannon_divergence(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    alpha: float | torch.Tensor = 0.5,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute weighted two-distribution Jensen-Shannon divergence."""
    weight_a = torch.as_tensor(alpha, dtype=torch.float32, device=probabilities_a.device)
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
    return generalized_jensen_shannon_divergence(
        [probabilities_a, probabilities_b], weights, epsilon
    )


def generalized_jensen_shannon_divergence(
    probabilities: Sequence[torch.Tensor],
    weights: torch.Tensor | Sequence[float],
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute weighted Jensen-Shannon divergence for aligned models."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if len(probabilities) < 2:
        raise ValueError("at least two probability tensors are required")
    reference = probabilities[0]
    for probability in probabilities:
        _validate_probabilities(reference, probability)

    model_weights = torch.as_tensor(
        weights, dtype=torch.float32, device=reference.device
    )
    model_count = len(probabilities)
    if model_weights.ndim == 1:
        if model_weights.shape[0] != model_count:
            raise ValueError("global weights must have one entry per model")
        entropy_weights = model_weights.view(model_count, 1)
        mixture_weights = model_weights.view(model_count, 1, 1)
    elif model_weights.ndim == 2 and reference.ndim == 2:
        if model_weights.shape != (reference.shape[0], model_count):
            raise ValueError("local weights must have shape [N, num_models]")
        entropy_weights = model_weights.transpose(0, 1)
        mixture_weights = entropy_weights.unsqueeze(-1)
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
        [probability.to(reference.device).float() for probability in probabilities]
    )
    mixture = (stacked * mixture_weights).sum(dim=0)
    component_entropies = torch.stack(
        [entropy(probability, epsilon) for probability in probabilities]
    )
    divergence = entropy(mixture, epsilon) - (
        component_entropies * entropy_weights
    ).sum(dim=0)
    return divergence.clamp_min(0.0)


def agreement_factor(
    divergence: torch.Tensor,
    alpha: float | torch.Tensor = 0.5,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Convert divergence into a bounded agreement score."""
    weight_a = torch.as_tensor(alpha, dtype=torch.float32, device=divergence.device)
    if weight_a.ndim == 2 and weight_a.shape[-1] == 1:
        weight_a = weight_a.squeeze(-1)
    if (weight_a < 0).any() or (weight_a > 1).any():
        raise ValueError("alpha must be between 0 and 1")
    weights = torch.stack([weight_a, 1 - weight_a], dim=-1)
    return generalized_agreement_factor(divergence, weights, epsilon)


def generalized_agreement_factor(
    divergence: torch.Tensor,
    weights: torch.Tensor | Sequence[float],
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Normalize M-way JSD by the entropy of its model weights."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (divergence < 0).any() or not torch.isfinite(divergence).all():
        raise ValueError("divergence must be finite and non-negative")
    model_weights = torch.as_tensor(
        weights, dtype=torch.float32, device=divergence.device
    )
    if model_weights.ndim == 1:
        if model_weights.numel() < 2:
            raise ValueError("weights must contain at least two models")
    elif model_weights.ndim == 2:
        if model_weights.shape[0] != divergence.shape[0]:
            raise ValueError("local weights must align with divergence positions")
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
    weight_entropy = -(
        model_weights * model_weights.clamp_min(epsilon).log()
    ).sum(dim=-1)
    normalization = weight_entropy.clamp_min(epsilon)
    return (1 - divergence.float() / normalization).clamp(0.0, 1.0)
