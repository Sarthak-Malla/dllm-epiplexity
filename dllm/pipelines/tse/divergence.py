"""Divergence metrics for TSE probability distributions."""

import math

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
    alpha: float = 0.5,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute weighted two-distribution Jensen-Shannon divergence."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    _validate_probabilities(probabilities_a, probabilities_b)
    mixture = alpha * probabilities_a.float() + (1 - alpha) * probabilities_b.float()
    divergence = entropy(mixture, epsilon) - (
        alpha * entropy(probabilities_a, epsilon)
        + (1 - alpha) * entropy(probabilities_b, epsilon)
    )
    return divergence.clamp_min(0.0)


def agreement_factor(
    divergence: torch.Tensor,
    alpha: float = 0.5,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Convert divergence into a bounded agreement score."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (divergence < 0).any() or not torch.isfinite(divergence).all():
        raise ValueError("divergence must be finite and non-negative")

    weight_entropy = 0.0
    if alpha > 0:
        weight_entropy -= alpha * math.log(alpha)
    if alpha < 1:
        weight_entropy -= (1 - alpha) * math.log(1 - alpha)
    normalization = max(weight_entropy, epsilon)
    return (1 - divergence.float() / normalization).clamp(0.0, 1.0)