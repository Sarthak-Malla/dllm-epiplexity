"""Divergence metrics for TSE probability distributions."""

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
    _validate_probabilities(probabilities_a, probabilities_b)
    weight_a = torch.as_tensor(alpha, dtype=torch.float32, device=probabilities_a.device)
    if weight_a.ndim == 1 and probabilities_a.ndim == 2:
        weight_a = weight_a.unsqueeze(-1)
    if weight_a.ndim not in {0, probabilities_a.ndim}:
        raise ValueError("alpha must be scalar or broadcastable over probabilities")
    if (weight_a < 0).any() or (weight_a > 1).any():
        raise ValueError("alpha must be between 0 and 1")

    mixture = weight_a * probabilities_a.float() + (1 - weight_a) * probabilities_b.float()
    weight_vector = weight_a.squeeze(-1) if weight_a.ndim == 2 else weight_a
    divergence = entropy(mixture, epsilon) - (
        weight_vector * entropy(probabilities_a, epsilon)
        + (1 - weight_vector) * entropy(probabilities_b, epsilon)
    )
    return divergence.clamp_min(0.0)


def agreement_factor(
    divergence: torch.Tensor,
    alpha: float | torch.Tensor = 0.5,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Convert divergence into a bounded agreement score."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (divergence < 0).any() or not torch.isfinite(divergence).all():
        raise ValueError("divergence must be finite and non-negative")

    weight_a = torch.as_tensor(alpha, dtype=torch.float32, device=divergence.device)
    if weight_a.ndim == 2 and weight_a.shape[-1] == 1:
        weight_a = weight_a.squeeze(-1)
    weight_b = 1 - weight_a
    if (weight_a < 0).any() or (weight_a > 1).any():
        raise ValueError("alpha must be between 0 and 1")
    weight_entropy = -(
        weight_a * weight_a.clamp_min(epsilon).log()
        + weight_b * weight_b.clamp_min(epsilon).log()
    )
    normalization = weight_entropy.clamp_min(epsilon)
    return (1 - divergence.float() / normalization).clamp(0.0, 1.0)