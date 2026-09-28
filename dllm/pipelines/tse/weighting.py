"""Adaptive model weighting strategies for TSE.

Run the focused tests with:
    pytest scripts/tests/test_tse_weighting.py -v
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dllm.pipelines.tse.divergence import entropy


@dataclass
class WeightingResult:
    """Model weights and optional diagnostics for active positions."""

    weights_a: torch.Tensor
    weights_b: torch.Tensor
    diagnostics: dict[str, torch.Tensor]

    @property
    def model_weights(self) -> torch.Tensor:
        """Return pair weights in the collection-oriented [N, M] form."""
        return torch.stack([self.weights_a, self.weights_b], dim=-1)


def _validate_probabilities(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
) -> None:
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError(
            "probability tensors must have identical shapes: "
            f"{tuple(probabilities_a.shape)} != {tuple(probabilities_b.shape)}"
        )
    if probabilities_a.ndim != 2:
        raise ValueError("active probabilities must have shape [N, vocab_size]")
    if (
        not torch.isfinite(probabilities_a).all()
        or not torch.isfinite(probabilities_b).all()
        or (probabilities_a < 0).any()
        or (probabilities_b < 0).any()
    ):
        raise ValueError("probabilities must be finite and non-negative")


def _validate_temperature(weight_temperature: float) -> None:
    if weight_temperature <= 0:
        raise ValueError("weight_temperature must be positive")


def _result_from_pairwise_weights(
    pairwise_weights: torch.Tensor,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> WeightingResult:
    if pairwise_weights.ndim != 2 or pairwise_weights.shape[1] != 2:
        raise ValueError("pairwise weights must have shape [N, 2]")
    if not torch.isfinite(pairwise_weights).all() or (pairwise_weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if not torch.allclose(
        pairwise_weights.sum(dim=-1),
        torch.ones(pairwise_weights.shape[0], device=pairwise_weights.device),
    ):
        raise ValueError("weights must sum to one for every active position")
    return WeightingResult(
        weights_a=pairwise_weights[:, 0],
        weights_b=pairwise_weights[:, 1],
        diagnostics=diagnostics or {},
    )


def _validate_probability_collection(
    probabilities: Sequence[torch.Tensor],
) -> None:
    if len(probabilities) < 2:
        raise ValueError("at least two model distributions are required")
    reference = probabilities[0]
    if reference.ndim != 2:
        raise ValueError("active probabilities must have shape [N, vocab_size]")
    for probability in probabilities:
        if probability.shape != reference.shape:
            raise ValueError("probability tensors must have identical shapes")
        if (
            not torch.isfinite(probability).all()
            or (probability < 0).any()
        ):
            raise ValueError("probabilities must be finite and non-negative")


def static_model_weights(
    weights: torch.Tensor | Sequence[float],
    num_positions: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Expand normalized global model weights over active positions."""
    if num_positions < 0:
        raise ValueError("num_positions must be non-negative")
    model_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if model_weights.ndim != 1 or model_weights.numel() < 2:
        raise ValueError("weights must be a vector with at least two entries")
    if (model_weights < 0).any() or not torch.isfinite(model_weights).all():
        raise ValueError("weights must be finite and non-negative")
    if not torch.isclose(model_weights.sum(), model_weights.new_tensor(1.0)):
        raise ValueError("weights must sum to one")
    return model_weights.expand(num_positions, -1)


def online_entropy_model_weights(
    probabilities: Sequence[torch.Tensor],
    weight_temperature: float = 1.0,
    epsilon: float = 1e-9,
    normalize_entropy: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute global M-way weights from each model's mean entropy."""
    _validate_probability_collection(probabilities)
    _validate_temperature(weight_temperature)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if probabilities[0].shape[0] == 0:
        return probabilities[0].new_empty((0, len(probabilities))), {}

    entropies = torch.stack(
        [entropy(probability, epsilon) for probability in probabilities]
    )
    if normalize_entropy:
        normalizer = torch.log(
            torch.tensor(
                probabilities[0].shape[-1],
                dtype=torch.float32,
                device=probabilities[0].device,
            )
        ).clamp_min(epsilon)
        entropies = entropies / normalizer
    mean_entropies = entropies.mean(dim=-1)
    weights = torch.softmax(-mean_entropies / weight_temperature, dim=0)
    return weights.expand(probabilities[0].shape[0], -1), {
        "mean_entropies": mean_entropies
    }


def per_token_margin_model_weights(
    probabilities: Sequence[torch.Tensor],
    weight_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute M-way local weights from top-one versus top-two margins."""
    _validate_probability_collection(probabilities)
    _validate_temperature(weight_temperature)
    if probabilities[0].shape[-1] < 2:
        raise ValueError("margin weighting requires at least two vocabulary entries")
    margins = []
    for probability in probabilities:
        top_two = torch.topk(probability.float(), k=2, dim=-1).values
        margins.append(top_two[:, 0] - top_two[:, 1])
    stacked_margins = torch.stack(margins, dim=-1)
    return torch.softmax(stacked_margins / weight_temperature, dim=-1), {
        "margins": stacked_margins
    }


def static_weights(
    alpha: float,
    num_positions: int,
    device: torch.device | None = None,
) -> WeightingResult:
    """Return fixed Model A/Model B weights for every active position."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    pairwise = static_model_weights([alpha, 1 - alpha], num_positions, device)
    return _result_from_pairwise_weights(pairwise)


def online_entropy_weights(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    weight_temperature: float = 1.0,
    epsilon: float = 1e-9,
    normalize_entropy: bool = True,
) -> WeightingResult:
    """Weight the lower-entropy model for the current denoising step."""
    pairwise, diagnostics = online_entropy_model_weights(
        [probabilities_a, probabilities_b],
        weight_temperature,
        epsilon,
        normalize_entropy,
    )
    mean_entropy = diagnostics.get("mean_entropies")
    return _result_from_pairwise_weights(
        pairwise,
        diagnostics=(
            {
                "mean_entropy_a": mean_entropy[0],
                "mean_entropy_b": mean_entropy[1],
            }
            if mean_entropy is not None
            else {}
        ),
    )


def per_token_margin_weights(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    weight_temperature: float = 1.0,
) -> WeightingResult:
    """Weight each model by its top-1 versus top-2 probability margin."""
    pairwise, diagnostics = per_token_margin_model_weights(
        [probabilities_a, probabilities_b], weight_temperature
    )
    margins = diagnostics["margins"]
    return _result_from_pairwise_weights(
        pairwise,
        diagnostics={"margin_a": margins[:, 0], "margin_b": margins[:, 1]},
    )
