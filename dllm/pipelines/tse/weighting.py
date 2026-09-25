"""Adaptive model weighting strategies for TSE."""

from dataclasses import dataclass

import torch

from dllm.pipelines.tse.divergence import entropy


@dataclass
class WeightingResult:
    """Model weights and optional diagnostics for active positions."""

    weights_a: torch.Tensor
    weights_b: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


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


def static_weights(
    alpha: float,
    num_positions: int,
    device: torch.device | None = None,
) -> WeightingResult:
    """Return fixed Model A/Model B weights for every active position."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if num_positions < 0:
        raise ValueError("num_positions must be non-negative")
    pairwise = torch.tensor(
        [alpha, 1 - alpha], dtype=torch.float32, device=device
    ).expand(num_positions, -1)
    return _result_from_pairwise_weights(pairwise)


def online_entropy_weights(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    weight_temperature: float = 1.0,
    epsilon: float = 1e-9,
    normalize_entropy: bool = True,
) -> WeightingResult:
    """Weight the lower-entropy model for the current denoising step."""
    _validate_probabilities(probabilities_a, probabilities_b)
    _validate_temperature(weight_temperature)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if probabilities_a.shape[0] == 0:
        return _result_from_pairwise_weights(
            probabilities_a.new_empty((0, 2), dtype=torch.float32)
        )

    entropy_a = entropy(probabilities_a, epsilon)
    entropy_b = entropy(probabilities_b, epsilon)
    if normalize_entropy:
        normalizer = torch.log(
            torch.tensor(
                probabilities_a.shape[-1],
                dtype=torch.float32,
                device=probabilities_a.device,
            )
        ).clamp_min(epsilon)
        entropy_a = entropy_a / normalizer
        entropy_b = entropy_b / normalizer

    mean_entropy = torch.stack([entropy_a.mean(), entropy_b.mean()])
    pairwise = torch.softmax(-mean_entropy / weight_temperature, dim=0).expand(
        probabilities_a.shape[0], -1
    )
    return _result_from_pairwise_weights(
        pairwise,
        diagnostics={
            "mean_entropy_a": mean_entropy[0],
            "mean_entropy_b": mean_entropy[1],
        },
    )


def per_token_margin_weights(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    weight_temperature: float = 1.0,
) -> WeightingResult:
    """Weight each model by its top-1 versus top-2 probability margin."""
    _validate_probabilities(probabilities_a, probabilities_b)
    _validate_temperature(weight_temperature)
    if probabilities_a.shape[-1] < 2:
        raise ValueError("margin weighting requires at least two vocabulary entries")

    top_two_a = torch.topk(probabilities_a.float(), k=2, dim=-1).values
    top_two_b = torch.topk(probabilities_b.float(), k=2, dim=-1).values
    margin_a = top_two_a[:, 0] - top_two_a[:, 1]
    margin_b = top_two_b[:, 0] - top_two_b[:, 1]
    pairwise = torch.softmax(
        torch.stack([margin_a, margin_b], dim=-1) / weight_temperature,
        dim=-1,
    )
    return _result_from_pairwise_weights(
        pairwise,
        diagnostics={"margin_a": margin_a, "margin_b": margin_b},
    )