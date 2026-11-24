"""Confidence and consensus scoring for TSE position selection.

Run with: python -m pytest scripts/tests/test_tse_components.py
"""

import torch

from dllm.core.samplers.utils import add_gumbel_noise

from .divergence import agreement_factor, jensen_shannon_divergence


def baseline_candidates(
    probabilities: torch.Tensor, temperature: float, remasking: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict baseline tokens and score their positions for commitment."""
    if temperature == 0:
        scores, tokens = probabilities.max(dim=-1)
    else:
        predictions = add_gumbel_noise(probabilities.log(), temperature)
        predictions.masked_fill_(probabilities == 0, -torch.inf)
        tokens = predictions.argmax(dim=-1)
        scores = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    if remasking == "random":
        scores = torch.rand_like(scores)
    elif remasking != "low_confidence":
        raise ValueError(f"Unsupported remasking strategy: {remasking}")
    return scores, tokens


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


def distribution_consensus(
    confidence: torch.Tensor,
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    weights_a: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute the agreement-weighted confidence used in diagnostics."""
    divergence = jensen_shannon_divergence(
        probabilities_a, probabilities_b, weights_a, epsilon
    )
    return consensus_scores(confidence, agreement_factor(divergence, weights_a, epsilon))
