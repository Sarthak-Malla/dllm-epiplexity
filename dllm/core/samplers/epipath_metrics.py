"""
Metrics used by the EpiPath sampler.

Run tests from the repository root with:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_metrics.py -v
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1e-12


def compute_probs_and_logprobs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return softmax probabilities and log probabilities for logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs


def compute_entropy(probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """Compute predictive entropy over the final dimension."""
    return -(probs * log_probs).sum(dim=-1)


def compute_topk_metrics(
    logits: torch.Tensor,
    margin_threshold: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute per-position top-token, entropy, risk, NLL, and margin metrics."""
    probs, log_probs = compute_probs_and_logprobs(logits)
    entropy = compute_entropy(probs, log_probs)
    topk = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top1_prob = topk.values[..., 0]
    top1_token_id = topk.indices[..., 0]
    if topk.values.shape[-1] > 1:
        top2_prob = topk.values[..., 1]
    else:
        top2_prob = torch.zeros_like(top1_prob)
    margin = top1_prob - top2_prob
    risk = 1.0 - top1_prob
    nll = -torch.log(top1_prob.clamp_min(EPS))
    margin_deficit = torch.clamp(
        torch.as_tensor(margin_threshold, device=logits.device, dtype=margin.dtype)
        - margin,
        min=0.0,
    )
    stable = margin >= margin_threshold
    return {
        "top1_token_id": top1_token_id,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": margin,
        "entropy": entropy,
        "risk": risk,
        "nll": nll,
        "margin_deficit": margin_deficit,
        "stable": stable,
        "probs": probs,
        "log_probs": log_probs,
    }


def tensor_to_float(value: torch.Tensor | float | int) -> float:
    """Convert a scalar tensor or number to a JSON-safe float."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.detach().float().item()
    return float(value)


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the masked sum as a Python float."""
    if mask.sum().item() == 0:
        return 0.0
    return tensor_to_float(values[mask].sum())


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the masked mean as a Python float."""
    if mask.sum().item() == 0:
        return 0.0
    return tensor_to_float(values[mask].mean())


def compute_state_metrics(
    metrics: dict[str, torch.Tensor],
    mask: torch.Tensor,
    entropy_weight: float = 0.5,
    margin_deficit_weight: float = 1.0,
) -> dict[str, float]:
    """Aggregate per-position metrics over a mask."""
    residual = (
        entropy_weight * metrics["entropy"]
        + margin_deficit_weight * metrics["margin_deficit"]
    )
    total = int(mask.sum().item())
    return {
        "num_masked": float(total),
        "state_entropy_sum": masked_sum(metrics["entropy"], mask),
        "state_entropy_mean": masked_mean(metrics["entropy"], mask),
        "state_risk_sum": masked_sum(metrics["risk"], mask),
        "state_risk_mean": masked_mean(metrics["risk"], mask),
        "state_margin_deficit_sum": masked_sum(metrics["margin_deficit"], mask),
        "state_margin_deficit_mean": masked_mean(metrics["margin_deficit"], mask),
        "state_stable_frac": masked_mean(metrics["stable"].float(), mask),
        "state_residual_epiplexity_mean": masked_mean(residual, mask),
        "state_residual_epiplexity_sum": masked_sum(residual, mask),
    }


def compute_commit_metrics(
    metrics: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute confidence and fragility metrics for committed positions."""
    return {
        "commit_conf_mean": masked_mean(metrics["top1_prob"], candidate_mask),
        "commit_risk_mean": masked_mean(metrics["risk"], candidate_mask),
        "commit_nll_mean": masked_mean(metrics["nll"], candidate_mask),
        "commit_margin_deficit_mean": masked_mean(
            metrics["margin_deficit"], candidate_mask
        ),
    }


def compute_js_divergence(
    base_probs: torch.Tensor,
    lookahead_probs: torch.Tensor,
    base_log_probs: torch.Tensor,
    lookahead_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute exact Jensen-Shannon divergence for matching distributions."""
    mixture = 0.5 * (base_probs + lookahead_probs)
    log_mixture = torch.log(mixture.clamp_min(EPS))
    kl_base = (base_probs * (base_log_probs - log_mixture)).sum(dim=-1)
    kl_lookahead = (lookahead_probs * (lookahead_log_probs - log_mixture)).sum(dim=-1)
    return 0.5 * (kl_base + kl_lookahead)


def compute_coverage_metrics(
    positions: list[int],
    revealed_positions: list[int],
    sequence_length: int,
) -> dict[str, float]:
    """Compute simple span, distance, coverage, and clustering metrics."""
    if not positions:
        return {
            "span_norm": 0.0,
            "mean_pairwise_distance_norm": 0.0,
            "dist_to_revealed_mean": 0.0,
            "cluster_penalty": 0.0,
            "coverage_bonus": 0.0,
        }

    denom = max(1, sequence_length - 1)
    if len(positions) > 1:
        span_norm = (max(positions) - min(positions)) / denom
        pairwise = [
            abs(i - j) / denom for i, j in itertools.combinations(positions, 2)
        ]
        mean_pairwise_distance_norm = sum(pairwise) / len(pairwise)
    else:
        span_norm = 0.0
        mean_pairwise_distance_norm = 0.0

    if revealed_positions:
        distances = [
            min(abs(pos - revealed) for revealed in revealed_positions) / denom
            for pos in positions
        ]
        dist_to_revealed_mean = sum(distances) / len(distances)
    else:
        dist_to_revealed_mean = 1.0

    cluster_penalty = 1.0 - mean_pairwise_distance_norm
    coverage_bonus = 0.5 * span_norm + 0.5 * dist_to_revealed_mean
    return {
        "span_norm": float(span_norm),
        "mean_pairwise_distance_norm": float(mean_pairwise_distance_norm),
        "dist_to_revealed_mean": float(dist_to_revealed_mean),
        "cluster_penalty": float(cluster_penalty),
        "coverage_bonus": float(coverage_bonus),
    }


def compute_lookahead_metrics(
    base_metrics: dict[str, torch.Tensor],
    lookahead_metrics: dict[str, torch.Tensor],
    current_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    entropy_weight: float = 0.5,
    margin_deficit_weight: float = 1.0,
) -> dict[str, float]:
    """Compute heldout-only candidate metrics for one batch row."""
    heldout_mask = current_mask & (~candidate_mask)
    if heldout_mask.sum().item() == 0:
        return {
            "Delta_H_sum": 0.0,
            "Delta_H_mean": 0.0,
            "Delta_R_sum": 0.0,
            "Delta_R_mean": 0.0,
            "Delta_margin_mean": 0.0,
            "Delta_stable_count": 0.0,
            "Delta_stable_frac": 0.0,
            "top1_agreement_frac": 0.0,
            "top1_change_frac": 0.0,
            "structural_leverage_sum": 0.0,
            "structural_leverage_mean": 0.0,
            "EpiGain_1": 0.0,
        }

    base_entropy = base_metrics["entropy"]
    lookahead_entropy = lookahead_metrics["entropy"]
    base_risk = base_metrics["risk"]
    lookahead_risk = lookahead_metrics["risk"]
    base_margin = base_metrics["margin"]
    lookahead_margin = lookahead_metrics["margin"]
    base_stable = base_metrics["stable"].float()
    lookahead_stable = lookahead_metrics["stable"].float()
    base_residual = (
        entropy_weight * base_entropy
        + margin_deficit_weight * base_metrics["margin_deficit"]
    )
    lookahead_residual = (
        entropy_weight * lookahead_entropy
        + margin_deficit_weight * lookahead_metrics["margin_deficit"]
    )

    delta_entropy = base_entropy - lookahead_entropy
    delta_risk = base_risk - lookahead_risk
    delta_stable = lookahead_stable - base_stable
    top1_agreement = (
        base_metrics["top1_token_id"] == lookahead_metrics["top1_token_id"]
    ).float()

    js = compute_js_divergence(
        base_metrics["probs"][heldout_mask],
        lookahead_metrics["probs"][heldout_mask],
        base_metrics["log_probs"][heldout_mask],
        lookahead_metrics["log_probs"][heldout_mask],
    )
    sharpened = (
        lookahead_entropy[heldout_mask] < base_entropy[heldout_mask]
    ).float()
    structural_leverage = js * sharpened
    epi_gain = base_residual - lookahead_residual

    return {
        "Delta_H_sum": masked_sum(delta_entropy, heldout_mask),
        "Delta_H_mean": masked_mean(delta_entropy, heldout_mask),
        "Delta_R_sum": masked_sum(delta_risk, heldout_mask),
        "Delta_R_mean": masked_mean(delta_risk, heldout_mask),
        "Delta_margin_mean": masked_mean(
            lookahead_margin - base_margin, heldout_mask
        ),
        "Delta_stable_count": masked_sum(delta_stable, heldout_mask),
        "Delta_stable_frac": masked_mean(delta_stable, heldout_mask),
        "top1_agreement_frac": masked_mean(top1_agreement, heldout_mask),
        "top1_change_frac": 1.0 - masked_mean(top1_agreement, heldout_mask),
        "structural_leverage_sum": tensor_to_float(structural_leverage.sum()),
        "structural_leverage_mean": tensor_to_float(structural_leverage.mean()),
        "EpiGain_1": masked_mean(epi_gain, heldout_mask),
    }


def score_candidates(
    candidates: list[Any],
    score_mode: str,
    normalize: bool = True,
    lambda_commit: float = 0.5,
    lambda_cluster: float = 0.1,
    lambda_coverage: float = 0.1,
) -> None:
    """Assign scores in-place to CandidateReveal-like objects."""
    if not candidates:
        return

    if score_mode == "greedy_baseline":
        for candidate in candidates:
            candidate.score = 1.0 if "greedy" in candidate.proposal_sources else 0.0
        return

    key_by_mode = {
        "entropy_drop": "Delta_H_mean",
        "risk_reduction": "Delta_R_mean",
        "epigain": "EpiGain_1",
    }
    if score_mode in key_by_mode:
        metric_key = key_by_mode[score_mode]
        for candidate in candidates:
            candidate.score = float(candidate.metrics.get(metric_key, 0.0))
        return

    if score_mode == "hybrid_epipath":
        weights = {
            "Delta_H_mean": 0.5,
            "Delta_R_mean": 0.5,
            "structural_leverage_mean": 1.0,
            "EpiGain_1": 1.0,
            "commit_risk_mean": -lambda_commit,
            "cluster_penalty": -lambda_cluster,
            "coverage_bonus": lambda_coverage,
        }
        values_by_key: dict[str, list[float]] = {
            key: [float(c.metrics.get(key, 0.0)) for c in candidates]
            for key in weights
        }
        normalized: dict[str, list[float]] = {}
        for key, values in values_by_key.items():
            if not normalize or len(values) == 1:
                normalized[key] = values
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance) + EPS
            normalized[key] = [(value - mean) / std for value in values]
        for index, candidate in enumerate(candidates):
            candidate.score = sum(
                weight * normalized[key][index] for key, weight in weights.items()
            )
        return

    raise ValueError(f"Unknown EpiPath score mode: {score_mode}")
