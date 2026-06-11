"""
Unit tests for EpiPath metrics.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_metrics.py -v
"""

import torch

from dllm.core.samplers.epipath_metrics import (
    compute_lookahead_metrics,
    compute_probs_and_logprobs,
    compute_topk_metrics,
    score_candidates,
)
from dllm.core.samplers.epipath_proposals import CandidateReveal


def test_compute_topk_metrics_matches_softmax_quantities():
    logits = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])
    metrics = compute_topk_metrics(logits, margin_threshold=0.2)
    probs, _ = compute_probs_and_logprobs(logits)

    expected_top1 = probs.amax(dim=-1)
    assert torch.allclose(metrics["top1_prob"], expected_top1)
    assert torch.allclose(metrics["risk"], 1.0 - expected_top1)
    assert torch.equal(metrics["top1_token_id"], torch.tensor([[0, 0]]))
    assert metrics["stable"][0, 0]


def test_lookahead_metrics_ignore_committed_positions():
    base_logits = torch.zeros(3, 4)
    lookahead_logits = base_logits.clone()
    lookahead_logits[0, 2] = 8.0

    base_metrics = compute_topk_metrics(base_logits, margin_threshold=0.2)
    lookahead_metrics = compute_topk_metrics(lookahead_logits, margin_threshold=0.2)
    current_mask = torch.tensor([True, True, True])
    candidate_mask = torch.tensor([True, False, False])

    metrics = compute_lookahead_metrics(
        base_metrics=base_metrics,
        lookahead_metrics=lookahead_metrics,
        current_mask=current_mask,
        candidate_mask=candidate_mask,
    )

    assert metrics["Delta_H_mean"] == 0.0
    assert metrics["Delta_R_mean"] == 0.0
    assert metrics["EpiGain_1"] == 0.0


def test_hybrid_score_normalizes_within_candidate_set():
    low = CandidateReveal(
        candidate_id="low",
        proposal_type="low",
        positions=[0],
        token_ids=[1],
        proposal_sources=["low"],
        metrics={
            "Delta_H_mean": 0.0,
            "Delta_R_mean": 0.0,
            "structural_leverage_mean": 0.0,
            "EpiGain_1": 0.0,
            "commit_risk_mean": 1.0,
            "cluster_penalty": 1.0,
            "coverage_bonus": 0.0,
        },
    )
    high = CandidateReveal(
        candidate_id="high",
        proposal_type="high",
        positions=[1],
        token_ids=[1],
        proposal_sources=["high"],
        metrics={
            "Delta_H_mean": 1.0,
            "Delta_R_mean": 1.0,
            "structural_leverage_mean": 1.0,
            "EpiGain_1": 1.0,
            "commit_risk_mean": 0.0,
            "cluster_penalty": 0.0,
            "coverage_bonus": 1.0,
        },
    )

    score_candidates([low, high], score_mode="hybrid_epipath", normalize=True)

    assert high.score > low.score
