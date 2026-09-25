"""Run with: pytest scripts/tests/test_tse_components.py -v"""

import pytest
import torch

from dllm.pipelines.tse.divergence import agreement_factor, entropy, jensen_shannon_divergence
from dllm.pipelines.tse.scoring import consensus_scores, fused_confidence
from dllm.pipelines.tse.selection import commit_tokens, select_positions


def test_identical_distributions_have_zero_jsd_and_full_agreement():
    probabilities = torch.tensor([[0.75, 0.25]])

    divergence = jensen_shannon_divergence(probabilities, probabilities)
    agreement = agreement_factor(divergence)

    assert torch.allclose(divergence, torch.zeros(1))
    assert torch.allclose(agreement, torch.ones(1))


def test_entropy_and_jsd_are_finite_with_zero_probabilities():
    probabilities_a = torch.tensor([[1.0, 0.0]])
    probabilities_b = torch.tensor([[0.0, 1.0]])

    values = torch.stack(
        [
            entropy(probabilities_a),
            jensen_shannon_divergence(probabilities_a, probabilities_b),
        ]
    )

    assert torch.isfinite(values).all()
    assert values[1] > 0


def test_consensus_score_uses_confidence_and_agreement():
    probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    confidence, token_ids = fused_confidence(probabilities)
    scores = consensus_scores(confidence, torch.tensor([1.0, 0.5]))

    assert torch.equal(token_ids, torch.tensor([0, 0]))
    assert torch.allclose(confidence, torch.tensor([0.8, 0.6]))
    assert torch.allclose(scores, torch.tensor([0.8, 0.3]))


def test_select_positions_respects_per_batch_transfer_counts():
    positions = torch.tensor([[0, 4], [0, 5], [1, 4]])
    scores = torch.tensor([0.2, 0.9, 0.7])
    predicted_tokens = torch.tensor([10, 11, 12])

    selected_positions, selected_tokens = select_positions(
        scores,
        predicted_tokens,
        positions,
        torch.tensor([1, 1]),
    )

    assert torch.equal(selected_positions, torch.tensor([[0, 5], [1, 4]]))
    assert torch.equal(selected_tokens, torch.tensor([11, 12]))


def test_commit_tokens_only_updates_selected_positions():
    canvas = torch.zeros((2, 6), dtype=torch.long)
    positions = torch.tensor([[0, 5], [1, 4]])
    token_ids = torch.tensor([11, 12])

    updated = commit_tokens(canvas, positions, token_ids)

    assert canvas.sum() == 0
    assert updated[0, 5] == 11
    assert updated[1, 4] == 12


def test_selection_rejects_excess_transfer_count():
    with pytest.raises(ValueError, match="exceeds active positions"):
        select_positions(
            torch.tensor([0.5]),
            torch.tensor([1]),
            torch.tensor([[0, 2]]),
            torch.tensor([2]),
        )