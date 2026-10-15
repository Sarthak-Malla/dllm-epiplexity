"""Run with: pytest scripts/tests/test_tse_components.py -v"""

import pytest
import torch

from dllm.pipelines.tse.divergence import (
    agreement_factor,
    entropy,
    generalized_jensen_shannon_divergence,
    jensen_shannon_divergence,
)
from dllm.pipelines.tse.scoring import consensus_scores, fused_confidence
from dllm.pipelines.tse.selection import (
    commit_tokens,
    select_positions,
    suppress_master_commit_scores,
    suppress_token_probabilities,
)


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


def test_begin_suppression_only_changes_master_commit_scores():
    scores = torch.tensor([0.9, 0.8, 0.7])
    predicted_tokens = torch.tensor([126081, 42, 126348])

    updated = suppress_master_commit_scores(
        scores,
        predicted_tokens,
        [126081, 126348],
    )

    assert torch.equal(predicted_tokens, torch.tensor([126081, 42, 126348]))
    assert torch.equal(scores, torch.tensor([0.9, 0.8, 0.7]))
    assert torch.isneginf(updated[[0, 2]]).all()
    assert updated[1] == scores[1]


def test_empty_begin_suppression_is_a_noop():
    scores = torch.tensor([0.9, 0.8])

    updated = suppress_master_commit_scores(scores, torch.tensor([1, 2]), ())

    assert updated is scores


def test_always_on_suppression_renormalizes_probabilities():
    probabilities = torch.tensor([[0.7, 0.2, 0.1]])

    updated = suppress_token_probabilities(probabilities, [0])

    assert updated[0, 0] == 0
    assert torch.allclose(updated.sum(dim=-1), torch.ones(1))
    assert torch.allclose(updated[0, 1:], torch.tensor([2 / 3, 1 / 3]))


def test_always_on_suppression_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="exceeds"):
        suppress_token_probabilities(torch.tensor([[0.5, 0.5]]), [2])


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


def test_generalized_jsd_supports_m_way_probabilities():
    probabilities = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.5, 0.5]]),
    ]

    divergence = generalized_jensen_shannon_divergence(
        probabilities, [0.25, 0.25, 0.5]
    )

    assert divergence.shape == (1,)
    assert torch.isfinite(divergence).all()
    assert divergence.item() > 0
