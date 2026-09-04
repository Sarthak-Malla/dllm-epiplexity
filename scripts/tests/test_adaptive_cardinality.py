"""Test Phase-7 adaptive candidate construction and size-aware scoring.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_adaptive_cardinality.py -v
"""

import pytest
import torch

from dllm.core.samplers.adaptive_cardinality import (
    apply_size_aware_scoring,
    generate_joint_k_dependency_candidates,
    generate_stopped_soft_full_candidates,
    parse_action_sizes,
)
from dllm.core.samplers.batched_lookahead import BatchedLookaheadOutput
from dllm.core.samplers.candidates import CandidateBatch


def _conflicted_four_position_case():
    """Return a case where the safe soft-full prefix has size two."""
    dependency = torch.zeros((1, 4, 4), dtype=torch.float32)
    dependency[0, 0, 1] = 4.0
    dependency[0, 1, 0] = 3.0
    dependency[0, 2, 3] = 2.0
    dependency[0, 3, 2] = 1.0
    entropy = torch.ones((1, 4), dtype=torch.float32)
    confidence = torch.zeros((1, 4), dtype=torch.float32)
    eligible = torch.ones((1, 4), dtype=torch.bool)
    return dependency, entropy, confidence, eligible


def _variable_size_candidate_batch() -> CandidateBatch:
    """Build one k=1 and one k=2 action over the same two positions."""
    masks = torch.tensor(
        [[[True, False]], [[True, True]]],
        dtype=torch.bool,
    )
    return CandidateBatch(
        candidate_masks=masks,
        names=("k1", "k2"),
        proposal_scores=torch.tensor([[1.0], [2.0]]),
        seed_anchors=torch.tensor([[0], [0]]),
        selected_positions=torch.tensor([[[0, -1]], [[0, 1]]]),
        mean_within_set_dependency=torch.zeros((2, 1)),
        candidate_valid=torch.ones((2, 1), dtype=torch.bool),
        eligible_mask=torch.ones((1, 2), dtype=torch.bool),
        requested_k=torch.tensor([2]),
        clipped_k=torch.tensor([2]),
        metadata=({"source": "k1"}, {"source": "k2"}),
        generation_seed=0,
        configuration={"variable": True},
        action_sizes=torch.tensor([[1], [2]]),
    )


def _lookahead(candidates: CandidateBatch, scores: tuple[float, float]):
    """Build an aligned raw lookahead result for controlled rescoring."""
    raw = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    return BatchedLookaheadOutput(
        metric="entropy_drop",
        scores=raw,
        base_metric_sums=torch.zeros_like(raw),
        lookahead_metric_sums=torch.zeros_like(raw),
        heldout_counts=torch.tensor([[1], [0]]),
        best_index=torch.tensor([1]),
        best_score=raw[1].clone(),
        best_mask=candidates.candidate_masks[1].clone(),
        best_names=("k2",),
        best_metadata=(candidates.metadata[1],),
        model_calls=2,
        candidate_chunk_size=1,
    )


def test_action_size_parser_requires_sorted_unique_positive_sizes():
    assert parse_action_sizes("1|2|4") == (1, 2, 4)
    assert parse_action_sizes((1, 3)) == (1, 3)
    for invalid in ("", "1|1", "2|1", "0|1", "one|two"):
        with pytest.raises((TypeError, ValueError)):
            parse_action_sizes(invalid)


def test_marginal_stopping_guarantees_one_and_accepts_safe_second_position():
    dependency, entropy, confidence, eligible = _conflicted_four_position_case()
    singleton = generate_stopped_soft_full_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        candidate_budget=1,
        maximum_action_size=4,
        stopping_rule="marginal_utility",
        utility_threshold=3.0,
        conflict_normalization="none",
        position_temperature=0.0,
    )
    safe_pair = generate_stopped_soft_full_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        candidate_budget=1,
        maximum_action_size=4,
        stopping_rule="marginal_utility",
        utility_threshold=0.0,
        conflict_normalization="none",
        position_temperature=0.0,
    )

    assert singleton.selected_positions[0, 0].tolist() == [0, -1, -1, -1]
    assert singleton.action_sizes[0, 0].item() == 1
    assert safe_pair.selected_positions[0, 0].tolist() == [0, 2, -1, -1]
    assert safe_pair.action_sizes[0, 0].item() == 2
    assert safe_pair.metadata[0]["stopping_reason_by_batch"] == (
        "marginal_utility_threshold",
    )


def test_entropy_budget_stops_before_an_uncertain_addition():
    dependency = torch.zeros((1, 3, 3), dtype=torch.float32)
    entropy = torch.tensor([[0.2, 0.3, 0.8]])
    confidence = torch.full((1, 3), 0.5)
    eligible = torch.ones((1, 3), dtype=torch.bool)

    candidates = generate_stopped_soft_full_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        candidate_budget=1,
        maximum_action_size=3,
        stopping_rule="entropy_budget",
        entropy_budget=0.6,
        position_temperature=0.0,
    )

    assert candidates.selected_positions[0, 0].tolist() == [0, 1, -1]
    assert candidates.action_sizes[0, 0].item() == 2
    assert candidates.metadata[0]["stopping_reason_by_batch"] == (
        "entropy_budget",
    )


def test_joint_pool_includes_size_identity_and_clips_when_masks_are_fewer():
    dependency = torch.zeros((1, 4, 4), dtype=torch.float32)
    entropy = torch.ones((1, 4), dtype=torch.float32)
    confidence = torch.full((1, 4), 0.5)
    all_eligible = torch.ones((1, 4), dtype=torch.bool)

    complete = generate_joint_k_dependency_candidates(
        dependency,
        entropy,
        confidence,
        all_eligible,
        action_sizes="1|2|4",
        candidate_budget_per_size=1,
        position_temperature=0.0,
    )
    assert complete.action_sizes[:, 0].tolist() == [1, 2, 4]
    assert [row["requested_action_size"] for row in complete.metadata] == [1, 2, 4]

    two_eligible = torch.tensor([[True, True, False, False]])
    clipped = generate_joint_k_dependency_candidates(
        dependency,
        entropy,
        confidence,
        two_eligible,
        action_sizes="1|2|4",
        candidate_budget_per_size=1,
        position_temperature=0.0,
    )
    assert clipped.action_sizes[:, 0].tolist() == [1, 2, 0]
    assert clipped.candidate_valid[:, 0].tolist() == [True, True, False]


def test_size_aware_rules_can_choose_k1_or_safe_k2():
    candidates = _variable_size_candidate_batch()
    uncertain_second = torch.tensor([[0.1, 5.0]])
    raw_prefers_k2 = _lookahead(candidates, (4.0, 6.0))

    immediate_cost = apply_size_aware_scoring(
        raw_prefers_k2,
        candidates,
        uncertain_second,
        rule="immediate_cost",
        immediate_cost_weight=1.0,
    )
    per_token = apply_size_aware_scoring(
        raw_prefers_k2,
        candidates,
        uncertain_second,
        rule="per_token",
    )
    assert immediate_cost.lookahead.best_index.item() == 0
    assert per_token.lookahead.best_index.item() == 0
    assert torch.equal(immediate_cost.raw_scores, raw_prefers_k2.scores)

    safe_k2_gain = _lookahead(candidates, (3.0, 7.0))
    safe_pair = apply_size_aware_scoring(
        safe_k2_gain,
        candidates,
        torch.tensor([[0.1, 0.1]]),
        rule="per_token",
    )
    assert safe_pair.lookahead.best_index.item() == 1
    assert safe_pair.lookahead.best_mask.sum().item() == 2
