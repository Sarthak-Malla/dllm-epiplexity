"""Test Phase-6 conflicts, parallel subsets, and committed-anchor support.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_parallel_candidates.py -v
"""

import pytest
import torch
from unittest import mock

from dllm.core.samplers import parallel_candidates as parallel_module

from dllm.core.samplers.parallel_candidates import (
    anchor_support_scores,
    build_symmetric_conflict_matrix,
    gather_committed_anchor_state,
    generate_parallel_dependency_candidates,
    initialize_committed_anchor_state,
    record_committed_anchors,
    within_action_conflict_statistics,
)


def _four_position_case():
    """Return utilities whose top two positions have a strong conflict."""
    dependency = torch.zeros((1, 4, 4), dtype=torch.float32)
    dependency[0, 0, 1] = 4.0
    dependency[0, 1, 0] = 3.0
    dependency[0, 2, 3] = 2.0
    dependency[0, 3, 2] = 1.0
    entropy = torch.ones((1, 4), dtype=torch.float32)
    confidence = torch.zeros((1, 4), dtype=torch.float32)
    eligible = torch.ones((1, 4), dtype=torch.bool)
    return dependency, entropy, confidence, eligible


def _seed_direction_case():
    """Make incoming, entropy-weighted incoming, and confidence prefer different seeds."""
    dependency = torch.zeros((1, 5, 5))
    dependency[0, 0, 1] = 4.0
    dependency[0, 2, 1] = 3.0
    dependency[0, 1, 3] = 0.5
    return (dependency, torch.tensor([[1.0, 3.0, 1.0, 0.1, 1.0]]),
            torch.tensor([[0.95, 0.6, 0.7, 0.8, 0.5]]), torch.ones((1, 5), dtype=torch.bool))


@pytest.mark.parametrize("strategy,weight,seed,positions", [
    ("legacy", 0.0, 0, [0, 2]),
    ("incoming", 0.0, 1, [0, 1]),
    ("incoming", 1.0, 3, [0, 3]),
    ("confidence", 0.0, 0, [0, 2]),
])
def test_seed_direction_and_own_entropy_change_seed_without_changing_companion_rule(strategy, weight, seed, positions):
    # Incoming scores: seed 1 = .6*(4*1+3*1)=4.2; seed 3 = .8*.5*3=1.2.
    # Own-entropy weighting reverses them: 4.2*exp(-3) < 1.2*exp(-.1).
    # Outgoing companion utility still ranks position 0 first, then position 2.
    pool = generate_parallel_dependency_candidates(
        *_seed_direction_case(), requested_k=2, candidate_budget=1,
        confidence_exponent=1.0, conflict_penalty=0.0, position_temperature=0.0,
        seed_strategy=strategy, seed_entropy_weight=weight,
    )
    assert pool.seed_anchors[0, 0].item() == seed
    assert pool.selected_positions[0, 0].tolist() == positions


def test_seed_overrides_leave_companion_inputs_and_refill_order_unchanged():
    inputs = _seed_direction_case()
    snapshots = []
    for strategy, weight in (("legacy", 0.0), ("incoming", 0.0), ("incoming", 1.0), ("confidence", 0.0)):
        # Fix seed order to compare construction given identical seeds. A full
        # pool forces the combination refill path after seed-generated groups.
        with (
            mock.patch.object(parallel_module, "_seed_order", return_value=list(range(5))),
            mock.patch.object(parallel_module, "_construct_subset", wraps=parallel_module._construct_subset) as construct,
        ):
            pool = generate_parallel_dependency_candidates(
                *inputs, requested_k=2, candidate_budget=10, confidence_exponent=1.0,
                seed_strategy=strategy, seed_entropy_weight=weight,
            )
            snapshots.append((pool, construct.call_args_list))
    original, original_calls = snapshots[0]
    for pool, calls in snapshots[1:]:
        assert torch.equal(pool.candidate_masks, original.candidate_masks)
        assert torch.equal(pool.proposal_scores, original.proposal_scores)
        assert len(calls) == len(original_calls)
        for call, reference in zip(calls, original_calls):
            for key in ("base_utility", "support", "confidence", "conflict"):
                assert torch.equal(call.kwargs[key], reference.kwargs[key])


def test_legacy_seed_default_preserves_candidates_and_rng_order():
    kwargs = dict(requested_k=2, candidate_budget=6, confidence_exponent=1.0, generation_seed=42)
    default = generate_parallel_dependency_candidates(*_seed_direction_case(), **kwargs)
    explicit = generate_parallel_dependency_candidates(
        *_seed_direction_case(), **kwargs, seed_strategy="legacy", seed_entropy_weight=0.0,
    )
    assert torch.equal(default.candidate_masks, explicit.candidate_masks)
    assert torch.equal(default.seed_anchors, explicit.seed_anchors)
    assert torch.equal(default.proposal_scores, explicit.proposal_scores)


@pytest.mark.parametrize("strategy,weight", [("bad", 0), ("legacy", 1), ("confidence", 1), ("incoming", -1), ("incoming", float("nan"))])
def test_invalid_seed_settings_fail_early(strategy, weight):
    with pytest.raises(ValueError):
        generate_parallel_dependency_candidates(
            *_seed_direction_case(), requested_k=2, candidate_budget=1,
            seed_strategy=strategy, seed_entropy_weight=weight,
        )


def test_asymmetric_dependency_becomes_symmetric_masked_only_conflict():
    dependency = torch.tensor(
        [
            [
                [99.0, 0.8, 0.4, 7.0],
                [0.2, 99.0, 0.6, 8.0],
                [0.0, 1.0, 99.0, 9.0],
                [3.0, 4.0, 5.0, 99.0],
            ]
        ]
    )
    eligible = torch.tensor([[True, True, True, False]])

    raw = build_symmetric_conflict_matrix(
        dependency,
        eligible,
        normalization="none",
    )
    expected = torch.tensor(
        [[[0.0, 0.5, 0.2, 0.0], [0.5, 0.0, 0.8, 0.0], [0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    torch.testing.assert_close(raw.matrix, expected)
    assert torch.equal(raw.matrix, raw.matrix.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(raw.matrix, dim1=-2, dim2=-1)) == 0

    normalized = build_symmetric_conflict_matrix(
        dependency,
        eligible,
        normalization="max",
    )
    torch.testing.assert_close(normalized.scale_by_batch, torch.tensor([0.8]))
    assert normalized.matrix.max().item() == pytest.approx(1.0)


def test_soft_exact_risk_penalty_avoids_the_high_conflict_second_position():
    dependency, entropy, confidence, eligible = _four_position_case()

    soft = generate_parallel_dependency_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        requested_k=2,
        candidate_budget=1,
        variant="soft_no_anchor",
        conflict_normalization="none",
        conflict_penalty=1.0,
        position_temperature=0.0,
    )

    # Seed 0 has utility 4. Position 1 has utility 3 but pays conflict 3.5,
    # whereas position 2 has utility 2 and no conflict with the seed.
    assert soft.selected_positions[0, 0].tolist() == [0, 2]
    assert soft.configuration["interaction_risk"] == (
        "1 - min(confidence_i, confidence_r)"
    )
    assert soft.mean_within_set_dependency[0, 0].item() == 0.0
    assert soft.metadata[0]["max_within_set_conflict_by_batch"] == (0.0,)


def test_soft_hard_correlated_and_confidence_controls_share_the_seed():
    dependency, entropy, confidence, eligible = _four_position_case()
    confidence = torch.tensor([[0.9, 0.8, 0.4, 0.3]])
    variants = {
        variant: generate_parallel_dependency_candidates(
            dependency,
            entropy,
            confidence,
            eligible,
            requested_k=2,
            candidate_budget=1,
            variant=variant,
            conflict_normalization="none",
            conflict_penalty=5.0,
            hard_conflict_threshold=0.5,
            position_temperature=0.0,
        )
        for variant in (
            "soft_no_anchor",
            "hard_low_conflict",
            "correlated_together",
            "top_confidence",
        )
    }

    assert variants["soft_no_anchor"].selected_positions[0, 0].tolist() == [0, 2]
    assert variants["hard_low_conflict"].selected_positions[0, 0].tolist() == [0, 2]
    assert variants["correlated_together"].selected_positions[0, 0].tolist() == [0, 1]
    assert variants["top_confidence"].selected_positions[0, 0].tolist() == [0, 1]
    assert variants["soft_no_anchor"].mean_within_set_dependency[0, 0] < (
        variants["top_confidence"].mean_within_set_dependency[0, 0]
    )


def test_hard_independent_set_uses_documented_fallback_and_reaches_k():
    dependency = torch.ones((1, 4, 4), dtype=torch.float32)
    dependency[:, torch.arange(4), torch.arange(4)] = 0
    entropy = torch.ones((1, 4), dtype=torch.float32)
    confidence = torch.full((1, 4), 0.5)
    eligible = torch.ones((1, 4), dtype=torch.bool)

    hard = generate_parallel_dependency_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        requested_k=3,
        candidate_budget=1,
        variant="hard_low_conflict",
        conflict_normalization="none",
        hard_conflict_threshold=0.2,
        position_temperature=0.0,
    )

    assert hard.candidate_masks[0, 0].sum().item() == 3
    assert hard.metadata[0]["hard_fallback_count_by_batch"] == (2,)
    assert hard.metadata[0]["fallback_source_by_batch"] == (
        "hard_lowest_conflict",
    )
    assert hard.configuration["hard_fallback"] == (
        "lowest_max_conflict_then_lowest_sum_then_utility"
    )


def test_anchor_history_tracks_only_executed_positions_and_resets_cleanly():
    reference = torch.zeros((2, 5), dtype=torch.long)
    initial = initialize_committed_anchor_state(
        reference,
        confidence_threshold=0.8,
    )
    executed = torch.tensor(
        [[False, True, False, False, False], [False, False, True, False, False]]
    )
    confidence = torch.tensor(
        [[0.1, 0.9, 0.7, 0.2, 0.1], [0.1, 0.2, 0.7, 0.9, 0.1]]
    )
    token_ids = torch.arange(10).reshape(2, 5)
    updated = record_committed_anchors(
        initial,
        executed,
        confidence,
        token_ids,
        commit_step=3,
    )

    assert torch.equal(updated.committed_position_mask, executed)
    assert updated.reliable_anchor_mask.tolist() == [
        [False, True, False, False, False],
        [False, False, False, False, False],
    ]
    assert updated.confidence_at_commit[0, 1].item() == pytest.approx(0.9)
    assert updated.commit_step[0, 1].item() == 3
    assert updated.token_id_at_commit[0, 1].item() == 1
    assert not updated.committed_position_mask[0, 2]

    reset = initialize_committed_anchor_state(
        reference,
        confidence_threshold=0.8,
    )
    assert not torch.any(reset.committed_position_mask)
    assert torch.all(reset.commit_step == -1)


def test_anchor_support_uses_target_query_to_reliable_anchor_key_per_row():
    reference = torch.zeros((2, 4), dtype=torch.long)
    state = initialize_committed_anchor_state(
        reference,
        confidence_threshold=0.8,
    )
    executed = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    confidence = torch.tensor(
        [[0.1, 0.9, 0.4, 0.3], [0.1, 0.2, 0.7, 0.4]]
    )
    tokens = torch.arange(8).reshape(2, 4)
    state = record_committed_anchors(
        state,
        executed,
        confidence,
        tokens,
        commit_step=0,
    )
    dependency = torch.zeros((2, 4, 4), dtype=torch.float32)
    dependency[0, 2, 1] = 0.5
    dependency[0, 1, 2] = 0.1
    dependency[1, 3, 2] = 10.0
    eligible = torch.tensor(
        [[False, False, True, True], [True, True, False, True]]
    )

    support = anchor_support_scores(dependency, eligible, state)

    assert support[0, 2].item() == pytest.approx(0.45)
    assert support[0, 1].item() == 0.0
    assert torch.count_nonzero(support[1]).item() == 0


def test_no_committed_anchor_and_masked_predictions_give_zero_support():
    dependency = torch.ones((1, 3, 3), dtype=torch.float32)
    eligible = torch.tensor([[False, True, True]])
    empty = initialize_committed_anchor_state(
        eligible,
        confidence_threshold=0.0,
    )
    support = anchor_support_scores(dependency, eligible, empty)
    assert torch.equal(support, torch.zeros_like(support))


def test_compact_anchor_gather_preserves_independent_batch_histories():
    reference = torch.zeros((2, 5), dtype=torch.long)
    state = initialize_committed_anchor_state(
        reference,
        confidence_threshold=0.5,
    )
    executed = torch.tensor(
        [[False, True, False, False, False], [False, False, False, True, False]]
    )
    confidence = torch.full((2, 5), 0.9)
    tokens = torch.arange(10).reshape(2, 5)
    state = record_committed_anchors(
        state,
        executed,
        confidence,
        tokens,
        commit_step=1,
    )
    positions = torch.tensor([[1, 2, -1], [0, 3, 4]])
    valid = positions >= 0
    compact = gather_committed_anchor_state(state, positions, valid)

    assert compact.committed_position_mask.tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert compact.token_id_at_commit.tolist() == [[1, -1, -1], [-1, 8, -1]]


@pytest.mark.parametrize("k", (2, 4))
@pytest.mark.parametrize("variant", ("soft_full", "hard_low_conflict"))
def test_fixed_k_candidates_are_reproducible_valid_and_exact(k, variant):
    torch.manual_seed(4)
    dependency = torch.rand((2, 6, 6), dtype=torch.float32)
    entropy = torch.rand((2, 6), dtype=torch.float32)
    confidence = torch.rand((2, 6), dtype=torch.float32)
    eligible = torch.tensor(
        [[True, True, True, True, True, True], [False, True, True, True, True, False]]
    )
    first = generate_parallel_dependency_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        requested_k=k,
        candidate_budget=4,
        variant=variant,
        generation_seed=17,
    )
    repeated = generate_parallel_dependency_candidates(
        dependency,
        entropy,
        confidence,
        eligible,
        requested_k=k,
        candidate_budget=4,
        variant=variant,
        generation_seed=17,
    )

    assert torch.equal(first.candidate_masks, repeated.candidate_masks)
    assert torch.equal(first.seed_anchors, repeated.seed_anchors)
    assert first.metadata == repeated.metadata
    assert not torch.any(first.candidate_masks & ~eligible.unsqueeze(0))
    for row, eligible_count in enumerate(eligible.sum(dim=-1).tolist()):
        expected = min(k, eligible_count)
        valid = first.candidate_valid[:, row]
        assert torch.all(first.candidate_masks[valid, row].sum(dim=-1) == expected)


def test_within_action_statistics_match_direct_unordered_pairs():
    conflict = torch.tensor(
        [[[0.0, 0.2, 0.4], [0.2, 0.0, 0.6], [0.4, 0.6, 0.0]]]
    )
    actions = torch.tensor([[[True, True, True]], [[True, False, True]]])
    means, maxima = within_action_conflict_statistics(conflict, actions)
    torch.testing.assert_close(means[:, 0], torch.tensor([0.4, 0.4]))
    torch.testing.assert_close(maxima[:, 0], torch.tensor([0.6, 0.4]))
