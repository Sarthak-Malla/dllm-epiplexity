"""Test model-independent dependency-guided candidate generation.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_candidates.py -v
"""

from dataclasses import replace

import pytest
import torch

from dllm.core.samplers.candidates import (
    BASELINE_ONLY_PROPOSALS,
    compose_principled_dependency_candidates,
    deduplicate_and_refill_k1_candidates,
    dependency_anchor_scores,
    generate_current_mixed_candidates,
    generate_dependency_gumbel_top_n,
    generate_dependency_top_n,
    generate_high_entropy_candidates,
    generate_position_confidence_gumbel_candidates,
    generate_random_candidates,
    generate_spaced_candidates,
    generate_top_confidence_candidates,
)
from dllm.core.samplers.entropy_drop import EntropyDropSampler


def test_asymmetric_direction_and_diagonal_are_explicit():
    dependency = torch.tensor(
        [
            [
                [99.0, 1.0, 2.0],
                [3.0, 99.0, 4.0],
                [5.0, 6.0, 99.0],
            ]
        ]
    )
    entropy = torch.tensor([[1.0, 2.0, 3.0]])
    eligible = torch.ones((1, 3), dtype=torch.bool)

    outgoing = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        direction="outgoing",
    )
    incoming = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        direction="incoming",
    )
    symmetric = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        direction="symmetric",
    )

    torch.testing.assert_close(outgoing, torch.tensor([[8.0, 15.0, 17.0]]))
    torch.testing.assert_close(incoming, torch.tensor([[21.0, 19.0, 10.0]]))
    torch.testing.assert_close(symmetric, torch.tensor([[14.5, 17.0, 13.5]]))


def test_top_n_is_distinct_tie_stable_and_marks_short_rows_invalid():
    dependency = torch.zeros((2, 5, 5), dtype=torch.float32)
    entropy = torch.ones((2, 5), dtype=torch.float32)
    eligible = torch.tensor(
        [
            [False, True, True, False, True],
            [False, False, True, False, False],
        ]
    )

    batch = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=4,
        generation_seed=42,
    )

    assert tuple(batch.shape) == (3, 2, 5)
    assert batch.names == ("dependency_0", "dependency_1", "dependency_2")
    assert batch.candidate_valid.tolist() == [
        [True, True],
        [True, False],
        [True, False],
    ]
    assert batch.seed_anchors.tolist() == [[1, 2], [2, -1], [4, -1]]
    assert batch.selected_positions.tolist() == [
        [[1], [2]],
        [[2], [-1]],
        [[4], [-1]],
    ]
    assert batch.candidate_masks[:, 0].sum(dim=-1).tolist() == [1, 1, 1]
    assert batch.candidate_masks[:, 1].sum(dim=-1).tolist() == [1, 0, 0]
    assert not torch.any(batch.candidate_masks & ~eligible.unsqueeze(0))
    assert batch.generation_seed == 42
    assert batch.configuration["direction"] == "outgoing"
    assert [row["rank"] for row in batch.metadata] == [0, 1, 2]
    assert list(batch.as_legacy_dict()) == list(batch.names)


def test_working_score_generates_exact_expected_masks_without_prompt_selection():
    dependency = torch.zeros((1, 5, 5), dtype=torch.float32)
    dependency[0, 1, 2] = 1.0
    dependency[0, 2, 1] = 2.0
    dependency[0, 4, 1] = 3.0
    entropy = torch.tensor([[100.0, 2.0, 5.0, 100.0, 1.0]])
    eligible = torch.tensor([[False, True, True, False, True]])

    batch = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=2,
    )

    # Outgoing scores at positions 1, 2, 4 are 5, 4, and 6.
    assert batch.seed_anchors.tolist() == [[4], [1]]
    torch.testing.assert_close(
        batch.proposal_scores,
        torch.tensor([[6.0], [5.0]]),
    )
    expected = torch.tensor(
        [
            [[False, False, False, False, True]],
            [[False, True, False, False, False]],
        ]
    )
    assert torch.equal(batch.candidate_masks, expected)


def test_zero_dependency_and_empty_rows_have_defined_outputs():
    dependency = torch.zeros((2, 4, 4), dtype=torch.float32)
    entropy = torch.zeros((2, 4), dtype=torch.float32)
    eligible = torch.tensor(
        [
            [False, True, False, True],
            [False, False, False, False],
        ]
    )

    scores = dependency_anchor_scores(dependency, entropy, eligible)
    assert scores[0, [1, 3]].tolist() == [0.0, 0.0]
    assert torch.isneginf(scores[0, [0, 2]]).all()
    assert torch.isneginf(scores[1]).all()

    batch = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=8,
    )
    assert tuple(batch.shape) == (2, 2, 4)
    assert batch.seed_anchors.tolist() == [[1, -1], [3, -1]]
    assert batch.candidate_valid.tolist() == [[True, False], [True, False]]
    assert not torch.isnan(batch.proposal_scores).any()

    empty = generate_dependency_top_n(
        dependency[:1],
        entropy[:1],
        torch.zeros((1, 4), dtype=torch.bool),
        candidate_budget=8,
    )
    assert tuple(empty.shape) == (0, 1, 4)
    assert tuple(empty.selected_positions.shape) == (0, 1, 0)
    assert empty.names == ()


def test_candidate_batch_validation_rejects_corrupt_masks_and_names():
    dependency = torch.zeros((1, 3, 3), dtype=torch.float32)
    entropy = torch.ones((1, 3), dtype=torch.float32)
    eligible = torch.tensor([[False, True, True]])
    batch = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=2,
    )

    with pytest.raises(TypeError, match="candidate_masks must be boolean"):
        replace(batch, candidate_masks=batch.candidate_masks.float())

    corrupt = batch.candidate_masks.clone()
    corrupt[0, 0, 0] = True
    with pytest.raises(ValueError, match="inside eligible_mask"):
        replace(batch, candidate_masks=corrupt)

    with pytest.raises(ValueError, match="names must be unique"):
        replace(batch, names=("same", "same"))


def test_positive_confidence_exponent_is_optional_but_validated():
    dependency = torch.zeros((1, 3, 3), dtype=torch.float32)
    dependency[0, 0, 1] = 1.0
    dependency[0, 1, 0] = 1.0
    entropy = torch.ones((1, 3), dtype=torch.float32)
    eligible = torch.tensor([[True, True, False]])

    unweighted = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        confidence_exponent=0.0,
    )
    assert unweighted[0, :2].tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="confidence must have shape"):
        dependency_anchor_scores(
            dependency,
            entropy,
            eligible,
            confidence_exponent=1.0,
        )

    weighted = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        confidence=torch.tensor([[0.25, 0.75, 1.0]]),
        confidence_exponent=1.0,
    )
    assert weighted[0, :2].tolist() == [0.25, 0.75]


def test_zero_temperature_exactly_matches_deterministic_candidate_batch():
    dependency = torch.arange(50, dtype=torch.float32).reshape(2, 5, 5)
    entropy = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    eligible = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, True, False],
        ]
    )
    deterministic = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=3,
        generation_seed=17,
    )
    zero_temperature = generate_dependency_gumbel_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=3,
        position_temperature=0.0,
        generation_seed=17,
    )

    assert zero_temperature.names == deterministic.names
    assert zero_temperature.metadata == deterministic.metadata
    assert zero_temperature.configuration == deterministic.configuration
    assert zero_temperature.generation_seed == deterministic.generation_seed
    for field in (
        "candidate_masks",
        "proposal_scores",
        "seed_anchors",
        "selected_positions",
        "mean_within_set_dependency",
        "candidate_valid",
        "eligible_mask",
        "requested_k",
        "clipped_k",
    ):
        assert torch.equal(
            getattr(zero_temperature, field),
            getattr(deterministic, field),
        )


def test_gumbel_top_n_is_seeded_distinct_and_does_not_mutate_global_rng():
    dependency = torch.zeros((2, 7, 7), dtype=torch.float32)
    entropy = torch.ones((2, 7), dtype=torch.float32)
    eligible = torch.tensor(
        [
            [False, True, True, True, True, True, True],
            [True, False, True, False, True, False, True],
        ]
    )

    torch.manual_seed(123456)
    global_state_before = torch.random.get_rng_state().clone()
    first = generate_dependency_gumbel_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=5,
        position_temperature=1.0,
        generation_seed=42,
    )
    global_state_after = torch.random.get_rng_state()
    second = generate_dependency_gumbel_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=5,
        position_temperature=1.0,
        generation_seed=42,
    )
    another_seed = generate_dependency_gumbel_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=5,
        position_temperature=1.0,
        generation_seed=43,
    )

    assert torch.equal(global_state_before, global_state_after)
    assert torch.equal(first.candidate_masks, second.candidate_masks)
    assert torch.equal(first.seed_anchors, second.seed_anchors)
    assert first.metadata == second.metadata
    assert not torch.equal(first.candidate_masks, another_seed.candidate_masks)
    assert first.configuration["sampling_strategy"] == (
        "gumbel_top_n_without_replacement"
    )
    assert first.configuration["position_temperature"] == 1.0
    assert first.generation_seed == 42
    assert all(row["source"] == "dependency_gumbel" for row in first.metadata)
    assert all("gumbel_noise_by_batch" in row for row in first.metadata)

    for batch_index, valid_count in enumerate(eligible.sum(dim=-1).tolist()):
        anchors = first.seed_anchors[:, batch_index]
        anchors = anchors[anchors >= 0]
        assert anchors.numel() == min(5, valid_count)
        assert torch.unique(anchors).numel() == anchors.numel()
        assert torch.all(eligible[batch_index, anchors])


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_gumbel_temperature_and_seed_are_validated(temperature):
    dependency = torch.zeros((1, 2, 2), dtype=torch.float32)
    entropy = torch.ones((1, 2), dtype=torch.float32)
    eligible = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        generate_dependency_gumbel_top_n(
            dependency,
            entropy,
            eligible,
            candidate_budget=2,
            position_temperature=temperature,
            generation_seed=1,
        )

    with pytest.raises(ValueError, match="nonnegative integer seed"):
        generate_dependency_gumbel_top_n(
            dependency,
            entropy,
            eligible,
            candidate_budget=2,
            position_temperature=1.0,
            generation_seed=None,
        )


def _ordered_candidate_batch(order, *, eligible=None, name_prefix="source"):
    """Build a deterministic batch whose first positions follow ``order``."""
    sequence_length = 5
    if eligible is None:
        eligible = torch.ones((1, sequence_length), dtype=torch.bool)
    dependency = torch.zeros((eligible.shape[0], 5, 5), dtype=torch.float32)
    for batch_index in range(eligible.shape[0]):
        for rank, position in enumerate(order):
            target = (position + 1) % sequence_length
            dependency[batch_index, position, target] = len(order) - rank
    entropy = torch.ones_like(eligible, dtype=torch.float32)
    return generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=len(order),
        name_prefix=name_prefix,
    )


def test_stable_deduplication_and_fallback_order_are_auditable():
    primary = _ordered_candidate_batch([0, 1], name_prefix="primary")
    dependency = _ordered_candidate_batch([1, 2], name_prefix="dependency")
    confidence = _ordered_candidate_batch([2, 3], name_prefix="confidence")

    torch.manual_seed(9876)
    global_state_before = torch.random.get_rng_state().clone()
    combined = deduplicate_and_refill_k1_candidates(
        (primary, primary),
        candidate_budget=5,
        dependency_fallback=dependency,
        confidence_fallback=confidence,
        generation_seed=42,
    )
    global_state_after = torch.random.get_rng_state()

    assert torch.equal(global_state_before, global_state_after)
    assert combined.seed_anchors[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert combined.candidate_valid[:, 0].tolist() == [True] * 5
    assert combined.configuration["duplicates_skipped_by_batch"] == (4,)
    assert [row["source_by_batch"][0] for row in combined.metadata] == [
        "dependency",
        "dependency",
        "dependency_fallback",
        "confidence_fallback",
        "random_fallback",
    ]
    assert [row["is_fallback_by_batch"][0] for row in combined.metadata] == [
        False,
        False,
        True,
        True,
        True,
    ]
    flattened = combined.candidate_masks[:, 0].flatten(start_dim=1)
    assert torch.unique(flattened, dim=0).shape[0] == 5


def test_refill_handles_different_row_action_spaces_and_stops_when_exhausted():
    eligible = torch.tensor(
        [
            [False, True, True, False, True],
            [False, False, True, False, False],
        ]
    )
    primary = _ordered_candidate_batch(
        [2],
        eligible=eligible,
        name_prefix="primary",
    )

    combined = deduplicate_and_refill_k1_candidates(
        (primary, primary),
        candidate_budget=4,
        generation_seed=7,
    )
    repeated = deduplicate_and_refill_k1_candidates(
        (primary, primary),
        candidate_budget=4,
        generation_seed=7,
    )
    another_seed = deduplicate_and_refill_k1_candidates(
        (primary, primary),
        candidate_budget=4,
        generation_seed=9,
    )

    assert tuple(combined.shape) == (3, 2, 5)
    assert combined.candidate_valid.sum(dim=0).tolist() == [3, 1]
    assert combined.candidate_masks[:, 0].sum().item() == 3
    assert combined.candidate_masks[:, 1].sum().item() == 1
    assert torch.equal(combined.candidate_masks, repeated.candidate_masks)
    assert torch.equal(combined.seed_anchors, repeated.seed_anchors)
    assert combined.metadata == repeated.metadata
    assert not torch.equal(
        combined.seed_anchors[:, 0],
        another_seed.seed_anchors[:, 0],
    )
    for batch_index in range(2):
        anchors = combined.seed_anchors[:, batch_index]
        anchors = anchors[anchors >= 0]
        assert torch.unique(anchors).numel() == anchors.numel()
        assert torch.all(eligible[batch_index, anchors])
    assert combined.metadata[1]["source_by_batch"][1] is None


def test_refill_rejects_incompatible_candidate_batches():
    primary = _ordered_candidate_batch([0, 1])
    different_mask = _ordered_candidate_batch(
        [1],
        eligible=torch.tensor([[False, True, True, True, True]]),
    )

    with pytest.raises(ValueError, match="share eligible_mask"):
        deduplicate_and_refill_k1_candidates(
            (primary,),
            candidate_budget=3,
            dependency_fallback=different_mask,
            generation_seed=1,
        )


def test_confidence_and_true_entropy_adapters_rank_exact_values():
    eligible = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, True, False],
        ]
    )
    confidence = torch.tensor(
        [
            [0.99, 0.20, 0.80, 0.98, 0.50],
            [0.40, 0.99, 0.70, 0.10, 0.98],
        ]
    )
    entropy = torch.tensor(
        [
            [9.0, 5.0, 1.0, 8.0, 3.0],
            [2.0, 9.0, 1.0, 4.0, 8.0],
        ]
    )

    top_confidence = generate_top_confidence_candidates(
        confidence,
        eligible,
        candidate_budget=3,
    )
    high_entropy = generate_high_entropy_candidates(
        entropy,
        eligible,
        candidate_budget=3,
    )

    assert top_confidence.seed_anchors.tolist() == [[2, 2], [4, 0], [1, 3]]
    assert high_entropy.seed_anchors.tolist() == [[1, 3], [4, 0], [2, 2]]
    assert all(row["source"] == "top_confidence" for row in top_confidence.metadata)
    assert all(row["source"] == "high_entropy" for row in high_entropy.metadata)
    assert not torch.any(top_confidence.candidate_masks & ~eligible.unsqueeze(0))
    assert not torch.any(high_entropy.candidate_masks & ~eligible.unsqueeze(0))


def test_random_and_spaced_adapters_are_distinct_reproducible_and_row_safe():
    eligible = torch.tensor(
        [
            [False, True, True, False, True, True],
            [True, False, False, True, False, False],
        ]
    )
    torch.manual_seed(2468)
    global_state_before = torch.random.get_rng_state().clone()
    random_first = generate_random_candidates(
        eligible,
        candidate_budget=3,
        generation_seed=21,
    )
    global_state_after = torch.random.get_rng_state()
    random_repeat = generate_random_candidates(
        eligible,
        candidate_budget=3,
        generation_seed=21,
    )
    random_other = generate_random_candidates(
        eligible,
        candidate_budget=3,
        generation_seed=22,
    )
    spaced = generate_spaced_candidates(
        eligible,
        candidate_budget=3,
    )

    assert torch.equal(global_state_before, global_state_after)
    assert torch.equal(random_first.candidate_masks, random_repeat.candidate_masks)
    assert not torch.equal(random_first.candidate_masks, random_other.candidate_masks)
    assert spaced.seed_anchors.tolist() == [[1, 0], [2, 3], [5, -1]]
    assert spaced.candidate_valid.tolist() == [
        [True, True],
        [True, True],
        [True, False],
    ]
    for batch in (random_first, spaced):
        for batch_index in range(eligible.shape[0]):
            anchors = batch.seed_anchors[:, batch_index]
            anchors = anchors[anchors >= 0]
            assert torch.unique(anchors).numel() == anchors.numel()
            assert torch.all(eligible[batch_index, anchors])


def test_position_confidence_gumbel_is_explicit_and_zero_is_exact():
    confidence = torch.tensor([[0.1, 0.7, 0.4, 0.9]])
    eligible = torch.tensor([[False, True, True, True]])
    deterministic = generate_top_confidence_candidates(
        confidence,
        eligible,
        candidate_budget=3,
        generation_seed=5,
        name_prefix="position_confidence_gumbel",
    )
    zero = generate_position_confidence_gumbel_candidates(
        confidence,
        eligible,
        candidate_budget=3,
        position_temperature=0.0,
        generation_seed=5,
    )
    stochastic = generate_position_confidence_gumbel_candidates(
        confidence,
        eligible,
        candidate_budget=3,
        position_temperature=1.0,
        generation_seed=5,
    )

    assert zero.names == deterministic.names
    assert zero.metadata == deterministic.metadata
    assert zero.configuration == deterministic.configuration
    assert torch.equal(zero.candidate_masks, deterministic.candidate_masks)
    assert stochastic.configuration["not_token_logit_gumbel"] is True
    assert stochastic.configuration["position_temperature"] == 1.0
    assert all(
        row["source"] == "position_confidence_gumbel"
        for row in stochastic.metadata
    )


def test_current_mixed_adapter_matches_legacy_k1_masks_on_cpu():
    confidence = torch.tensor(
        [
            [0.99, 0.10, 0.20, 0.98, 0.40, 0.30],
            [0.60, 0.95, 0.70, 0.20, 0.90, 0.85],
        ]
    )
    eligible = torch.tensor(
        [
            [False, True, True, False, True, True],
            [True, False, True, True, False, False],
        ]
    )
    seed = 314
    torch.manual_seed(seed)
    legacy = EntropyDropSampler(
        model=None,
        tokenizer=None,
    ).generate_candidate_sets(
        confidence=confidence,
        mask_idx=eligible,
        num_transfer=1,
        strategy="mixed",
    )
    adapted = generate_current_mixed_candidates(
        confidence,
        eligible,
        generation_seed=seed,
    )

    assert adapted.names == tuple(legacy)
    for candidate_index, name in enumerate(adapted.names):
        assert torch.equal(adapted.candidate_masks[candidate_index], legacy[name])
    assert adapted.configuration["historical_high_entropy_definition"] == (
        "lowest confidence"
    )
    assert adapted.metadata[3]["uses_measured_entropy"] is False


def test_all_baseline_metadata_exactly_matches_candidate_masks():
    eligible = torch.tensor([[False, True, True, False, True]])
    confidence = torch.tensor([[0.9, 0.2, 0.8, 0.7, 0.5]])
    entropy = torch.tensor([[9.0, 1.0, 4.0, 8.0, 2.0]])
    batches = (
        generate_top_confidence_candidates(
            confidence,
            eligible,
            candidate_budget=3,
        ),
        generate_high_entropy_candidates(
            entropy,
            eligible,
            candidate_budget=3,
        ),
        generate_random_candidates(
            eligible,
            candidate_budget=3,
            generation_seed=1,
        ),
        generate_spaced_candidates(
            eligible,
            candidate_budget=3,
        ),
        generate_position_confidence_gumbel_candidates(
            confidence,
            eligible,
            candidate_budget=3,
            position_temperature=0.5,
            generation_seed=1,
        ),
        generate_current_mixed_candidates(
            confidence,
            eligible,
            generation_seed=1,
        ),
    )

    for batch in batches:
        assert len(batch.names) == len(batch.metadata) == batch.shape[0]
        for candidate_index in range(batch.shape[0]):
            if not batch.candidate_valid[candidate_index, 0]:
                continue
            position = int(batch.selected_positions[candidate_index, 0, 0])
            assert batch.seed_anchors[candidate_index, 0].item() == position
            assert batch.candidate_masks[candidate_index, 0, position]
            assert batch.candidate_masks[candidate_index, 0].sum().item() == 1
    assert set(BASELINE_ONLY_PROPOSALS) == {
        "top_confidence",
        "high_entropy",
        "random",
        "spaced",
        "position_confidence_gumbel",
        "current_mixed",
    }


def test_principled_pool_uses_only_dependency_candidates_and_records_zero_signal():
    dependency = torch.zeros((1, 5, 5), dtype=torch.float32)
    entropy = torch.ones((1, 5), dtype=torch.float32)
    eligible = torch.ones((1, 5), dtype=torch.bool)
    deterministic = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=4,
        generation_seed=12,
    )
    diverse = generate_dependency_gumbel_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=1,
        position_temperature=1.0,
        generation_seed=12,
    )
    proposed = compose_principled_dependency_candidates(
        deterministic,
        diverse,
        candidate_budget=4,
        generation_seed=12,
    )

    assert proposed.seed_anchors[0, 0].item() == 0
    assert torch.unique(proposed.seed_anchors[:, 0]).numel() == 4
    sources = {row["source_by_batch"][0] for row in proposed.metadata}
    assert sources <= {
        "dependency",
        "dependency_gumbel",
        "dependency_fallback",
    }
    assert not any(
        source is not None
        and any(name in source for name in ("confidence", "random", "spaced"))
        for source in sources
    )
    assert proposed.configuration["proposal"] == "principled_dependency_pool"
    assert proposed.configuration["unprincipled_fallbacks"] == ()
    assert proposed.configuration["random_fallback_enabled"] is False
    assert proposed.configuration["zero_dependency_signal_by_batch"] == (True,)
    assert proposed.configuration["constant_dependency_signal_by_batch"] == (True,)


def test_principled_pool_refuses_to_hide_an_incomplete_dependency_fallback():
    dependency = torch.zeros((1, 4, 4), dtype=torch.float32)
    entropy = torch.ones((1, 4), dtype=torch.float32)
    eligible = torch.ones((1, 4), dtype=torch.bool)
    incomplete = generate_dependency_top_n(
        dependency,
        entropy,
        eligible,
        candidate_budget=1,
    )

    with pytest.raises(RuntimeError, match="without random"):
        compose_principled_dependency_candidates(
            incomplete,
            None,
            candidate_budget=3,
            generation_seed=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_seeded_baseline_adapters_match_between_cpu_and_cuda():
    eligible_cpu = torch.tensor([[False, True, True, False, True]])
    confidence_cpu = torch.tensor([[0.9, 0.2, 0.8, 0.7, 0.5]])

    random_cpu = generate_random_candidates(
        eligible_cpu,
        candidate_budget=3,
        generation_seed=33,
    )
    random_cuda = generate_random_candidates(
        eligible_cpu.cuda(),
        candidate_budget=3,
        generation_seed=33,
    )
    gumbel_cpu = generate_position_confidence_gumbel_candidates(
        confidence_cpu,
        eligible_cpu,
        candidate_budget=3,
        position_temperature=1.0,
        generation_seed=33,
    )
    gumbel_cuda = generate_position_confidence_gumbel_candidates(
        confidence_cpu.cuda(),
        eligible_cpu.cuda(),
        candidate_budget=3,
        position_temperature=1.0,
        generation_seed=33,
    )

    assert torch.equal(random_cpu.candidate_masks, random_cuda.candidate_masks.cpu())
    assert torch.equal(gumbel_cpu.candidate_masks, gumbel_cuda.candidate_masks.cpu())
