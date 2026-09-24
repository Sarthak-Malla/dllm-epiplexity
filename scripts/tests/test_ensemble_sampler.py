"""Run: python -m pytest /home/sarthak.malla/dllm-learning-decoding-path/scripts/tests/test_ensemble_sampler.py

Activate the dllm conda environment first. All tests run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from dllm.core.samplers import (
    EnsembleSampler,
    EnsembleSamplerConfig,
    MDLMSampler,
    MDLMSamplerConfig,
)


@pytest.fixture
def sampler():
    return EnsembleSampler(
        model=None,
        tokenizer=SimpleNamespace(mask_token_id=4, bos_token_id=0, eos_token_id=3),
    )


@pytest.mark.parametrize("strategy", ["low_confidence", "random"])
@pytest.mark.parametrize("sampler_class", [MDLMSampler, EnsembleSampler])
def test_base_scores(sampler_class, strategy):
    sampler = sampler_class(model=None, tokenizer=None)
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]])
    predicted_ids = torch.tensor([[0, 2]])  # Neither proposal is the argmax.
    with torch.random.fork_rng():
        torch.manual_seed(42)
        expected = (
            torch.tensor([[0.09003057, 0.24472847]])
            if strategy == "low_confidence"
            else torch.rand(1, 2)
        )
        torch.manual_seed(42)
        actual = sampler._score_positions(logits, predicted_ids, strategy)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_entropy_handles_zero_probabilities(sampler, dtype):
    logits = torch.tensor(
        [[[0.0, 0.0, -torch.inf], [0.0, -torch.inf, -torch.inf]]], dtype=dtype
    )
    scores = sampler._score_positions(logits, logits.argmax(-1), "min_entropy")
    torch.testing.assert_close(scores, torch.tensor([[-0.69314718, 0.0]]))
    assert scores.argmax(-1).item() == 1


def test_top_two_is_probability_gap(sampler):
    p = torch.tensor([[[0.49, 0.48, 0.03], [0.7, 0.2, 0.1]]])
    predicted_ids = torch.tensor([[2, 2]])  # Top-two margin ignores the proposal.
    scores = sampler._score_positions(p.log(), predicted_ids, "max_top2_prob")
    torch.testing.assert_close(scores, torch.tensor([[0.01, 0.5]]))
    assert scores.argmax(-1).item() == 1


def test_unknown_strategy(sampler):
    with pytest.raises(ValueError, match="Unknown strategy: missing"):
        sampler._score_positions(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), "missing"
        )


@pytest.mark.parametrize("method", ["sample", "infill"])
@pytest.mark.parametrize("noisy_predictions", [False, True])
@pytest.mark.parametrize(
    "strategy, greedy_first_position, noisy_first_position",
    [("low_confidence", 2, 1), ("min_entropy", 1, 1), ("max_top2_prob", 2, 2)],
)
def test_decoding_reveal_order_and_eligibility(
    sampler,
    monkeypatch,
    method,
    noisy_predictions,
    strategy,
    greedy_first_position,
    noisy_first_position,
):
    class FixedModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask):
            # Position 1 has lower entropy, while position 2 has the largest
            # maximum token probability and the largest top-two gap.
            p = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0, 0.0],
                    [0.6, 0.2, 0.1, 0.1, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            return SimpleNamespace(logits=p[: input_ids.shape[1]].log().unsqueeze(0))

    if noisy_predictions:

        def fixed_noise(logits, temperature):
            assert temperature == 1.0
            # Propose token 1: its probability is 0.5 at position 1 and 0.2 at
            # position 2. Only low_confidence should reverse its reveal order.
            return logits.roll(1, dims=-1)

        monkeypatch.setattr("dllm.core.samplers.ensemble.add_gumbel_noise", fixed_noise)

    sampler.model = FixedModel()
    config = EnsembleSamplerConfig(
        max_new_tokens=3,
        block_size=2,
        strategies=(strategy,),
        candidate_fraction=0.5,
        temperature=1.0 if noisy_predictions else 0.0,
        return_dict=True,
    )
    inputs = [[2]] if method == "sample" else [[2, 4, 4, 1]]
    if method == "infill":
        config.block_size = 4
    output = getattr(sampler, method)(inputs, config)
    first_position = (
        noisy_first_position if noisy_predictions else greedy_first_position
    )
    first = output.histories[1][0]
    assert first[first_position] == (1 if noisy_predictions else 0)
    assert first[3 - first_position] == 4
    for history in output.histories:
        assert history[0, 0] == 2
        if method == "infill":
            assert history[0, 3] == 1
    if method == "sample":
        assert first[3] == 4  # The later block cannot be revealed yet.
    assert not (output.sequences == 4).any()


@pytest.mark.parametrize(
    "rankings, policy, expected",
    [
        ([[0, 3, 1, 2], [2, 3, 1, 0]], "candidate_expansion", [3]),
        ([[0, 3, 1, 2], [2, 3, 1, 0]], "majority_voting", [3]),
        ([[0, 1, 2], [0, 2, 1], [1, 2, 0]], "majority_voting", [0]),
        ([[0, 1, 2], [0, 2, 1], [1, 2, 0]], "candidate_expansion", [0, 1, 2]),
        # Two votes out of four is a tie, so expand until position 1 wins.
        ([[0, 1, 2], [0, 1, 2], [1, 2, 0], [1, 2, 0]], "majority_voting", [1]),
        # Empty initial majority: expansion admits all three positions at k=2.
        ([[0, 1, 2], [1, 2, 0], [2, 0, 1]], "majority_voting", [0, 1, 2]),
    ],
)
def test_agreement_and_expansion(sampler, rankings, policy, expected):
    order = torch.tensor(rankings)
    scores = torch.empty_like(order, dtype=torch.float32)
    scores.scatter_(
        1, order, torch.arange(order.shape[1], 0, -1).float().expand_as(scores)
    )
    eligible = torch.ones(1, order.shape[1], dtype=torch.bool)
    selected = sampler._select_positions(
        scores[:, None, :], eligible, policy, 0.25,
        strategies=tuple(f"strategy_{i}" for i in range(order.shape[0])),
    )
    assert selected[0].nonzero().flatten().tolist() == expected


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_selection_filters_before_ranking_and_skips_finished_rows(sampler, policy):
    scores = torch.tensor(
        [
            [[100.0, 2.0, 1.0, 100.0]],
            [[100.0, 1.0, 2.0, 100.0]],
        ]
    ).expand(-1, 2, -1)
    eligible = torch.tensor([[False, True, True, False], [False] * 4])
    selected = sampler._select_positions(
        scores, eligible, policy, 0.25,
        strategies=("low_confidence", "min_entropy"),
    )
    torch.testing.assert_close(selected, eligible)


@pytest.mark.parametrize(
    "rankings, expected",
    [
        ([[0, 1, 2, 3], [0, 2, 1, 3], [2, 3, 0, 1]], [0, 1, 0, 1]),
        ([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]], [1] * 4),
        ([[0, 1, 2, 3]] * 3, [2] * 4),
    ],
)
def test_position_overlap_measures_all_and_pairs_in_configured_order(
    sampler, rankings, expected
):
    order = torch.tensor(rankings)
    scores = torch.empty_like(order, dtype=torch.float32)
    scores.scatter_(1, order, torch.arange(4, 0, -1).float().expand_as(scores))
    strategies = ("max_top2_prob", "low_confidence", "min_entropy")
    metrics = {}
    selected = sampler._select_positions(
        scores[:, None, :], torch.ones(1, 4, dtype=torch.bool),
        "majority_voting", 0.5,
        strategies=strategies, metrics=metrics,
    )
    names = [
        "all", "max_top2_prob__low_confidence",
        "max_top2_prob__min_entropy", "low_confidence__min_entropy",
    ]
    assert metrics["position_overlap"] == {
        f"{stage}/{name}": [value]
        for stage in ("before_expansion", "after_expansion")
        for name, value in zip(names, expected)
    }
    assert metrics["position_proposals"] == {
        f"{stage}/{strategy}": [2]
        for stage in ("before_expansion", "after_expansion")
        for strategy in strategies
    }
    assert metrics["expanded_sequences"] == 0
    assert selected.any()  # Majority can commit even with no all-way overlap.


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_position_overlap_retains_zero_before_fallback(sampler, policy):
    scores = torch.tensor([[[2.0, 1.0]], [[1.0, 2.0]]])
    metrics = {}
    eligible = torch.ones(1, 2, dtype=torch.bool)
    selected = sampler._select_positions(
        scores, eligible, policy, 0.1,
        strategies=("low_confidence", "min_entropy"), metrics=metrics,
    )
    assert metrics == {
        "expanded_sequences": 1,
        "position_overlap": {
            "before_expansion/all": [0],
            "before_expansion/low_confidence__min_entropy": [0],
            "after_expansion/all": [2],
            "after_expansion/low_confidence__min_entropy": [2],
        },
        "position_proposals": {
            "before_expansion/low_confidence": [1],
            "before_expansion/min_entropy": [1],
            "after_expansion/low_confidence": [2],
            "after_expansion/min_entropy": [2],
        },
    }
    torch.testing.assert_close(selected, eligible)


@pytest.mark.parametrize(
    "policy, after_overlap, after_proposals",
    [
        ("majority_voting", [[0, 0], [1, 1], [0, 0], [1, 0]], [2, 1]),
        ("candidate_expansion", [[2, 2], [3, 2], [2, 2], [2, 2]], [3, 2]),
    ],
)
def test_position_counts_use_each_active_rows_eligible_candidates(
    sampler, policy, after_overlap, after_proposals
):
    # Row zero proposes two of four eligible positions. Row one proposes one
    # of its two remaining masks. Row two is already complete.
    scores = torch.tensor([
        [[4.0, 3.0, 2.0, 1.0, 100.0, 100.0]] * 3,
        [[4.0, 2.0, 3.0, 1.0, 100.0, 100.0]] * 3,
        [[2.0, 1.0, 4.0, 3.0, 100.0, 100.0]] * 3,
    ])
    eligible = torch.tensor([
        [True, True, True, True, False, False],
        [False, True, False, True, False, False],
        [False] * 6,
    ])
    metrics = {}
    strategies = ("low_confidence", "min_entropy", "max_top2_prob")
    selected = sampler._select_positions(
        scores, eligible, policy, 0.5,
        strategies=strategies,
        metrics=metrics,
    )
    names = [
        "all", "low_confidence__min_entropy", "low_confidence__max_top2_prob",
        "min_entropy__max_top2_prob",
    ]
    for stage, overlaps, proposals in (
        ("before_expansion", [[0, 0], [1, 1], [0, 0], [1, 0]], [2, 1]),
        ("after_expansion", after_overlap, after_proposals),
    ):
        for name, counts in zip(names, overlaps):
            assert metrics["position_overlap"][f"{stage}/{name}"] == counts
        for strategy in strategies:
            assert metrics["position_proposals"][f"{stage}/{strategy}"] == proposals
    assert metrics["expanded_sequences"] == (
        2 if policy == "candidate_expansion" else 0
    )
    assert not selected[~eligible].any()


def test_single_strategy_overlap_is_full_agreement_without_pairs(sampler):
    metrics = {}
    sampler._select_positions(
        torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]),
        torch.ones(1, 4, dtype=torch.bool), "candidate_expansion", 0.5,
        strategies=("min_entropy",), metrics=metrics,
    )
    assert metrics["position_overlap"] == {
        "before_expansion/all": [2],
        "after_expansion/all": [2],
    }
    assert metrics["position_proposals"] == {
        "before_expansion/min_entropy": [2],
        "after_expansion/min_entropy": [2],
    }


class FixedModel:
    """Return fixed per-position logits and record each model call on CPU."""

    device = torch.device("cpu")

    def __init__(self, probabilities=None):
        self.probabilities = probabilities
        self.calls = []

    def __call__(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        if self.probabilities is None:
            # A mask is the model's favorite token; decoding must suppress it.
            p = torch.tensor([0.2, 0.1, 0.05, 0.05, 0.6])
            logits = p.log().expand(*input_ids.shape, -1).clone()
        else:
            logits = self.probabilities.log().unsqueeze(0).expand(
                input_ids.shape[0], -1, -1
            ).clone()
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("method", ["sample", "infill"])
@pytest.mark.parametrize(
    "policy, expected_first, expected_calls",
    [
        ("candidate_expansion", [False, True, True], 1),
        ("majority_voting", [False, False, True], 2),
    ],
)
def test_policies_control_commit_counts_without_scheduler(
    sampler, method, policy, expected_first, expected_calls
):
    class ForbiddenScheduler:
        def reverse_mask_prob(self, *args, **kwargs):
            pytest.fail("Ensemble decoding must not use the scheduler")

    assert sampler.scheduler is None
    sampler.scheduler = ForbiddenScheduler()
    sampler.model = FixedModel(
        torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.1, 0.0],
        ])
    )
    config = EnsembleSamplerConfig(max_new_tokens=2, block_size=3, return_dict=True)
    inputs = [[2]] if method == "sample" else [[2, 4, 4]]
    # Exercise inference-time overrides without mutating the config.
    output = getattr(sampler, method)(inputs, config, ensemble_policy=policy)
    changed = output.histories[1][0] != output.histories[0][0]
    assert changed.tolist() == expected_first
    assert len(sampler.model.calls) == expected_calls
    assert len(output.histories) == expected_calls + 1
    assert config.ensemble_policy == "all"
    torch.testing.assert_close(output.sequences, torch.tensor([[2, 0, 0]]))


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_sample_preserves_prompts_padding_and_block_order(sampler, policy):
    sampler.model = FixedModel()
    prompts = [[4, 2], [1]]  # Even an existing mask in a prompt stays fixed.
    config = EnsembleSamplerConfig(
        max_new_tokens=5, block_size=2, ensemble_policy=policy, return_dict=True
    )
    output = sampler.sample(prompts, config)
    expected = torch.tensor([[4, 2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 3]])
    torch.testing.assert_close(output.sequences, expected)
    assert len(sampler.model.calls) <= 5
    for x, attention_mask in sampler.model.calls:
        torch.testing.assert_close(
            attention_mask, torch.tensor([[1] * 7, [1] * 6 + [0]])
        )
    for previous, current in zip(output.histories, output.histories[1:]):
        for j, prompt in enumerate(prompts):
            pl = len(prompt)
            assert current[j, :pl].tolist() == prompt
            assert current[j, pl + 5:].tolist() == previous[j, pl + 5:].tolist()
            changed = (current[j] != previous[j]).nonzero().flatten()
            assert changed.numel() > 0
            current_block = int((changed[0] - pl) // 2)
            assert ((changed - pl) // 2 == current_block).all()
            assert not (previous[j, pl:pl + current_block * 2] == 4).any()


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_infill_handles_different_mask_counts_and_empty_blocks(sampler, policy):
    sampler.model = FixedModel()
    output = sampler.infill(
        [[1, 2, 4, 4, 2, 4], [1, 4], [2]],
        EnsembleSamplerConfig(block_size=2, ensemble_policy=policy, return_dict=True),
    )
    torch.testing.assert_close(
        output.sequences,
        torch.tensor([[1, 2, 0, 0, 2, 0], [1, 0, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3]]),
    )
    for previous, current in zip(output.histories, output.histories[1:]):
        changed = previous != current
        assert (previous[changed] == 4).all()
        assert changed.any()
    assert len(sampler.model.calls) <= 4


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_candidates_shrink_with_remaining_masks_and_reset_per_block(sampler, policy):
    # Identical distributions make all strategies agree, exposing proposal sizes
    # through actual reveals without overriding position selection.
    sampler.model = FixedModel()
    prompts = [[1], [1, 2]]
    output = sampler.sample(
        prompts, max_new_tokens=133, block_size=64,
        ensemble_policy=policy, return_dict=True,
    )
    full_block_counts = [7, 6, 6, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2] + [1] * 9
    expected_counts = full_block_counts * 2 + [1] * 5
    assert len(output.histories) == len(expected_counts) + 1
    for step, (previous, current) in enumerate(
        zip(output.histories, output.histories[1:])
    ):
        assert (current != previous).sum(dim=-1).tolist() == [expected_counts[step]] * 2
        block = min(step // len(full_block_counts), 2)
        for row, prompt in enumerate(prompts):
            start = len(prompt) + block * 64
            end = min(start + 64, len(prompt) + 133)
            assert current[row, :start].tolist() == previous[row, :start].tolist()
            assert current[row, end:].tolist() == previous[row, end:].tolist()
    assert not (output.sequences == 4).any()


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_infill_candidates_use_each_rows_remaining_masks(sampler, policy):
    sampler.model = FixedModel()
    inputs = [[1] * 4 + [4] * 12 + [4, 1, 4], [2] * 12 + [4] * 4]
    output = sampler.infill(
        inputs, block_size=16, ensemble_policy=policy, return_dict=True,
    )
    changes = [
        (current != previous).sum(dim=-1).tolist()
        for previous, current in zip(output.histories, output.histories[1:])
    ]
    first_block_changes = [[2, 1]] + [[1, 1]] * 3 + [[1, 0]] * 7
    assert changes == first_block_changes + [[1, 0]] * 2
    for history in output.histories:
        known = output.histories[0] != 4
        torch.testing.assert_close(history[known], output.histories[0][known])
    # The next block remains masked until the current block is finished.
    assert output.histories[len(first_block_changes)][0, 16:].tolist() == [4, 1, 4]
    assert not (output.sequences == 4).any()


def test_candidate_fraction_override_uses_remaining_masks(sampler):
    sampler.model = FixedModel()
    output = sampler.sample(
        [[1]], max_new_tokens=20, block_size=20,
        ensemble_policy="candidate_expansion",
        candidate_fraction=0.25, return_dict=True,
    )
    assert [
        (current != previous).sum().item()
        for previous, current in zip(output.histories, output.histories[1:])
    ] == [5, 4, 3, 2, 2, 1, 1, 1, 1]


def test_length_options_raw_output_and_no_work(sampler):
    sampler.model = FixedModel()
    output = sampler.sample([[1, 2]], max_new_tokens=0, max_length=4)
    torch.testing.assert_close(output, torch.tensor([[1, 2, 0, 0]]))
    sampler.model.calls.clear()
    torch.testing.assert_close(
        sampler.sample([[1]], max_new_tokens=0), torch.tensor([[1]])
    )
    output = sampler.infill([[1, 2]], return_dict=True, block_size=None)
    assert len(output.histories) == 1
    assert not sampler.model.calls


def test_right_shift_empty_prompt_and_suppression(sampler):
    sampler.model = FixedModel(torch.tensor([
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    ]))
    output = sampler.sample(
        [torch.tensor([], dtype=torch.long)], max_new_tokens=2,
        right_shift_logits=True, candidate_fraction=1.0,
    )
    torch.testing.assert_close(output, torch.tensor([[0, 1, 2]]))
    sampler.model = FixedModel()
    output = sampler.sample([[2]], max_new_tokens=1, suppress_tokens=[0])
    torch.testing.assert_close(output, torch.tensor([[2, 1]]))


def test_cfg_preserves_keep_tokens_and_guides_predictions(sampler):
    class GuidedModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask):
            torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 4], [1, 4, 4]]))
            assert (attention_mask == 1).all()
            logits = torch.tensor([
                [0.0, 1.0, -10.0, -10.0, -10.0],
                [-1.0, 2.0, -10.0, -10.0, -10.0],
            ])[:, None, :].expand(-1, 3, -1).clone()
            return SimpleNamespace(logits=logits)

    sampler.model = GuidedModel()
    output = sampler.sample(
        [[1, 2]], max_new_tokens=1, cfg_scale=1.0, cfg_keep_tokens=[1]
    )
    torch.testing.assert_close(output, torch.tensor([[1, 2, 0]]))


@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_all_strategies_share_noisy_predictions(sampler, monkeypatch, policy):
    sampler.model = FixedModel()
    proposals = []
    scored_ids = []
    score_positions = sampler._score_positions

    def fixed_noise(logits, temperature):
        assert temperature == 1.0
        noisy = logits.clone()
        noisy[..., 1] = 10.0  # Force a shared non-argmax proposal.
        proposals.append(noisy.argmax(-1))
        return noisy

    def record_scores(logits, predicted_ids, strategy):
        scored_ids.append(predicted_ids.clone())
        return score_positions(logits, predicted_ids, strategy)

    monkeypatch.setattr("dllm.core.samplers.ensemble.add_gumbel_noise", fixed_noise)
    monkeypatch.setattr(sampler, "_score_positions", record_scores)
    output = sampler.sample(
        [[2]], max_new_tokens=3, temperature=1.0, ensemble_policy=policy
    )
    torch.testing.assert_close(output, torch.tensor([[2, 1, 1, 1]]))
    assert len(proposals) == len(sampler.model.calls)
    assert len(scored_ids) == 3 * len(proposals)
    for i, proposal in enumerate(proposals):
        for predicted_ids in scored_ids[3 * i:3 * (i + 1)]:
            torch.testing.assert_close(predicted_ids, proposal)


def test_begin_suppression_affects_scores_after_prediction(sampler, monkeypatch):
    sampler.model = FixedModel()
    score_positions = sampler._score_positions

    def record_scores(logits, predicted_ids, strategy):
        assert predicted_ids[0, 1] == 0
        assert torch.isneginf(logits[..., 0]).all()
        scores = score_positions(logits, predicted_ids, strategy)
        if strategy == "low_confidence":
            assert scores[0, 1] == 0.0
        return scores

    monkeypatch.setattr(sampler, "_score_positions", record_scores)
    output = sampler.sample([[2]], max_new_tokens=1, begin_suppress_tokens=[0])
    torch.testing.assert_close(output, torch.tensor([[2, 0]]))


def test_invalid_mask_prediction_cannot_loop_forever(sampler, monkeypatch):
    sampler.model = FixedModel()

    def invalid_noise(logits, temperature):
        noisy = torch.zeros_like(logits)
        noisy[..., sampler.tokenizer.mask_token_id] = 1.0
        return noisy

    monkeypatch.setattr("dllm.core.samplers.ensemble.add_gumbel_noise", invalid_noise)
    with pytest.raises(ValueError, match="mask instead of a reveal"):
        sampler.sample([[2]], max_new_tokens=1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"strategies": ()}, "nonempty"),
        ({"strategies": "low_confidence"}, "nonempty"),
        ({"strategies": ("low_confidence", "low_confidence")}, "unique"),
        ({"strategies": ("missing",)}, "Unknown strategy"),
        ({"ensemble_policy": "missing"}, "Unknown ensemble policy"),
        ({"ensemble_policy": "candidate_expansion", "candidate_fraction": 0.0}, "candidate_fraction"),
        ({"ensemble_policy": "majority_voting", "candidate_fraction": 1.1}, "candidate_fraction"),
        ({"block_size": 0}, "block_size"),
        ({"suppress_tokens": [0, 1, 2, 3]}, "No finite token logits"),
    ],
)
def test_invalid_config_and_fully_suppressed_predictions(sampler, kwargs, message):
    sampler.model = FixedModel()
    with pytest.raises(ValueError, match=message):
        sampler.sample([[1]], max_new_tokens=1, **kwargs)


@pytest.mark.parametrize("method", ["sample", "infill"])
@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_ensemble_step_callback_counts_agreement_and_expansion(method, policy):
    events = []
    sampler = EnsembleSampler(
        model=FixedModel(torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.1, 0.0],
        ])),
        tokenizer=SimpleNamespace(mask_token_id=4, bos_token_id=0, eos_token_id=3),
        step_callback=events.append,
    )
    # Infill's second row is already complete and must not inflate row counts.
    inputs = [[2], [1]] if method == "sample" else [[2, 4, 4], [1, 2]]
    active = 2 if method == "sample" else 1
    config = EnsembleSamplerConfig(
        max_new_tokens=2, block_size=3, ensemble_policy=policy, return_dict=True
    )
    output = getattr(sampler, method)(inputs, config)
    if policy == "candidate_expansion":
        expected = [{
            "tokens_committed": 2 * active,
            "remaining_masks": 0,
            "active_sequences": active,
            "expanded_sequences": active,
        }]
    else:
        expected = [
            {
                "tokens_committed": active,
                "remaining_masks": remaining,
                "active_sequences": active,
                "expanded_sequences": 0,
            }
            for remaining in (active, 0)
        ]
    assert [
        {
            key: value for key, value in event.items()
            if key not in {"position_overlap", "position_proposals"}
        }
        for event in events
    ] == expected
    assert len(events) == len(sampler.model.calls)
    assert all(
        type(value) is int
        for event in events
        for key, value in event.items()
        if key not in {"position_overlap", "position_proposals"}
    )
    for event in events:
        for metric in ("position_overlap", "position_proposals"):
            for counts in event[metric].values():
                assert len(counts) == active
                assert all(type(value) is int for value in counts)

    sampler.step_callback = None
    without_metrics = getattr(sampler, method)(inputs, config)
    assert len(events) == len(expected)
    for observed, unobserved in zip(output.histories, without_metrics.histories):
        torch.testing.assert_close(observed, unobserved)
    assert len(output.histories) == len(without_metrics.histories)


@pytest.mark.parametrize("method", ["sample", "infill"])
@pytest.mark.parametrize("policy", ["candidate_expansion", "majority_voting"])
def test_overlap_observer_preserves_noisy_decoding_and_rng(sampler, method, policy):
    sampler.model = FixedModel()
    inputs = [[1], [1, 2]] if method == "sample" else [[1, 4, 4, 4], [4, 2]]
    config = EnsembleSamplerConfig(
        max_new_tokens=5, block_size=2, ensemble_policy=policy,
        temperature=0.75, return_dict=True,
    )
    events = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        sampler.step_callback = events.append
        observed = getattr(sampler, method)(inputs, config)
        observed_rng = torch.random.get_rng_state()
        torch.manual_seed(42)
        sampler.step_callback = None
        unobserved = getattr(sampler, method)(inputs, config)
        unobserved_rng = torch.random.get_rng_state()
    torch.testing.assert_close(observed_rng, unobserved_rng)
    assert len(observed.histories) == len(unobserved.histories)
    for observed_step, unobserved_step in zip(observed.histories, unobserved.histories):
        torch.testing.assert_close(observed_step, unobserved_step)
    assert len(events) == len(observed.histories) - 1
    assert all(
        len(counts) == event["active_sequences"]
        for event in events
        for metric in ("position_overlap", "position_proposals")
        for counts in event[metric].values()
    )


@pytest.mark.parametrize("method", ["sample", "infill"])
def test_mdlm_step_callback_counts_current_block_and_actual_reveals(method):
    events = []
    sampler = MDLMSampler(
        model=FixedModel(),
        tokenizer=SimpleNamespace(mask_token_id=4, bos_token_id=0, eos_token_id=3),
        step_callback=events.append,
    )
    config = MDLMSamplerConfig(
        max_new_tokens=3,
        block_size=2,
        steps=4 if method == "sample" else 6,
        suppress_tokens=[4],
        return_dict=True,
    )
    if method == "sample":
        inputs = [[1], [1, 2]]
        expected = [(2, 2, 2), (2, 0, 2), (2, 0, 2)]
    else:
        inputs = [[1, 4, 4, 4, 4], [1, 4], [1, 2]]
        expected = [(2, 0, 2), (1, 1, 1), (1, 0, 1), (1, 0, 1)]
    output = getattr(sampler, method)(inputs, config)
    assert events == [
        {
            "tokens_committed": committed,
            "remaining_masks": remaining,
            "active_sequences": active,
        }
        for committed, remaining, active in expected
    ]
    assert len(events) == len(sampler.model.calls)
    assert all(type(value) is int for event in events for value in event.values())

    sampler.step_callback = None
    without_metrics = getattr(sampler, method)(inputs, config)
    assert len(events) == len(expected)
    for observed, unobserved in zip(output.histories, without_metrics.histories):
        torch.testing.assert_close(observed, unobserved)
    assert len(output.histories) == len(without_metrics.histories)


@pytest.mark.parametrize("method", ["sample", "infill"])
def test_mdlm_step_callback_does_not_count_mask_predictions_as_reveals(method):
    events = []
    sampler = MDLMSampler(
        model=FixedModel(),
        tokenizer=SimpleNamespace(mask_token_id=4, bos_token_id=0, eos_token_id=3),
        step_callback=events.append,
    )
    inputs = [[1]] if method == "sample" else [[1, 4]]
    output = getattr(sampler, method)(
        inputs, MDLMSamplerConfig(max_new_tokens=1, block_size=2, steps=1)
    )
    assert events == [{
        "tokens_committed": 0,
        "remaining_masks": 1,
        "active_sequences": 1,
    }]
    torch.testing.assert_close(output, torch.tensor([[1, 4]]))


def test_ensemble_no_masks_produces_no_step_events(sampler):
    sampler.model = FixedModel()
    events = []
    sampler.step_callback = events.append
    sampler.sample([[1]], max_new_tokens=0)
    sampler.infill([[1, 2]], block_size=None)
    assert events == []
    assert not sampler.model.calls


@pytest.mark.parametrize("k, expected", [(0, []), (1, [0, 1]), (2, [0, 1, 2]), (9, [0, 1, 2, 3])])
def test_all_commits_unique_proposals_without_expansion(sampler, k, expected):
    scores = torch.tensor([
        [[4., 3., 2., 1., 100.]],
        [[4., 2., 3., 1., 100.]],
        [[3., 4., 2., 1., 100.]],
    ])
    metrics = {}
    selected = sampler._select_positions(
        scores, torch.tensor([[True, True, True, True, False]]),
        "all", None, ("low_confidence", "min_entropy", "max_top2_prob"),
        metrics=metrics, proposal_k=torch.tensor([k]),
    )
    assert selected[0].nonzero().flatten().tolist() == expected
    assert metrics["expanded_sequences"] == 0
    assert all(counts == [min(k, 4)] for counts in metrics["position_proposals"].values())


@pytest.mark.parametrize("method", ["sample", "infill"])
@pytest.mark.parametrize("stochastic", [False, True])
def test_all_uses_schedule_once_per_block_and_finishes_early(
    sampler, monkeypatch, method, stochastic
):
    sampler.model = FixedModel()
    calls = []
    scheduler = object()
    sampler.scheduler = scheduler

    def schedule(mask_index, steps, scheduler, stochastic):
        calls.append((mask_index.sum(-1).tolist(), steps, scheduler, stochastic))
        # The first block has five masks, the partial final block has two.
        return torch.tensor([[2, 3] if mask_index.sum() == 5 else [1, 1]])

    def scores(logits, predicted_ids, strategy):
        values = torch.arange(logits.shape[1], dtype=torch.float).expand(logits.shape[:2])
        return -values if strategy == "low_confidence" else values

    monkeypatch.setattr("dllm.core.samplers.ensemble.get_num_transfer_tokens", schedule)
    monkeypatch.setattr(sampler, "_score_positions", scores)
    inputs = [[1]] if method == "sample" else [[4] * 7]
    output = getattr(sampler, method)(
        inputs, ensemble_policy="all", steps=4, block_size=5, max_new_tokens=7,
        strategies=("low_confidence", "min_entropy"), candidate_fraction=None,
        stochastic_transfer=stochastic, return_dict=True,
    )
    assert calls == [([5], 2, scheduler, stochastic), ([2], 2, scheduler, stochastic)]
    assert [(b != a).sum().item() for a, b in zip(output.histories, output.histories[1:])] == [4, 1, 2]
    assert not (output.sequences == 4).any()


@pytest.mark.parametrize("stochastic", [False, True])
def test_all_real_schedule_handles_unequal_rows(sampler, stochastic):
    sampler.model = FixedModel()
    output = sampler.infill(
        [[1, 4, 4, 4, 4, 4, 4], [2, 4], [1]],
        ensemble_policy="all", steps=4, block_size=4,
        stochastic_transfer=stochastic, candidate_fraction=0,
    )
    torch.testing.assert_close(output, torch.tensor([
        [1, 0, 0, 0, 0, 0, 0], [2, 0, 3, 3, 3, 3, 3], [1, 3, 3, 3, 3, 3, 3],
    ]))


def test_all_requires_positive_steps(sampler):
    sampler.model = FixedModel()
    with pytest.raises(ValueError, match="steps must be positive"):
        sampler.sample([[1]], steps=0)
