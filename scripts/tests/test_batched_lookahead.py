"""Test candidate-major state expansion and shared batched lookahead scoring.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_batched_lookahead.py" -v
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.samplers.batched_lookahead import (
    candidate_batch_from_mask_mapping,
    evaluate_batched_lookahead,
    expand_candidate_states,
)
from dllm.core.samplers.candidates import generate_top_confidence_candidates
from dllm.core.samplers.counterfactual import (
    decoding_risk_per_token,
    entropy_per_token,
)
from dllm.core.samplers.entropy_drop import EntropyDropSampler
from dllm.core.samplers.risk_reduction import RiskReductionSampler


class ContextModel(nn.Module):
    """Return deterministic per-row logits that react to revealed tokens."""

    def __init__(
        self,
        vocabulary_size: int = 5,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.output_dtype = output_dtype
        self.batch_sizes: list[int] = []
        self.seen_states: list[torch.Tensor] = []
        self.seen_attention_masks: list[torch.Tensor | None] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.batch_sizes.append(input_ids.shape[0])
        self.seen_states.append(input_ids.detach().clone())
        self.seen_attention_masks.append(
            None if attention_mask is None else attention_mask.detach().clone()
        )
        valid = (
            torch.ones_like(input_ids, dtype=torch.float32)
            if attention_mask is None
            else (attention_mask != 0).float()
        )
        context = (input_ids.float() * valid).sum(dim=-1, keepdim=True)
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        centers = torch.remainder(context + 1.3 * positions, self.vocabulary_size)
        vocabulary = torch.arange(
            self.vocabulary_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        logits = -0.37 * (vocabulary - centers.unsqueeze(-1)).square()
        logits += 0.02 * input_ids.float().unsqueeze(-1) * vocabulary
        return SimpleNamespace(logits=logits.to(self.output_dtype))


class StaticModel(nn.Module):
    """Repeat one fixed logit map, making all candidate scores tie."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.batch_sizes: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(
            logits=self.logits.expand(input_ids.shape[0], -1, -1)
        )


def _case(output_dtype: torch.dtype = torch.float32):
    """Build a B=2, N=3 case with one invalid candidate/example pair."""
    input_ids = torch.tensor(
        [
            [7, 9, 9, 3, 9, 0],
            [9, 4, 2, 9, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    active_mask = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    confidence = torch.tensor(
        [
            [0.0, 0.8, 0.3, 0.0, 0.5, 0.0],
            [0.2, 0.0, 0.0, 0.9, 0.0, 0.0],
        ]
    )
    candidates = generate_top_confidence_candidates(
        confidence,
        active_mask,
        candidate_budget=3,
    )
    model = ContextModel(output_dtype=output_dtype)
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    predicted_token_ids = base_logits.argmax(dim=-1)
    model.batch_sizes.clear()
    model.seen_states.clear()
    model.seen_attention_masks.clear()
    return (
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_logits,
    )


def _manual_metric(logits: torch.Tensor, metric: str) -> torch.Tensor:
    """Compute the test reference without calling production metric helpers."""
    if metric == "entropy_drop":
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        probabilities = log_probabilities.exp()
        terms = torch.where(
            probabilities > 0,
            probabilities * log_probabilities,
            torch.zeros_like(probabilities),
        )
        return -terms.sum(dim=-1)
    if metric == "risk_reduction":
        probabilities = F.softmax(logits.float(), dim=-1)
        return 1.0 - probabilities.amax(dim=-1)
    raise AssertionError(metric)


def _sequential_reference(
    model: nn.Module,
    input_ids: torch.Tensor,
    predicted_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    active_mask: torch.Tensor,
    candidates,
    base_metric_map: torch.Tensor,
    metric: str,
):
    """Independently reproduce the old one-forward-per-candidate loop."""
    candidate_count, batch_size, _ = candidates.shape
    scores = torch.full((candidate_count, batch_size), -torch.inf)
    base_sums = torch.zeros_like(scores)
    lookahead_sums = torch.zeros_like(scores)
    best_score = torch.full((batch_size,), -torch.inf)
    best_index = torch.full((batch_size,), -1, dtype=torch.long)
    for candidate_index in range(candidate_count):
        candidate_mask = candidates.candidate_masks[candidate_index]
        candidate_state = input_ids.clone()
        candidate_state[candidate_mask] = predicted_token_ids[candidate_mask]
        logits = model(
            input_ids=candidate_state,
            attention_mask=attention_mask,
        ).logits
        lookahead_metric = _manual_metric(logits, metric)
        heldout = (
            active_mask
            & ~candidate_mask
            & candidates.candidate_valid[candidate_index].unsqueeze(-1)
        )
        base_sum = torch.where(
            heldout,
            base_metric_map.float(),
            torch.zeros_like(base_metric_map.float()),
        ).sum(dim=-1)
        lookahead_sum = torch.where(
            heldout,
            lookahead_metric,
            torch.zeros_like(lookahead_metric),
        ).sum(dim=-1)
        score = base_sum - lookahead_sum
        score = torch.where(
            candidates.candidate_valid[candidate_index],
            score,
            torch.full_like(score, -torch.inf),
        )
        scores[candidate_index] = score
        base_sums[candidate_index] = base_sum
        lookahead_sums[candidate_index] = lookahead_sum
        improved = score > best_score
        best_score = torch.where(improved, score, best_score)
        best_index = torch.where(
            improved,
            torch.full_like(best_index, candidate_index),
            best_index,
        )
    return scores, base_sums, lookahead_sums, best_index, best_score


def test_expand_candidate_states_is_candidate_major_reversible_and_immutable():
    (
        _,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        _,
    ) = _case()
    input_before = input_ids.clone()
    prediction_before = predicted_token_ids.clone()
    masks_before = candidates.candidate_masks.clone()

    expansion = expand_candidate_states(
        input_ids,
        predicted_token_ids,
        candidates,
        attention_mask=attention_mask,
        masked_active_mask=active_mask,
    )

    assert expansion.candidate_states.shape == (3, 2, 6)
    assert expansion.flat_states.shape == (6, 6)
    assert expansion.candidate_indices.tolist() == [0, 0, 1, 1, 2, 2]
    assert expansion.batch_indices.tolist() == [0, 1, 0, 1, 0, 1]
    assert expansion.flat_row(2, 1) == 5
    assert torch.equal(
        expansion.unflatten(expansion.flat_states),
        expansion.candidate_states,
    )
    assert torch.equal(expansion.flat_states[0], expansion.candidate_states[0, 0])
    assert torch.equal(expansion.flat_states[1], expansion.candidate_states[0, 1])
    assert torch.equal(expansion.flat_states[2], expansion.candidate_states[1, 0])
    assert torch.equal(
        expansion.flat_attention_mask,
        attention_mask.repeat(3, 1),
    )

    for candidate_index in range(3):
        for batch_index in range(2):
            expected = input_ids[batch_index].clone()
            candidate_mask = candidates.candidate_masks[candidate_index, batch_index]
            expected[candidate_mask] = predicted_token_ids[batch_index, candidate_mask]
            assert torch.equal(
                expansion.candidate_states[candidate_index, batch_index],
                expected,
            )
            expected_heldout = active_mask[batch_index] & ~candidate_mask
            if not candidates.candidate_valid[candidate_index, batch_index]:
                expected_heldout.zero_()
                assert torch.equal(expected, input_ids[batch_index])
            assert torch.equal(
                expansion.heldout_masks[candidate_index, batch_index],
                expected_heldout,
            )

    assert torch.equal(input_ids, input_before)
    assert torch.equal(predicted_token_ids, prediction_before)
    assert torch.equal(candidates.candidate_masks, masks_before)


def test_expansion_rejects_active_positions_outside_attention():
    (
        _,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        _,
    ) = _case()
    bad_attention = attention_mask.clone()
    bad_attention[0, 1] = 0

    with pytest.raises(ValueError, match="subset of attention_mask"):
        expand_candidate_states(
            input_ids,
            predicted_token_ids,
            candidates,
            attention_mask=bad_attention,
            masked_active_mask=active_mask,
        )


@pytest.mark.parametrize("metric", ["entropy_drop", "risk_reduction"])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_batched_scores_and_selection_match_independent_sequential_reference(
    metric,
    output_dtype,
):
    (
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_logits,
    ) = _case(output_dtype)
    base_metric_map = _manual_metric(base_logits, metric)
    reference = _sequential_reference(
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_metric_map,
        metric,
    )
    model.batch_sizes.clear()

    result = evaluate_batched_lookahead(
        model,
        input_ids,
        predicted_token_ids,
        candidates,
        base_metric_map=base_metric_map,
        metric=metric,
        attention_mask=attention_mask,
        masked_active_mask=active_mask,
    )

    scores, base_sums, lookahead_sums, best_index, best_score = reference
    tolerance = 1e-6 if output_dtype == torch.float32 else 2e-5
    torch.testing.assert_close(result.scores, scores, atol=tolerance, rtol=0)
    torch.testing.assert_close(
        result.base_metric_sums,
        base_sums,
        atol=tolerance,
        rtol=0,
    )
    torch.testing.assert_close(
        result.lookahead_metric_sums,
        lookahead_sums,
        atol=tolerance,
        rtol=0,
    )
    assert torch.equal(result.best_index, best_index)
    torch.testing.assert_close(result.best_score, best_score, atol=tolerance, rtol=0)
    assert result.model_calls == 1
    assert model.batch_sizes == [6]
    for batch_index, candidate_index in enumerate(best_index.tolist()):
        assert result.best_names[batch_index] == candidates.names[candidate_index]
        assert torch.equal(
            result.best_mask[batch_index],
            candidates.candidate_masks[candidate_index, batch_index],
        )


@pytest.mark.parametrize("metric", ["entropy_drop", "risk_reduction"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_candidate_chunking_preserves_global_scores_indexes_and_call_count(
    metric,
    chunk_size,
):
    (
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_logits,
    ) = _case()
    base_metric_map = (
        entropy_per_token(base_logits)
        if metric == "entropy_drop"
        else decoding_risk_per_token(base_logits)
    )
    expected = evaluate_batched_lookahead(
        model,
        input_ids,
        predicted_token_ids,
        candidates,
        base_metric_map=base_metric_map,
        metric=metric,
        attention_mask=attention_mask,
        masked_active_mask=active_mask,
    )
    model.batch_sizes.clear()

    actual = evaluate_batched_lookahead(
        model,
        input_ids,
        predicted_token_ids,
        candidates,
        base_metric_map=base_metric_map,
        metric=metric,
        attention_mask=attention_mask,
        masked_active_mask=active_mask,
        candidate_chunk_size=chunk_size,
    )

    torch.testing.assert_close(actual.scores, expected.scores, atol=1e-6, rtol=0)
    assert torch.equal(actual.best_index, expected.best_index)
    assert torch.equal(actual.best_mask, expected.best_mask)
    assert actual.best_names == expected.best_names
    assert actual.model_calls == (3 + chunk_size - 1) // chunk_size
    expected_batch_sizes = []
    for start in range(0, 3, chunk_size):
        expected_batch_sizes.append(min(chunk_size, 3 - start) * 2)
    assert model.batch_sizes == expected_batch_sizes


def test_ties_select_first_valid_candidate_and_empty_rows_select_nothing():
    input_ids = torch.tensor([[9, 9, 3], [2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    active_mask = torch.tensor([[True, True, False], [False, False, False]])
    confidence = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    candidates = generate_top_confidence_candidates(
        confidence,
        active_mask,
        candidate_budget=2,
    )
    logits = torch.zeros((1, 3, 4))
    model = StaticModel(logits)
    predicted_token_ids = torch.zeros_like(input_ids)
    base_metric_map = entropy_per_token(logits.expand(2, -1, -1))

    result = evaluate_batched_lookahead(
        model,
        input_ids,
        predicted_token_ids,
        candidates,
        base_metric_map=base_metric_map,
        metric="entropy_drop",
        attention_mask=attention_mask,
        masked_active_mask=active_mask,
    )

    assert torch.equal(result.scores[:, 0], torch.zeros(2))
    assert torch.isneginf(result.scores[:, 1]).all()
    assert result.best_index.tolist() == [0, -1]
    assert result.best_names == (candidates.names[0], None)
    assert not result.best_mask[1].any()


def test_legacy_mapping_adapter_preserves_order_masks_k_and_invalid_rows():
    eligible = torch.tensor(
        [
            [False, True, True, True],
            [True, False, False, False],
        ]
    )
    masks = {
        "first": torch.tensor(
            [
                [False, True, False, True],
                [True, False, False, False],
            ]
        ),
        "second": torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, False],
            ]
        ),
    }

    batch = candidate_batch_from_mask_mapping(masks, eligible_mask=eligible)

    assert batch.names == ("first", "second")
    assert torch.equal(batch.candidate_masks, torch.stack(tuple(masks.values())))
    assert batch.requested_k.tolist() == [2, 1]
    assert batch.clipped_k.tolist() == [2, 1]
    assert batch.candidate_valid.tolist() == [[True, True], [True, False]]
    assert batch.selected_positions.tolist() == [
        [[1, 3], [0, -1]],
        [[2, 3], [-1, -1]],
    ]


def test_no_candidates_returns_defined_no_action_without_model_call():
    input_ids = torch.tensor([[1, 2, 3]])
    active_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    candidates = generate_top_confidence_candidates(
        torch.zeros_like(input_ids, dtype=torch.float32),
        active_mask,
        candidate_budget=4,
    )
    model = ContextModel()

    result = evaluate_batched_lookahead(
        model,
        input_ids,
        input_ids.clone(),
        candidates,
        base_metric_map=torch.zeros_like(input_ids, dtype=torch.float32),
        metric="risk_reduction",
        masked_active_mask=active_mask,
    )

    assert result.scores.shape == (0, 1)
    assert result.best_index.tolist() == [-1]
    assert torch.isneginf(result.best_score).all()
    assert not result.best_mask.any()
    assert result.best_names == (None,)
    assert result.model_calls == 0
    assert model.batch_sizes == []


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_candidate_chunk_size_must_be_a_positive_integer(chunk_size):
    (
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_logits,
    ) = _case()

    with pytest.raises(ValueError, match="candidate_chunk_size"):
        evaluate_batched_lookahead(
            model,
            input_ids,
            predicted_token_ids,
            candidates,
            base_metric_map=entropy_per_token(base_logits),
            metric="entropy_drop",
            attention_mask=attention_mask,
            masked_active_mask=active_mask,
            candidate_chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    ("sampler_class", "base_map_name", "chunk_size", "expected_batch_sizes"),
    [
        (EntropyDropSampler, "entropy", None, [6]),
        (EntropyDropSampler, "entropy", 2, [4, 2]),
        (RiskReductionSampler, "risk", None, [6]),
        (RiskReductionSampler, "risk", 2, [4, 2]),
    ],
)
def test_existing_samplers_use_shared_batched_evaluator(
    sampler_class,
    base_map_name,
    chunk_size,
    expected_batch_sizes,
):
    (
        model,
        input_ids,
        predicted_token_ids,
        attention_mask,
        active_mask,
        candidates,
        base_logits,
    ) = _case()
    sampler = sampler_class(model=model, tokenizer=None)
    if base_map_name == "entropy":
        base_map = sampler.get_entropy_per_token(base_logits)
    else:
        base_map = sampler.get_decoding_risk_per_token(base_logits)

    selected_mask, score_mapping, selected_names = sampler._select_best_candidate(
        x=input_ids,
        x0=predicted_token_ids,
        mask_index=active_mask,
        candidates=candidates.as_legacy_dict(),
        attention_mask=attention_mask,
        candidate_chunk_size=chunk_size,
        **{f"base_{base_map_name}_map": base_map},
    )

    assert model.batch_sizes == expected_batch_sizes
    assert tuple(score_mapping) == candidates.names
    assert selected_mask.shape == active_mask.shape
    assert len(selected_names) == input_ids.shape[0]
