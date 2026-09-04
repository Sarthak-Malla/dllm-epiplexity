"""
Test the one-position counterfactual entropy/risk oracle.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_counterfactual_oracle.py" -v
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.samplers.counterfactual import (
    evaluate_one_position_counterfactuals,
)


def _manual_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=-1)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


def _manual_risk(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.softmax(logits.float(), dim=-1).amax(dim=-1)


class TinyContextModel(nn.Module):
    """Produce deterministic logits that change globally after one reveal."""

    def __init__(self, vocabulary_size: int = 5) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.batch_sizes: list[int] = []
        self.attention_masks: list[torch.Tensor | None] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.batch_sizes.append(input_ids.shape[0])
        self.attention_masks.append(
            None if attention_mask is None else attention_mask.detach().clone()
        )
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.float32)
        else:
            valid = (attention_mask != 0).to(dtype=torch.float32)
        context = (input_ids.float() * valid).sum(dim=-1, keepdim=True)
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        centers = torch.remainder(context + 1.7 * positions, self.vocabulary_size)
        vocabulary = torch.arange(
            self.vocabulary_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        logits = -0.4 * (vocabulary - centers.unsqueeze(-1)).square()
        logits = logits + 0.03 * input_ids.float().unsqueeze(-1) * vocabulary
        return SimpleNamespace(logits=logits)


class StaticLookaheadModel(nn.Module):
    """Return fixed lookahead logits for region-isolation assertions."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.seen_attention_mask: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.seen_attention_mask = attention_mask
        return SimpleNamespace(logits=self.logits.expand(input_ids.shape[0], -1, -1))


def test_heldout_mask_excludes_only_the_revealed_active_position():
    model = TinyContextModel()
    input_ids = torch.tensor([[1, 4, 4, 2, 4]])
    attention_mask = torch.ones_like(input_ids)
    active_mask = torch.tensor([[False, True, True, False, True]])
    evaluation_mask = torch.tensor([[False, False, True, False, False]])
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    result = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
        evaluation_mask=evaluation_mask,
    )

    assert result.candidate_count == 1
    assert torch.equal(result.positions, torch.tensor([2]))
    assert torch.equal(
        result.heldout_mask,
        torch.tensor([[False, True, False, False, True]]),
    )
    assert torch.equal(result.heldout_counts, torch.tensor([2]))
    assert torch.allclose(
        result.entropy_drop_per_heldout * result.heldout_counts,
        result.entropy_drop_sum,
    )
    assert torch.allclose(
        result.risk_reduction_per_heldout * result.heldout_counts,
        result.risk_reduction_sum,
    )


def test_prompt_inactive_and_padding_positions_contribute_zero():
    input_ids = torch.tensor([[1, 2, 9, 3, 9, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    active_mask = torch.tensor([[0, 0, 1, 0, 1, 0]])
    evaluation_mask = torch.tensor([[0, 0, 1, 0, 0, 0]])
    base_logits = torch.zeros((1, 6, 2))
    lookahead_logits = torch.tensor(
        [
            [
                [12.0, -12.0],
                [12.0, -12.0],
                [12.0, -12.0],
                [12.0, -12.0],
                [1.3862944, 0.0],
                [12.0, -12.0],
            ]
        ]
    )
    model = StaticLookaheadModel(lookahead_logits)

    result = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
        evaluation_mask=evaluation_mask,
    )

    expected_base_entropy = _manual_entropy(base_logits)[0, 4]
    expected_lookahead_entropy = _manual_entropy(lookahead_logits)[0, 4]
    expected_base_risk = _manual_risk(base_logits)[0, 4]
    expected_lookahead_risk = _manual_risk(lookahead_logits)[0, 4]
    assert torch.equal(
        result.heldout_mask,
        torch.tensor([[False, False, False, False, True, False]]),
    )
    assert torch.allclose(result.base_entropy_sum, expected_base_entropy[None])
    assert torch.allclose(
        result.lookahead_entropy_sum,
        expected_lookahead_entropy[None],
    )
    assert torch.allclose(result.base_risk_sum, expected_base_risk[None])
    assert torch.allclose(result.lookahead_risk_sum, expected_lookahead_risk[None])
    assert torch.equal(model.seen_attention_mask, attention_mask)


def test_single_remaining_position_returns_defined_zero_scores():
    model = TinyContextModel()
    input_ids = torch.tensor([[1, 4, 2]])
    attention_mask = torch.ones_like(input_ids)
    active_mask = torch.tensor([[False, True, False]])
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    result = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
    )

    assert result.candidate_count == 1
    assert torch.equal(result.heldout_counts, torch.tensor([0]))
    for values in (
        result.base_entropy_sum,
        result.lookahead_entropy_sum,
        result.entropy_drop_sum,
        result.entropy_drop_per_heldout,
        result.base_risk_sum,
        result.lookahead_risk_sum,
        result.risk_reduction_sum,
        result.risk_reduction_per_heldout,
    ):
        assert torch.equal(values, torch.zeros_like(values))
        assert torch.isfinite(values).all()


def test_empty_evaluation_returns_empty_output_without_lookahead():
    model = TinyContextModel()
    input_ids = torch.tensor([[1, 2, 3]])
    base_logits = model(input_ids=input_ids).logits
    model.batch_sizes.clear()

    result = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=torch.zeros_like(input_ids, dtype=torch.bool),
    )

    assert result.candidate_count == 0
    assert result.heldout_mask.shape == (0, 3)
    assert result.entropy_drop_sum.shape == (0,)
    assert result.risk_reduction_sum.shape == (0,)
    assert model.batch_sizes == []


def test_batched_oracle_matches_per_position_tiny_model_reference():
    model = TinyContextModel()
    input_ids = torch.tensor(
        [
            [1, 4, 4, 2, 4, 0],
            [2, 3, 4, 4, 1, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    active_mask = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    evaluation_mask = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    original_input_ids = input_ids.clone()
    model.batch_sizes.clear()

    result = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
        evaluation_mask=evaluation_mask,
    )

    assert model.batch_sizes == [4]
    assert torch.equal(input_ids, original_input_ids)
    assert torch.equal(result.batch_indices, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(result.positions, torch.tensor([1, 4, 2, 3]))

    reference_values = {
        "base_entropy_sum": [],
        "lookahead_entropy_sum": [],
        "entropy_drop_sum": [],
        "entropy_drop_per_heldout": [],
        "base_risk_sum": [],
        "lookahead_risk_sum": [],
        "risk_reduction_sum": [],
        "risk_reduction_per_heldout": [],
    }
    for batch_index, position in zip(
        result.batch_indices.tolist(),
        result.positions.tolist(),
    ):
        candidate_state = input_ids[batch_index : batch_index + 1].clone()
        token_id = base_logits[batch_index, position].argmax()
        candidate_state[0, position] = token_id
        lookahead = model(
            input_ids=candidate_state,
            attention_mask=attention_mask[batch_index : batch_index + 1],
        ).logits
        heldout = active_mask[batch_index].clone()
        heldout[position] = False
        count = int(heldout.sum().item())

        base_entropy = _manual_entropy(base_logits[batch_index])[heldout].sum()
        lookahead_entropy = _manual_entropy(lookahead[0])[heldout].sum()
        entropy_drop = base_entropy - lookahead_entropy
        base_risk = _manual_risk(base_logits[batch_index])[heldout].sum()
        lookahead_risk = _manual_risk(lookahead[0])[heldout].sum()
        risk_reduction = base_risk - lookahead_risk
        reference_values["base_entropy_sum"].append(base_entropy)
        reference_values["lookahead_entropy_sum"].append(lookahead_entropy)
        reference_values["entropy_drop_sum"].append(entropy_drop)
        reference_values["entropy_drop_per_heldout"].append(
            entropy_drop / count if count else torch.zeros_like(entropy_drop)
        )
        reference_values["base_risk_sum"].append(base_risk)
        reference_values["lookahead_risk_sum"].append(lookahead_risk)
        reference_values["risk_reduction_sum"].append(risk_reduction)
        reference_values["risk_reduction_per_heldout"].append(
            risk_reduction / count if count else torch.zeros_like(risk_reduction)
        )

    for name, expected_values in reference_values.items():
        expected = torch.stack(expected_values)
        assert torch.allclose(getattr(result, name), expected, atol=1e-6)


def test_chunked_oracle_exactly_matches_single_batch_candidate_order_and_scores():
    model = TinyContextModel()
    input_ids = torch.tensor(
        [
            [1, 4, 4, 2, 4, 0],
            [2, 3, 4, 4, 1, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    active_mask = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    evaluation_mask = torch.tensor(
        [
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    model.batch_sizes.clear()

    unchunked = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
        evaluation_mask=evaluation_mask,
    )
    assert model.batch_sizes == [4]
    model.batch_sizes.clear()
    chunked = evaluate_one_position_counterfactuals(
        model,
        input_ids,
        base_logits,
        masked_active_mask=active_mask,
        attention_mask=attention_mask,
        evaluation_mask=evaluation_mask,
        candidate_chunk_size=2,
    )
    assert model.batch_sizes == [2, 2]

    for field in fields(unchunked):
        expected = getattr(unchunked, field.name)
        actual = getattr(chunked, field.name)
        if torch.is_floating_point(expected):
            assert torch.allclose(actual, expected, atol=1e-6)
        else:
            assert torch.equal(actual, expected)


def test_candidate_chunk_size_must_be_positive():
    model = TinyContextModel()
    input_ids = torch.tensor([[1, 4, 4]])
    base_logits = model(input_ids=input_ids).logits

    with pytest.raises(ValueError, match="candidate_chunk_size"):
        evaluate_one_position_counterfactuals(
            model,
            input_ids,
            base_logits,
            masked_active_mask=torch.tensor([[False, True, True]]),
            candidate_chunk_size=0,
        )


@pytest.mark.parametrize(
    ("active_mask", "evaluation_mask", "message"),
    [
        (
            torch.tensor([[False, True, False]]),
            torch.tensor([[True, False, False]]),
            "evaluation_mask must be a subset",
        ),
        (
            torch.tensor([[False, True, True]]),
            torch.tensor([[False, True, False]]),
            "masked_active_mask must be a subset",
        ),
    ],
)
def test_invalid_regions_are_rejected(active_mask, evaluation_mask, message):
    model = TinyContextModel()
    input_ids = torch.tensor([[1, 4, 4]])
    attention_mask = torch.tensor([[1, 1, 0]])
    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    with pytest.raises(ValueError, match=message):
        evaluate_one_position_counterfactuals(
            model,
            input_ids,
            base_logits,
            masked_active_mask=active_mask,
            attention_mask=attention_mask,
            evaluation_mask=evaluation_mask,
        )
