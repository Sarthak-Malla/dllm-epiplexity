"""
Test the one-position counterfactual entropy/risk oracle.

Run on a compute node with:
    source /home/sarthak.malla/.zshrc
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_counterfactual_oracle.py -v
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.samplers.counterfactual import (
    entropy_per_token,
    evaluate_one_position_counterfactuals,
)


def _manual_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute a full-tensor FP32 reference, including suppressed tokens."""
    probabilities = F.softmax(logits.float(), dim=-1)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    terms = torch.where(
        probabilities > 0,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=-1)


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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("layout", ("contiguous", "transposed", "strided_vocabulary"))
def test_entropy_chunks_preserve_fp32_full_vocabulary_values_and_inputs(
    monkeypatch, dtype, layout,
):
    generator = torch.Generator().manual_seed(314)
    logits = (torch.randn((3, 97, 257), generator=generator) * 4).to(dtype)
    logits[..., ::13] = -torch.inf
    # A deterministic distribution in a later chunk must have defined zero entropy.
    logits[2, 80] = -torch.inf
    logits[2, 80, 24] = 0.0
    if layout == "transposed":
        logits = logits.transpose(0, 1)
    elif layout == "strided_vocabulary":
        logits = logits[..., ::2]
    if layout != "contiguous":
        assert not logits.is_contiguous()
    before = logits.clone()
    expected = _manual_entropy(logits)
    original = F.log_softmax
    observed_rows = []

    def bounded_fp32_log_softmax(values, *args, **kwargs):
        rows = values.numel() // values.shape[-1]
        observed_rows.append(rows)
        assert 0 < rows <= 64
        assert values.dtype == torch.float32
        assert values.shape[-1] == logits.shape[-1]
        return original(values, *args, **kwargs)

    monkeypatch.setattr(F, "log_softmax", bounded_fp32_log_softmax)
    actual = entropy_per_token(logits)

    assert sum(observed_rows) == 3 * 97
    assert len(observed_rows) > 1
    assert actual.shape == logits.shape[:-1]
    assert actual.dtype == torch.float32
    assert actual.device == logits.device
    assert torch.isfinite(actual).all()
    assert torch.equal(logits, before)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    deterministic_position = (80, 2) if layout == "transposed" else (2, 80)
    assert actual[deterministic_position].item() == 0.0


@pytest.mark.parametrize(
    ("invalid", "message"),
    (
        ("nan", "NaN or positive infinity"),
        ("positive_infinity", "NaN or positive infinity"),
        ("all_suppressed", "at least one finite logit"),
    ),
)
def test_entropy_rejects_invalid_logits_in_later_chunks(invalid, message):
    logits = torch.zeros((2, 97, 7), dtype=torch.bfloat16)
    if invalid == "all_suppressed":
        logits[1, 80] = -torch.inf
    else:
        logits[1, 80, 3] = torch.nan if invalid == "nan" else torch.inf

    with pytest.raises(ValueError, match=message):
        entropy_per_token(logits)


def test_entropy_retains_invalid_logit_precedence_across_chunks():
    logits = torch.zeros((2, 97, 7))
    logits[0, 0] = -torch.inf
    logits[1, 80, 3] = torch.nan

    with pytest.raises(ValueError, match="NaN or positive infinity"):
        entropy_per_token(logits)


@pytest.mark.parametrize("shape", ((0, 97, 7), (2, 0, 7), (0, 7)))
def test_entropy_preserves_empty_leading_dimensions(shape):
    logits = torch.empty(shape, dtype=torch.bfloat16)

    actual = entropy_per_token(logits)

    assert actual.shape == shape[:-1]
    assert actual.numel() == 0
    assert actual.dtype == torch.float32
    assert actual.device == logits.device


def test_entropy_chunking_preserves_finite_logits_gradients():
    generator = torch.Generator().manual_seed(72)
    logits = torch.randn((2, 81, 17), generator=generator, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    weights = torch.linspace(0.2, 1.2, 2 * 81).reshape(2, 81)

    actual = entropy_per_token(logits)
    expected = _manual_entropy(reference_logits)
    actual_gradient, = torch.autograd.grad((actual * weights).sum(), logits)
    expected_gradient, = torch.autograd.grad(
        (expected * weights).sum(), reference_logits,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=2e-6, rtol=1e-5,
    )


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
