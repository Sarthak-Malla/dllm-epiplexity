"""Run with: pytest scripts/tests/test_tse_fusion.py -v"""

import pytest
import torch

from dllm.pipelines.tse.fusion import fuse_probabilities, logits_to_probabilities


def test_temperature_one_matches_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    actual = logits_to_probabilities(logits, temperature=1.0)

    assert torch.allclose(actual, torch.softmax(logits, dim=-1))
    assert actual.dtype == torch.float32


def test_lower_temperature_sharpens_distribution():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    cool = logits_to_probabilities(logits, temperature=0.5)
    neutral = logits_to_probabilities(logits, temperature=1.0)

    assert cool[0, -1] > neutral[0, -1]
    assert cool[0, 0] < neutral[0, 0]


def test_higher_temperature_flattens_distribution():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    warm = logits_to_probabilities(logits, temperature=2.0)
    neutral = logits_to_probabilities(logits, temperature=1.0)

    assert warm[0, -1] < neutral[0, -1]
    assert warm[0, 0] > neutral[0, 0]


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_temperature_must_be_positive(temperature):
    with pytest.raises(ValueError, match="temperature must be positive"):
        logits_to_probabilities(torch.zeros(1, 3), temperature=temperature)


def test_equal_weight_fusion_is_arithmetic_mean():
    probabilities_a = torch.tensor([[0.8, 0.2]])
    probabilities_b = torch.tensor([[0.2, 0.8]])

    actual = fuse_probabilities(probabilities_a, probabilities_b, alpha=0.5)

    assert torch.allclose(actual, torch.tensor([[0.5, 0.5]]))


def test_alpha_endpoints_select_one_model():
    probabilities_a = torch.tensor([[0.8, 0.2]])
    probabilities_b = torch.tensor([[0.2, 0.8]])

    assert torch.allclose(
        fuse_probabilities(probabilities_a, probabilities_b, alpha=1.0),
        probabilities_a,
    )
    assert torch.allclose(
        fuse_probabilities(probabilities_a, probabilities_b, alpha=0.0),
        probabilities_b,
    )


def test_fusion_rejects_invalid_alpha_and_shapes():
    with pytest.raises(ValueError, match="alpha"):
        fuse_probabilities(torch.ones(1, 2), torch.ones(1, 2), alpha=1.1)

    with pytest.raises(ValueError, match="identical shapes"):
        fuse_probabilities(torch.ones(1, 2), torch.ones(1, 3))