"""Run with: pytest scripts/tests/test_tse_weighting.py -v"""

import pytest
import torch

from dllm.pipelines.tse.weighting import (
    online_entropy_weights,
    per_token_margin_weights,
    static_weights,
)


def test_static_weights_are_constant_and_normalized():
    result = static_weights(alpha=0.25, num_positions=3)

    assert torch.allclose(result.weights_a, torch.tensor([0.25] * 3))
    assert torch.allclose(result.weights_b, torch.tensor([0.75] * 3))


def test_online_entropy_favors_lower_entropy_model():
    probabilities_a = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    probabilities_b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    result = online_entropy_weights(probabilities_a, probabilities_b)

    assert torch.all(result.weights_b > result.weights_a)
    assert torch.allclose(result.weights_a + result.weights_b, torch.ones(2))
    assert "mean_entropy_a" in result.diagnostics
    assert "mean_entropy_b" in result.diagnostics


def test_online_entropy_equal_models_get_equal_weights():
    probabilities = torch.tensor([[0.75, 0.25], [0.75, 0.25]])

    result = online_entropy_weights(probabilities, probabilities)

    assert torch.allclose(result.weights_a, torch.full((2,), 0.5))
    assert torch.allclose(result.weights_b, torch.full((2,), 0.5))


def test_per_token_margin_favors_larger_margin():
    probabilities_a = torch.tensor([[0.9, 0.05, 0.05]])
    probabilities_b = torch.tensor([[0.55, 0.4, 0.05]])

    result = per_token_margin_weights(probabilities_a, probabilities_b)

    assert result.weights_a[0] > result.weights_b[0]
    assert torch.allclose(
        result.weights_a + result.weights_b, torch.ones(1), atol=1e-6
    )
    assert result.diagnostics["margin_a"][0] > result.diagnostics["margin_b"][0]


def test_per_token_margin_equal_margins_get_equal_weights():
    probabilities = torch.tensor([[0.7, 0.2, 0.1]])

    result = per_token_margin_weights(probabilities, probabilities)

    assert torch.allclose(result.weights_a, torch.tensor([0.5]))
    assert torch.allclose(result.weights_b, torch.tensor([0.5]))


@pytest.mark.parametrize("weight_temperature", [0.0, -1.0])
def test_weight_temperature_must_be_positive(weight_temperature):
    probabilities = torch.tensor([[0.7, 0.2, 0.1]])

    with pytest.raises(ValueError, match="weight_temperature must be positive"):
        per_token_margin_weights(probabilities, probabilities, weight_temperature)