"""Test base-pass-only candidate scoring for dependency ablations.

Run from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_non_lookahead.py -v
"""

import pytest
import torch

from dllm.core.samplers.batched_lookahead import candidate_batch_from_mask_mapping
from dllm.core.samplers.non_lookahead import (
    select_candidates_without_lookahead,
    top2_probability_margin,
)


def _candidates():
    """Return three equal-size candidate actions over one batch row."""
    eligible = torch.ones((1, 4), dtype=torch.bool)
    return candidate_batch_from_mask_mapping(
        {
            "candidate_0": torch.tensor([[True, True, False, False]]),
            "candidate_1": torch.tensor([[False, True, True, False]]),
            "candidate_2": torch.tensor([[False, False, True, True]]),
        },
        eligible_mask=eligible,
    )


@pytest.mark.parametrize(
    ("selector", "expected_index"),
    (
        ("max_confidence", 0),
        ("min_entropy", 2),
        ("min_top2_margin", 1),
    ),
)
def test_candidate_selectors_use_per_token_means(selector, expected_index):
    candidates = _candidates()
    output = select_candidates_without_lookahead(
        candidates,
        confidence=torch.tensor([[0.90, 0.80, 0.40, 0.20]]),
        entropy=torch.tensor([[0.80, 0.70, 0.20, 0.10]]),
        top2_margin=torch.tensor([[0.90, 0.10, 0.10, 0.90]]),
        selector=selector,
    )

    assert output.best_index.tolist() == [expected_index]
    assert output.best_names == (f"candidate_{expected_index}",)
    assert torch.equal(
        output.best_mask,
        candidates.candidate_masks[expected_index],
    )


def test_top2_probability_margin_uses_the_two_largest_probabilities():
    probabilities = torch.tensor(
        [[[0.10, 0.60, 0.25, 0.05], [0.40, 0.35, 0.20, 0.05]]]
    )

    torch.testing.assert_close(
        top2_probability_margin(probabilities),
        torch.tensor([[0.35, 0.05]]),
    )


def test_top2_selector_requires_a_margin_map():
    with pytest.raises(ValueError, match="requires top2_margin"):
        select_candidates_without_lookahead(
            _candidates(),
            confidence=torch.ones((1, 4)),
            entropy=torch.ones((1, 4)),
            top2_margin=None,
            selector="min_top2_margin",
        )
