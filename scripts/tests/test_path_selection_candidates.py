"""
Test current entropy-drop candidate masks and a deterministic score trace.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_path_selection_candidates.py -v

Add ``-k deterministic_entropy_drop_trace -s`` to print the mocked score trace.
"""

import json
from types import SimpleNamespace

import torch

from dllm.core.samplers.entropy_drop import EntropyDropSampler


def _make_sampler() -> EntropyDropSampler:
    """Create a sampler without a model; candidate generation is model-independent."""
    return EntropyDropSampler(model=None, tokenizer=None)


class _DeterministicEntropyModel(torch.nn.Module):
    """Return successor logits determined only by which test position is revealed."""

    mask_token_id = 99

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape

        strengths = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.float32
        )
        for row in range(input_ids.shape[0]):
            if input_ids[row, 1] != self.mask_token_id:
                strengths[row] = 4.0
            elif input_ids[row, 2] != self.mask_token_id:
                strengths[row] = 0.0
            elif input_ids[row, 3] != self.mask_token_id:
                strengths[row] = 1.0

        logits = torch.zeros(
            (*input_ids.shape, 2), device=input_ids.device, dtype=torch.float32
        )
        logits[..., 0] = strengths.unsqueeze(-1)
        return SimpleNamespace(logits=logits)


def _build_entropy_drop_trace() -> dict:
    """Build a JSON-serializable trace for three one-position candidates."""
    model = _DeterministicEntropyModel()
    sampler = EntropyDropSampler(model=model, tokenizer=None)
    current_state = torch.tensor([[7, 99, 99, 99]])
    attention_mask = torch.ones_like(current_state)
    mask_index = current_state == model.mask_token_id

    base_logits = model(current_state, attention_mask=attention_mask).logits
    predicted_x0 = torch.argmax(base_logits, dim=-1)
    base_entropy_map = sampler.get_entropy_per_token(base_logits)
    candidates = {
        "left": torch.tensor([[False, True, False, False]]),
        "middle": torch.tensor([[False, False, True, False]]),
        "right": torch.tensor([[False, False, False, True]]),
    }

    selected_mask, scores, selected_names = sampler._select_best_candidate(
        x=current_state,
        x0=predicted_x0,
        mask_index=mask_index,
        candidates=candidates,
        attention_mask=attention_mask,
        base_entropy_map=base_entropy_map,
    )

    return {
        "current_state": current_state.tolist(),
        "predicted_x0": predicted_x0.tolist(),
        "mask_index": mask_index.tolist(),
        "candidate_masks": {
            name: candidate.tolist() for name, candidate in candidates.items()
        },
        "entropy_drop_scores": {
            name: round(score.item(), 6) for name, score in scores.items()
        },
        "selected_candidate": selected_names[0],
        "selected_mask": selected_mask.tolist(),
    }


def test_mixed_candidates_respect_masks_shapes_and_per_row_k():
    sampler = _make_sampler()
    confidence = torch.tensor(
        [
            [0.99, 0.10, 0.20, 0.98, 0.40, 0.30],
            [0.60, 0.95, 0.70, 0.20, 0.90, 0.85],
        ]
    )
    mask_idx = torch.tensor(
        [
            [False, True, True, False, True, True],
            [True, False, True, True, False, False],
        ]
    )
    per_row_k = torch.tensor([2, 1])

    candidates = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=per_row_k,
        strategy="mixed",
    )

    assert list(candidates) == [
        "spaced_0",
        "spaced_1",
        "random",
        "high_entropy",
    ]
    for candidate in candidates.values():
        assert candidate.dtype == torch.bool
        assert candidate.shape == mask_idx.shape
        assert not torch.any(candidate & ~mask_idx)
        assert candidate.sum(dim=1).tolist() == [2, 1]


def test_greedy_selects_highest_masked_confidence():
    sampler = _make_sampler()
    confidence = torch.tensor([[0.99, 0.20, 0.80, 0.95, 0.50]])
    mask_idx = torch.tensor([[False, True, True, False, True]])

    candidates = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=2,
        strategy="greedy",
    )

    expected = torch.tensor([[False, False, True, False, True]])
    assert torch.equal(candidates["greedy"], expected)


def test_high_entropy_selects_lowest_masked_confidence():
    sampler = _make_sampler()
    confidence = torch.tensor([[0.01, 0.20, 0.80, 0.02, 0.50]])
    mask_idx = torch.tensor([[False, True, True, False, True]])

    candidates = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=2,
        strategy="high_entropy",
    )

    expected = torch.tensor([[False, True, False, False, True]])
    assert torch.equal(candidates["high_entropy"], expected)


def test_random_is_deterministic_with_fixed_torch_seed():
    sampler = _make_sampler()
    confidence = torch.zeros((2, 6))
    mask_idx = torch.tensor(
        [
            [False, True, True, False, True, True],
            [True, False, True, True, False, True],
        ]
    )
    per_row_k = torch.tensor([2, 3])

    torch.manual_seed(1234)
    first = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=per_row_k,
        strategy="random",
    )["random"]
    torch.manual_seed(1234)
    second = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=per_row_k,
        strategy="random",
    )["random"]

    assert torch.equal(first, second)
    assert first.sum(dim=1).tolist() == [2, 3]
    assert not torch.any(first & ~mask_idx)


def test_zero_and_oversized_k_are_clamped_per_row():
    sampler = _make_sampler()
    confidence = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70, 0.80],
        ]
    )
    mask_idx = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, False],
        ]
    )

    candidates = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=torch.tensor([0, 99]),
        strategy="mixed",
    )

    for candidate in candidates.values():
        assert candidate[0].sum().item() == 0
        assert torch.equal(candidate[1], mask_idx[1])


def test_deterministic_entropy_drop_trace_matches_reference():
    expected_trace = {
        "current_state": [[7, 99, 99, 99]],
        "predicted_x0": [[0, 0, 0, 0]],
        "mask_index": [[False, True, True, True]],
        "candidate_masks": {
            "left": [[False, True, False, False]],
            "middle": [[False, False, True, False]],
            "right": [[False, False, False, True]],
        },
        "entropy_drop_scores": {
            "left": 1.206105,
            "middle": 0.0,
            "right": 0.221888,
        },
        "selected_candidate": "left",
        "selected_mask": [[False, True, False, False]],
    }

    first_trace = _build_entropy_drop_trace()
    second_trace = _build_entropy_drop_trace()

    print(json.dumps(first_trace, indent=2, sort_keys=True))
    assert first_trace == second_trace
    assert first_trace == expected_trace
