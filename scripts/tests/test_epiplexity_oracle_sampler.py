"""
Unit tests for the oracle epiplexity sampler.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epiplexity_oracle_sampler.py -v
"""

from types import SimpleNamespace

import torch

from dllm.core.samplers.epiplexity_oracle import (
    OracleEpiplexitySampler,
    OracleEpiplexitySamplerConfig,
)


class _DummyTokenizer:
    mask_token_id = 99
    bos_token_id = 0
    eos_token_id = 1


class _DummyModel:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=input_ids.device)
        for pos in range(seq_len):
            logits[:, pos, 3 + (pos % (self.vocab_size - 3))] = 5.0
        return SimpleNamespace(logits=logits)


def _make_sampler():
    return OracleEpiplexitySampler(model=_DummyModel(), tokenizer=_DummyTokenizer())


def test_generate_candidate_sets_keeps_batch_rows_independent():
    sampler = _make_sampler()
    mask_idx = torch.tensor(
        [
            [True, False, False, True, False, True],
            [False, True, True, False, True, False],
        ]
    )
    confidence = torch.tensor(
        [
            [0.9, -torch.inf, -torch.inf, 0.8, -torch.inf, 0.7],
            [-torch.inf, 0.9, 0.2, -torch.inf, 0.8, -torch.inf],
        ]
    )

    candidates = sampler.generate_candidate_sets(
        confidence=confidence,
        mask_idx=mask_idx,
        num_transfer=torch.tensor([2, 1]),
        strategy="mixed",
    )

    assert set(candidates) == {"greedy", "spaced_0", "spaced_1"}
    for candidate in candidates.values():
        assert candidate.shape == mask_idx.shape
        assert candidate.dtype == torch.bool
        assert not (candidate & ~mask_idx).any()
        assert candidate.sum(dim=1).tolist() == [2, 1]

    greedy = candidates["greedy"]
    assert greedy[0, 0]
    assert greedy[0, 3]
    assert not greedy[0, 1]
    assert greedy[1, 1]
    assert not greedy[1, 0]


def test_sample_respects_current_block_for_batched_inputs():
    sampler = _make_sampler()
    config = OracleEpiplexitySamplerConfig(
        max_new_tokens=4,
        block_size=2,
        steps=4,
        return_dict=True,
        oracle_candidate_strategy="mixed",
    )
    inputs = [
        torch.tensor([10], dtype=torch.long),
        torch.tensor([10, 11], dtype=torch.long),
    ]

    output = sampler.sample(inputs=inputs, config=config)

    assert output.sequences.shape == (2, 6)
    assert len(output.histories) >= 5

    first_block_histories = output.histories[1:3]
    for history in first_block_histories:
        assert torch.equal(history[0, 3:5], torch.full((2,), _DummyTokenizer.mask_token_id))
        assert torch.equal(history[1, 4:6], torch.full((2,), _DummyTokenizer.mask_token_id))

    assert not (output.sequences[0, 1:5] == _DummyTokenizer.mask_token_id).any()
    assert not (output.sequences[1, 2:6] == _DummyTokenizer.mask_token_id).any()
