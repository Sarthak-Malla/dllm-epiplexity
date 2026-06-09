"""
Unit tests for the no-lookahead spaced epiplexity sampler.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epiplexity_spaced_sampler.py -v
"""

from types import SimpleNamespace

import torch

from dllm.core.samplers.epiplexity_spaced import (
    SpacedEpiplexitySampler,
    SpacedEpiplexitySamplerConfig,
)


class _DummyTokenizer:
    mask_token_id = 99
    bos_token_id = 0
    eos_token_id = 1


class _CountingModel:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(self, input_ids, attention_mask=None):
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(
            batch_size, seq_len, self.vocab_size, device=input_ids.device
        )
        for pos in range(seq_len):
            logits[:, pos, 3 + (pos % (self.vocab_size - 3))] = 5.0
        return SimpleNamespace(logits=logits)


def _make_sampler(model=None):
    return SpacedEpiplexitySampler(
        model=model or _CountingModel(),
        tokenizer=_DummyTokenizer(),
    )


def test_select_spaced_transfer_index_uses_requested_offset():
    sampler = _make_sampler()
    mask_idx = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [False, True, True, False, True, True, False, True],
        ]
    )

    spaced_0 = sampler._select_spaced_transfer_index(
        mask_idx=mask_idx,
        num_transfer=torch.tensor([4, 3]),
        spaced_offset=0,
    )
    spaced_1 = sampler._select_spaced_transfer_index(
        mask_idx=mask_idx,
        num_transfer=torch.tensor([4, 3]),
        spaced_offset=1,
    )

    assert torch.where(spaced_0[0])[0].tolist() == [0, 2, 4, 6]
    assert torch.where(spaced_1[0])[0].tolist() == [1, 3, 5, 7]
    assert torch.where(spaced_0[1])[0].tolist() == [1, 3, 5]
    assert torch.where(spaced_1[1])[0].tolist() == [1, 4, 6]

    for transfer_index in (spaced_0, spaced_1):
        assert transfer_index.shape == mask_idx.shape
        assert transfer_index.dtype == torch.bool
        assert not (transfer_index & ~mask_idx).any()
        assert transfer_index.sum(dim=1).tolist() == [4, 3]


def test_select_spaced_transfer_index_rejects_invalid_offset():
    sampler = _make_sampler()
    mask_idx = torch.ones((1, 4), dtype=torch.bool)

    try:
        sampler._select_spaced_transfer_index(
            mask_idx=mask_idx,
            num_transfer=2,
            spaced_offset=2,
        )
    except ValueError as exc:
        assert "spaced_offset must be 0 or 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid spaced_offset")


def test_sample_uses_one_model_forward_per_unmask_step_without_lookahead():
    model = _CountingModel()
    sampler = _make_sampler(model=model)
    config = SpacedEpiplexitySamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=2,
        return_dict=True,
        spaced_offset=1,
    )

    output = sampler.sample(
        inputs=[torch.tensor([10], dtype=torch.long)],
        config=config,
    )

    assert output.sequences.shape == (1, 5)
    assert not (output.sequences[0, 1:5] == _DummyTokenizer.mask_token_id).any()
    assert model.calls == len(output.histories) - 1
