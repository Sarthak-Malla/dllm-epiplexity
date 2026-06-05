"""
Unit tests for the decoding-risk epiplexity sampler.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epiplexity_risk_sampler.py -v
"""

from pathlib import Path
from types import SimpleNamespace

import torch

from dllm.core.samplers.epiplexity_risk import RiskEpiplexitySampler


class _DummyTokenizer:
    mask_token_id = 99
    bos_token_id = 0
    eos_token_id = 1


class _RiskLookaheadModel:
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=input_ids.device)

        for b in range(batch_size):
            if input_ids[b, 0] == 2:
                logits[b, :, 3] = 8.0
            elif input_ids[b, 1] == 3:
                logits[b, :, 3] = 0.5
            else:
                logits[b, :, 3] = 2.0

        return SimpleNamespace(logits=logits)


def _make_sampler():
    return RiskEpiplexitySampler(model=_RiskLookaheadModel(), tokenizer=_DummyTokenizer())


def test_get_decoding_risk_per_token_uses_one_minus_max_probability():
    sampler = _make_sampler()
    logits = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])

    risk = sampler.get_decoding_risk_per_token(logits)
    expected = 1.0 - torch.softmax(logits, dim=-1).amax(dim=-1)

    assert torch.allclose(risk, expected)


def test_select_best_candidate_prefers_heldout_risk_reduction():
    sampler = _make_sampler()
    x = torch.tensor([[99, 99, 99]])
    x0 = torch.tensor([[2, 3, 4]])
    mask_index = x == 99
    attention_mask = torch.ones_like(x)
    base_risk_map = torch.full_like(x, 0.9, dtype=torch.float)
    candidates = {
        "reveal_pos_0": torch.tensor([[True, False, False]]),
        "reveal_pos_1": torch.tensor([[False, True, False]]),
    }

    best_c_idx, risk_reductions = sampler._select_best_candidate(
        x=x,
        x0=x0,
        mask_index=mask_index,
        candidates=candidates,
        attention_mask=attention_mask,
        base_risk_map=base_risk_map,
    )

    assert torch.equal(best_c_idx, candidates["reveal_pos_0"])
    assert risk_reductions["reveal_pos_0"].item() > risk_reductions["reveal_pos_1"].item()


def test_risk_sampler_does_not_define_entropy_drop_verifier():
    source_path = Path(
        "/home/sarthak.malla/dllm-epiplexity/dllm/core/samplers/epiplexity_risk.py"
    )
    source = source_path.read_text()

    assert "entropy_drop" not in source
    assert "base_entropy_map" not in source
    assert "get_entropy_per_token" not in source
