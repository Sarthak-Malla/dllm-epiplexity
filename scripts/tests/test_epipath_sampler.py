"""
Unit tests for the EpiPath sampler.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_sampler.py -v
"""

import json
from types import SimpleNamespace

import torch

from dllm.core.samplers.epipath import EpiPathSampler, EpiPathSamplerConfig
from dllm.core.samplers.epipath_proposals import CandidateReveal


class _DummyTokenizer:
    mask_token_id = 99
    bos_token_id = 0
    eos_token_id = 1


class _PositionModel:
    def __init__(self, vocab_size=32):
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
            token_id = 3 + (pos % (self.vocab_size - 3))
            logits[:, pos, token_id] = 2.0 + pos * 0.1
        return SimpleNamespace(logits=logits)


class _TwoCandidateGenerator:
    def __init__(self):
        self.forced_positions = None

    def generate(
        self,
        current_mask,
        revealed_mask,
        base_metrics,
        token_ids,
        num_transfer,
        proposal_preset="dynamic_light",
        proposals=None,
    ):
        valid_positions = torch.where(current_mask)[0].tolist()
        k = int(num_transfer)
        first_positions = valid_positions[:k]
        last_positions = valid_positions[-k:]
        if self.forced_positions is None:
            self.forced_positions = last_positions
        return [
            CandidateReveal(
                candidate_id="cand_000_first",
                proposal_type="first",
                positions=first_positions,
                token_ids=[int(token_ids[pos].item()) for pos in first_positions],
                proposal_sources=["first"],
            ),
            CandidateReveal(
                candidate_id="cand_001_forced",
                proposal_type="forced",
                positions=last_positions,
                token_ids=[int(token_ids[pos].item()) for pos in last_positions],
                proposal_sources=["forced"],
            ),
        ]


def _make_sampler():
    return EpiPathSampler(model=_PositionModel(), tokenizer=_DummyTokenizer())


def test_force_proposal_commits_requested_candidate_positions(tmp_path):
    sampler = _make_sampler()
    generator = _TwoCandidateGenerator()
    sampler.proposal_generator = generator
    config = EpiPathSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=2,
        return_dict=True,
        epipath_score_mode="force_proposal",
        epipath_force_proposal="forced",
        epipath_log_path=str(tmp_path / "epipath.jsonl"),
    )

    output = sampler.sample(inputs=[torch.tensor([10])], config=config)
    first_step = output.histories[1][0]
    unmasked_positions = torch.where(first_step != _DummyTokenizer.mask_token_id)[0]
    generated_unmasked = [
        int(pos.item()) for pos in unmasked_positions if int(pos.item()) != 0
    ]

    assert generated_unmasked == generator.forced_positions


def test_sampler_writes_candidate_jsonl_records(tmp_path):
    log_path = tmp_path / "epipath.jsonl"
    sampler = _make_sampler()
    sampler.proposal_generator = _TwoCandidateGenerator()
    config = EpiPathSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=2,
        return_dict=True,
        epipath_score_mode="force_proposal",
        epipath_force_proposal="forced",
        epipath_log_path=str(log_path),
        epipath_run_id="unit",
    )

    sampler.sample(inputs=[torch.tensor([10])], config=config)

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert records[0]["record_type"] == "step"
    assert records[0]["run_id"] == "unit"
    assert records[0]["selected_proposal_type"] == "forced"
    assert records[0]["candidates"][1]["selected"]
