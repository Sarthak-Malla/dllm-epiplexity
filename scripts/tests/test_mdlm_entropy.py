"""Test native LLaDA entropy ranking.

Run with:
    source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
    pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_mdlm_entropy.py -q
"""

from types import SimpleNamespace

import torch

from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig


class TwoPositionModel(torch.nn.Module):
    """Give the two masked positions opposite confidence and entropy rankings."""

    def __init__(self):
        super().__init__()
        self.register_buffer("device_marker", torch.zeros(1))

    @property
    def device(self):
        return self.device_marker.device

    def forward(self, input_ids, attention_mask=None):
        logits = torch.full((*input_ids.shape, 5), -20.0)
        logits[:, 1, :3] = torch.tensor([0.6, 0.2, 0.2]).log()
        logits[:, 2, :3] = torch.tensor([0.55, 0.45, 1e-9]).log()
        return SimpleNamespace(logits=logits)


def test_entropy_commits_lowest_entropy_position_first():
    tokenizer = SimpleNamespace(mask_token_id=3, eos_token_id=4, bos_token_id=0)
    sampler = MDLMSampler(TwoPositionModel(), tokenizer)
    common = dict(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        temperature=0.0,
        probability_temperature=1.0,
        return_dict=True,
    )

    confidence = sampler.sample(
        [[0]], MDLMSamplerConfig(remasking="low_confidence", **common)
    )
    entropy = sampler.sample([[0]], MDLMSamplerConfig(remasking="entropy", **common))

    assert confidence.histories[1][0].tolist() == [0, 0, 3]
    assert entropy.histories[1][0].tolist() == [0, 3, 0]
    assert torch.equal(confidence.sequences, entropy.sequences)
