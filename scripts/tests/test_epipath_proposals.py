"""
Unit tests for EpiPath proposal generation.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_proposals.py -v
"""

import torch

from dllm.core.samplers.epipath_proposals import EpiProposalGenerator


def _base_metrics(sequence_length=12):
    top1_prob = torch.linspace(0.1, 0.9, sequence_length)
    margin = torch.linspace(0.05, 0.8, sequence_length)
    return {
        "top1_prob": top1_prob,
        "margin": margin,
    }


def test_dynamic_light_generates_valid_fixed_size_candidates():
    generator = EpiProposalGenerator()
    current_mask = torch.zeros(12, dtype=torch.bool)
    current_mask[2:11] = True
    revealed_mask = torch.zeros(12, dtype=torch.bool)
    revealed_mask[1] = True

    candidates = generator.generate(
        current_mask=current_mask,
        revealed_mask=revealed_mask,
        base_metrics=_base_metrics(),
        token_ids=torch.arange(12),
        num_transfer=3,
        proposal_preset="dynamic_light",
    )

    sources = {
        source for candidate in candidates for source in candidate.proposal_sources
    }
    assert sources == {
        "greedy",
        "spaced_0",
        "spaced_1",
        "coverage_anti_cluster",
        "frontier_conf_q60",
        "high_margin",
    }
    for candidate in candidates:
        assert len(candidate.positions) == 3
        assert len(candidate.token_ids) == 3
        assert all(current_mask[position].item() for position in candidate.positions)


def test_deduplicate_candidates_preserves_proposal_sources():
    generator = EpiProposalGenerator()
    current_mask = torch.ones(6, dtype=torch.bool)
    revealed_mask = torch.zeros(6, dtype=torch.bool)

    candidates = generator.generate(
        current_mask=current_mask,
        revealed_mask=revealed_mask,
        base_metrics=_base_metrics(sequence_length=6),
        token_ids=torch.arange(6),
        num_transfer=2,
        proposal_preset="minimal",
        proposals=["greedy", "high_margin"],
    )

    assert len(candidates) == 1
    assert candidates[0].proposal_sources == ["greedy", "high_margin"]
    assert candidates[0].metadata["proposal_sources"] == ["greedy", "high_margin"]
