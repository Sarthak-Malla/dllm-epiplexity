"""Test training-free proposal controls with small deterministic tensors.

Run on a compute node after preparing the environment:
    source /home/sarthak.malla/.zshrc
    conda activate /home/sarthak.malla/miniconda3/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_mechanisms.py -q
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dllm.core.samplers import dependency_guided as guided
from dllm.core.samplers.adaptive_cardinality import _construct_stopped_soft_full_subset
from dllm.core.samplers.dependency import DependencyCaptureOutput, filter_dependency_sinks
from dllm.core.samplers.parallel_candidates import generate_parallel_dependency_candidates


def _dependency(matrix, *, renormalized=False):
    """Wrap a batch of square attention matrices with absolute mappings."""
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    batch, width, _ = matrix.shape
    positions = torch.arange(width).expand(batch, -1).clone()
    valid = torch.ones_like(positions, dtype=torch.bool)
    return DependencyCaptureOutput(
        directed=matrix, query_positions=positions, key_positions=positions.clone(),
        query_valid_mask=valid, key_valid_mask=valid.clone(), layer_ids=(0,),
        renormalized_selected_keys=renormalized, diagonal_zeroed=True,
    )


def _threshold(confidence, eligible, *, config=None, entropy=None):
    """Call direct threshold selection without loading any model."""
    confidence = torch.tensor(confidence, dtype=torch.float32)
    eligible = torch.tensor(eligible, dtype=torch.bool)
    config = config or guided.DependencyGuidedSamplerConfig(
        proposal_strategy="confidence_threshold",
    )
    return guided.build_confidence_threshold_candidates(
        SimpleNamespace(structure=None, captures=None),
        active_mask=eligible, requested_k=eligible.sum(dim=-1), anchor_state=None,
        response_mask=eligible, attention_mask=torch.ones_like(eligible),
        entropy_map=torch.zeros_like(confidence) if entropy is None else entropy,
        confidence=confidence, config=config, generation_seed=42,
    )


def test_confidence_threshold_caps_qualified_tokens_and_falls_back_per_row():
    candidates, dependency, reconstruction, _ = _threshold(
        [[1, .98, .95, .9, .99, .94, .93],
         [1, .2, .3, .3, .1, .1, .1],
         [1, .9, .8, .7, .6, .5, .4],
         [1, .9, .8, .7, .6, .5, .4]],
        [[False, True, True, True, True, True, True],
         [False, True, True, True, True, True, True],
         [False, True, True, True, True, True, True],
         [False, False, False, False, False, False, False]],
    )
    assert dependency is None and reconstruction == 0
    assert candidates.selected_positions[0, 0].tolist() == [1, 2, 4, 5]
    assert candidates.selected_positions[0, 1].tolist() == [2, -1, -1, -1]
    assert candidates.selected_positions[0, 2].tolist() == [1, -1, -1, -1]
    assert candidates.action_sizes.tolist() == [[4, 1, 1, 0]]
    assert candidates.metadata[0]["fallback_by_batch"] == (False, True, False, False)
    assert candidates.seed_anchors.tolist() == [[4, 2, 1, -1]]


def test_incoming_ranking_uses_uncertain_queries_below_admission_threshold(monkeypatch):
    dependency = _dependency([[[0, 0, 0], [0, 0, 0], [.1, .4, 0]]])
    monkeypatch.setattr(guided, "reconstruct_dependency_for_proposals", lambda *args, **kwargs: dependency)
    config = guided.DependencyGuidedSamplerConfig(
        proposal_strategy="confidence_threshold", confidence_ranking="incoming",
        dependency_max_action_size=1, dependency_action_sizes="1",
    )
    candidates, _, _, _ = _threshold(
        [[.95, .91, .2]], [[True, True, True]], config=config,
        entropy=torch.tensor([[0., 0., 2.]]),
    )
    assert candidates.selected_positions.tolist() == [[[1]]]
    assert candidates.proposal_scores.item() == pytest.approx(.91 * .4 * 2)
    assert guided.dependency_capture_required(config)
    assert not guided.dependency_capture_required(replace(config, confidence_ranking="confidence"))


def test_bfloat16_confidence_does_not_round_the_admission_threshold_down():
    confidence = torch.tensor([[.8984375, .90234375]], dtype=torch.bfloat16)
    active = torch.ones_like(confidence, dtype=torch.bool)
    config = guided.DependencyGuidedSamplerConfig(
        proposal_strategy="confidence_threshold", confidence_threshold=.9,
    )
    candidates, _, _, _ = guided.build_confidence_threshold_candidates(
        SimpleNamespace(structure=None, captures=None), active_mask=active,
        requested_k=torch.tensor([2]), anchor_state=None, response_mask=active,
        attention_mask=active, entropy_map=torch.zeros_like(confidence),
        confidence=confidence, config=config, generation_seed=42,
    )
    assert candidates.selected_positions.tolist() == [[[1, -1]]]
    assert candidates.action_sizes.tolist() == [[1]]
    assert candidates.metadata[0]["fallback_by_batch"] == (False,)


def test_preserved_mass_uses_conditional_reference_sinks_and_keeps_raw_scale(monkeypatch):
    reference = _dependency([[[0, .9, .1], [.1, 0, .9], [.1, .9, 0]]], renormalized=True)
    absolute = _dependency([[[0, .01, .08], [.01, 0, .08], [.01, .01, 0]]])
    calls = []

    def reconstruct(*args, **kwargs):
        calls.append(kwargs["renormalize_selected_keys"])
        return reference if kwargs["renormalize_selected_keys"] else absolute

    monkeypatch.setattr(guided, "build_active_dependency_matrix", reconstruct)
    config = guided.DependencyGuidedSamplerConfig(
        proposal_strategy="dependency", dependency_preserve_attention_mass=True,
        dependency_sink_quantile=.5,
    )
    output = guided.reconstruct_dependency_for_proposals(
        SimpleNamespace(structure=object(), captures={}),
        active_mask=torch.ones((1, 3), dtype=torch.bool),
        response_mask=torch.ones((1, 3), dtype=torch.bool),
        attention_mask=torch.ones((1, 3), dtype=torch.bool), config=config,
    )
    assert calls == [True, False]
    assert output.sink_mask.tolist() == [[False, True, False]]
    assert output.directed[0, 0, 2].item() == pytest.approx(.08)
    assert output.directed.sum(dim=-1).tolist()[0] == pytest.approx([.08, .09, .01])
    assert absolute.directed[0, 0, 1].item() == pytest.approx(.01)
    # Detecting sinks on the changed representation would remove another key.
    independent = filter_dependency_sinks(absolute, sink_quantile=.5, renormalize_rows=False)
    assert independent.sink_mask.tolist() == [[False, False, True]]


def test_affordable_growth_recomputes_conflict_from_the_actual_accepted_set():
    conflict = torch.zeros((5, 5))
    conflict[3, 2] = 2
    conflict[4, 1] = 100
    arguments = dict(
        seed=0, eligible_positions=list(range(5)), maximum_action_size=3,
        utility=torch.tensor([10., 9., 8., 7., 6.]),
        entropy=torch.tensor([.1, 1., .2, .2, .2]), confidence=torch.zeros(5),
        conflict=conflict, conflict_penalty=1., stopping_rule="entropy_budget",
        utility_threshold=0., entropy_budget=.55,
    )
    original = _construct_stopped_soft_full_subset(**arguments)
    explicit_original = _construct_stopped_soft_full_subset(**arguments, budget_search="first_unaffordable")
    affordable = _construct_stopped_soft_full_subset(**arguments, budget_search="best_affordable")
    assert original == explicit_original
    assert original[0] == [0]
    assert affordable[0] == [0, 2, 4]
    assert affordable[1] == pytest.approx([10., 8., 6.])
    assert affordable[2] == "maximum_action_size"
    over_budget = _construct_stopped_soft_full_subset(
        **{**arguments, "entropy_budget": .05}, budget_search="best_affordable"
    )
    assert over_budget[0] == [0]
    assert over_budget[2] == "entropy_budget"


def test_fixed_candidates_preserve_construction_order_separately_from_sorted_positions():
    candidates = generate_parallel_dependency_candidates(
        torch.zeros((1, 4, 4)), torch.ones((1, 4)),
        torch.tensor([[.4, .7, .9, .6]]), torch.ones((1, 4), dtype=torch.bool),
        requested_k=3, candidate_budget=1, variant="top_confidence",
    )
    assert candidates.selected_positions.tolist() == [[[1, 2, 3]]]
    assert candidates.metadata[0]["construction_order_by_batch"] == ((2, 1, 3),)
    assert candidates.metadata[0]["construction_order_position_space"] == "compact_active_response"
    assert candidates.metadata[0]["construction_order_kind_by_batch"] == ("greedy",)


@pytest.mark.parametrize("field,value", [
    ("confidence_threshold", -1), ("confidence_threshold", float("nan")),
    ("confidence_ranking", "outgoing"), ("commit_mode", "reverse"),
    ("dependency_preserve_attention_mass", 1), ("dependency_budget_search", "skip"),
])
def test_new_configuration_rejects_invalid_controls(field, value):
    config = guided.DependencyGuidedSamplerConfig()
    setattr(config, field, value)
    with pytest.raises((ValueError, TypeError)):
        guided.validate_dependency_guided_config(config)
