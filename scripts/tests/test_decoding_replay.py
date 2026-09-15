"""Check frozen decisions, RNG isolation, commitment order, and exact replay.

Run on a compute node after sourcing ~/.zshrc and activating the dllm environment:
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_replay.py
"""

import io
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dllm.core.samplers.decoding_state import DecodeState
from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.dependency_guided import (
    run_base_forward_with_cfg_inputs,
    select_fixed_k_candidate,
    validate_fixed_k_schedule,
)
from dllm.core.samplers.entropy_drop import EntropyDropSampler, EntropyDropSamplerConfig
from dllm.core.samplers.parallel_candidates import (
    initialize_committed_anchor_state,
    record_committed_anchors,
)
from dllm.core.samplers.utils import get_num_transfer_tokens
from test_dependency_guided_decoder import _make_tiny_llada, _tokenizer


class PrecedenceModel(nn.Module):
    """Two masked positions change their predictions in opposite reveal orders."""

    def __init__(self):
        super().__init__()
        self.register_buffer("device_marker", torch.zeros(()))
        self.seen_states = []

    @property
    def device(self):
        return self.device_marker.device

    def forward(self, input_ids, attention_mask=None):
        self.seen_states.append(input_ids.clone())
        predicted = torch.ones_like(input_ids)
        predicted[:, 1] = torch.where(input_ids[:, 2] == 7, 3, 5)
        predicted[:, 2] = torch.where(input_ids[:, 1] == 7, 4, 6)
        logits = torch.full((*input_ids.shape, 8), -4.0, device=input_ids.device)
        logits.scatter_(-1, predicted.unsqueeze(-1), 4.0)
        return SimpleNamespace(logits=logits)


def _case(**overrides):
    model = PrecedenceModel().eval()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=0, eos_token_id=2)
    sampler = EntropyDropSampler(model=model, tokenizer=tokenizer)
    settings = dict(
        max_new_tokens=4, block_size=2, steps=2, temperature=0.0,
        return_dict=True, oracle_candidate_strategy="greedy", suppress_tokens=[7],
    )
    settings.update(overrides)
    return sampler, EntropyDropSamplerConfig(**settings)


def _assert_same_trajectory(left, right):
    assert torch.equal(left.sequences, right.sequences)
    assert len(left.histories) == len(right.histories)
    assert all(torch.equal(a, b) for a, b in zip(left.histories, right.histories))


@torch.no_grad()
def _previous_fixed_loop(sampler, prompt, config):
    """Independent pre-refactor block loop for deterministic, single-row cases.

    Deliberately does not call initialize/prepare/select/commit/continue. Keeping
    the original scheduling and anchor bookkeeping here tests migration parity,
    while the production implementation has only one decoding loop.
    """
    assert config.temperature == config.cfg_scale == 0.0
    assert config.dependency_cardinality_strategy == "fixed"
    mask_id = sampler.tokenizer.mask_token_id
    prompt_length = len(prompt)
    sequence_length = prompt_length + config.max_new_tokens
    x = torch.full((1, sequence_length), mask_id, dtype=torch.long)
    x[0, :prompt_length] = torch.tensor(prompt)
    attention_mask = torch.ones_like(x)
    response_mask = torch.zeros_like(x, dtype=torch.bool)
    response_mask[:, prompt_length:] = True
    num_blocks = math.ceil(config.max_new_tokens / config.block_size)
    steps = math.ceil(config.steps / num_blocks)
    histories, actions, anchor_counts = [x.clone()], [], []
    global_step = 0
    for block in range(num_blocks):
        anchors = (
            initialize_committed_anchor_state(
                x, confidence_threshold=config.dependency_anchor_confidence_threshold
            )
            if config.proposal_strategy == "dependency" else None
        )
        start = prompt_length + block * config.block_size
        end = min(start + config.block_size, sequence_length)
        block_span = torch.zeros_like(x, dtype=torch.bool)
        block_span[:, start:end] = True
        initial_block = torch.zeros((1, config.block_size), dtype=torch.bool)
        initial_block[:, :end - start] = x[:, start:end] == mask_id
        schedule = get_num_transfer_tokens(
            mask_index=initial_block, steps=steps, scheduler=sampler.scheduler,
            stochastic=False,
        )
        if config.proposal_strategy == "dependency":
            validate_fixed_k_schedule(schedule, config.dependency_commit_k)
        for step in range(schedule.shape[1]):
            requested = schedule[:, step]
            if not bool(torch.any(requested)):
                continue
            masked = x == mask_id
            active = masked & block_span
            base = run_base_forward_with_cfg_inputs(
                sampler.model, x, attention_mask, cfg_scale=0.0,
                unconditional_input_ids=None,
                capture_dependency=config.proposal_strategy == "dependency",
                dependency_last_n_layers=config.dependency_last_n_layers,
                measure_timing=False,
            )
            logits = base.logits
            for token_id in config.suppress_tokens or []:
                logits[:, :, token_id] = -torch.inf
            predicted = logits.argmax(-1)
            probabilities = F.softmax(logits, dim=-1)
            confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            entropy = sampler.get_entropy_per_token(logits)
            if config.proposal_strategy == "legacy":
                candidate_masks = sampler.generate_candidate_sets(
                    confidence=torch.where(active, confidence, -torch.inf),
                    mask_idx=active, num_transfer=requested,
                    strategy=config.oracle_candidate_strategy,
                )
                chosen, _, _ = sampler._select_best_candidate(
                    x, predicted, masked, candidate_masks, attention_mask, entropy,
                    candidate_chunk_size=config.candidate_chunk_size,
                )
            else:
                selection = select_fixed_k_candidate(
                    sampler.model, x, predicted, base_forward=base,
                    base_metric_map=entropy, entropy_map=entropy, confidence=confidence,
                    metric="entropy_drop", active_mask=active, requested_k=requested,
                    anchor_state=anchors, masked_active_mask=masked,
                    response_mask=response_mask, attention_mask=attention_mask,
                    config=config,
                    generation_seed=config.dependency_generation_seed + global_step,
                )
                chosen = selection.lookahead.best_mask
                anchor_counts.append(int(anchors.committed_position_mask.sum()))
                anchors = record_committed_anchors(
                    anchors, chosen, confidence, predicted, commit_step=global_step
                )
            actions.append(torch.where(chosen[0])[0].tolist())
            x[chosen] = predicted[chosen]
            histories.append(x.clone())
            global_step += 1
    return BaseSamplerOutput(sequences=x, histories=histories), actions, anchor_counts


@pytest.mark.parametrize("proposal", ["legacy", "dependency"])
def test_refactor_matches_previous_loop_with_clipped_final_block(proposal):
    sampler = EntropyDropSampler(model=_make_tiny_llada(), tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=9, block_size=8, steps=4, temperature=0.0,
        return_dict=True, proposal_strategy=proposal, dependency_commit_k=4,
        dependency_last_n_layers=2, dependency_sink_filter_enabled=False,
        dependency_anchor_confidence_threshold=0.0,
        candidate_budget=3, candidate_chunk_size=1, suppress_tokens=[31],
        dependency_size_scoring="per_token", diagnostic_metadata=proposal == "dependency",
    )
    torch.manual_seed(119)
    previous, actions, anchor_counts = _previous_fixed_loop(sampler, [3, 4], config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(119)
    actual = sampler.sample([[3, 4]], config)
    _assert_same_trajectory(previous, actual)
    actual_actions = [
        torch.where((before[0] == 31) & (after[0] != 31))[0].tolist()
        for before, after in zip(actual.histories, actual.histories[1:])
    ]
    assert actual_actions == actions
    assert [len(action) for action in actions] == [4, 4, 1]
    assert torch.equal(expected_rng, torch.get_rng_state())
    if proposal == "dependency":
        assert anchor_counts == [0, 4, 0]
        assert [record["committed_anchor_count_before"]
                for record in actual.diagnostics[0]] == anchor_counts


def test_snapshot_resume_preserves_blocks_and_complete_trajectory():
    sampler, config = _case()
    reference = sampler.sample([[0]], config=config)
    state = sampler.initialize_state([[0]], config)
    first = sampler.prepare_step(state)
    selected = sampler.select_step(first)
    sampler.commit_step(state, first, selected.best_mask, selection=selected)
    assert state.step_index == state.effective_steps
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    buffer.seek(0)
    restored = DecodeState.from_state_dict(torch.load(buffer, weights_only=True))
    replay = sampler.continue_from_state(restored)
    _assert_same_trajectory(reference, replay)
    assert restored.done


def test_repeated_preparation_and_observer_preserve_stochastic_carrier():
    sampler, config = _case(temperature=0.7, oracle_candidate_strategy="random")
    torch.manual_seed(711)
    reference = sampler.sample([[0]], config=config)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(711)
    observed = []

    def observer(prepared):
        observed.append(prepared.state.global_step_index)
        # Deliberate random draws and extra selections must not perturb decoding.
        torch.rand(23)
        sampler.select_step(prepared, "min_entropy")

    instrumented = sampler.sample([[0]], config=config, observer=observer)
    _assert_same_trajectory(reference, instrumented)
    assert observed == [0, 1]
    assert torch.equal(reference_rng, torch.get_rng_state())
    state = sampler.initialize_state([[0]], config)
    first = sampler.prepare_step(state)
    second = sampler.prepare_step(state)
    assert torch.equal(first.x0, second.x0)
    assert torch.equal(first.candidates.candidate_masks, second.candidates.candidate_masks)
    assert state.global_step_index == 0


def test_seed_and_reverse_probes_refresh_only_masked_companions():
    sampler, config = _case()
    state = sampler.initialize_state([[0]], config)
    prepared = sampler.prepare_step(state)
    original = state.input_ids.clone()
    original_rng = torch.get_rng_state().clone()
    seed_first = sampler.probe_seed_first(prepared, 0)
    assert sampler.model.seen_states[-1][0, 1:3].tolist() == [3, 7]
    assert seed_first.token_ids[0, 1:3].tolist() == [3, 6]
    assert seed_first.flipped_mask[0, 1:3].tolist() == [False, True]
    reverse = sampler.probe_seed_first(prepared, 0, reverse=True)
    assert sampler.model.seen_states[-1][0, 1:3].tolist() == [7, 4]
    assert reverse.token_ids[0, 1:3].tolist() == [5, 4]
    assert reverse.flipped_mask[0, 1:3].tolist() == [True, False]
    assert torch.equal(original, state.input_ids)
    assert torch.equal(original_rng, torch.get_rng_state())
    assert state.global_step_index == 0
    selected = sampler.select_step(prepared)
    outcomes = []
    for values in (prepared.x0, seed_first.token_ids, reverse.token_ids):
        branch = state.clone()
        sampler.commit_step(branch, prepared, selected.best_mask, token_ids=values)
        assert branch.global_step_index == 1
        assert branch.step_index == 1
        outcomes.append(sampler.continue_from_state(branch).sequences[0, 1:3].tolist())
    assert outcomes == [[3, 4], [3, 6], [5, 4]]


def test_probe_keeps_logit_argmax_when_bf16_probabilities_tie():
    class RoundedProbabilityModel(PrecedenceModel):
        def forward(self, input_ids, attention_mask=None):
            logits = super().forward(input_ids, attention_mask).logits.to(torch.bfloat16)
            for row in range(input_ids.shape[0]):
                if input_ids[row, 1] != 7:
                    logits[row, 2].fill_(-10.0)
                    logits[row, 2, 3] = 0.0
                    logits[row, 2, 4] = 0.001
            return SimpleNamespace(logits=logits)

    sampler, config = _case()
    sampler.model = RoundedProbabilityModel().eval()
    state = sampler.initialize_state([[0]], config)
    prepared = sampler.prepare_step(state)
    values = sampler.probe_seed_first(prepared, 0)
    probabilities = values.refreshed_probabilities[0, 2]
    assert probabilities[3] == probabilities[4]
    assert probabilities.argmax().item() == 3
    assert prepared.x0[0, 2].item() == 4
    assert values.refreshed_token_ids[0, 2].item() == 4
    assert values.token_ids[0, 2].item() == 4
    assert not values.flipped_mask[0, 2]


def test_singleton_probe_has_no_forward_and_reverse_rejects_it():
    sampler, config = _case(steps=4)
    state = sampler.initialize_state([[0]], config)
    prepared = sampler.prepare_step(state)
    calls = len(sampler.model.seen_states)
    values = sampler.probe_seed_first(prepared, 0)
    assert len(sampler.model.seen_states) == calls
    assert not values.companion_mask.any()
    with pytest.raises(ValueError, match="two-token"):
        sampler.probe_seed_first(prepared, 0, reverse=True)


def test_selectors_share_candidate_pool_and_entropy_scores_are_cached():
    sampler, config = _case(oracle_candidate_strategy="mixed", steps=4)
    state = sampler.initialize_state([[0]], config)
    prepared = sampler.prepare_step(state)
    entropy = sampler.select_step(prepared, "entropy_drop")
    count = len(sampler.model.seen_states)
    for selector in ("max_confidence", "min_entropy", "min_top2_margin"):
        cheap = sampler.select_step(prepared, selector)
        assert cheap.detail.candidates is entropy.detail.candidates
    assert sampler.select_step(prepared, "entropy_drop") is entropy
    assert len(sampler.model.seen_states) == count


def test_seed_first_deployment_uses_shared_commit_and_records_refreshed_values():
    sampler, config = _case(commit_mode="seed_first")
    output = sampler.sample([[0]], config=config)
    assert output.sequences[0, 1:3].tolist() == [3, 6]
    assert len(output.histories) == 3


def test_diagnostic_commit_freezes_anchor_confidence_and_deployment_can_refresh():
    sampler, config = _case()
    state = sampler.initialize_state([[0]], config)
    prepared = sampler.prepare_step(state)
    # Anchor storage is independent of attention capture; attach an empty history
    # to the toy case to isolate bookkeeping from a neural attention model.
    anchors = initialize_committed_anchor_state(state.input_ids, confidence_threshold=0.8)
    state.anchor_state = anchors
    prepared.state.anchor_state = anchors
    prepared.confidence.fill_(0.2)
    values = sampler.probe_seed_first(prepared, 0)
    selected = sampler.select_step(prepared)
    frozen, refreshed = state.clone(), state.clone()
    sampler.commit_step(frozen, prepared, selected.best_mask, token_ids=values.token_ids)
    sampler.commit_step(
        refreshed, prepared, selected.best_mask, token_ids=values.token_ids,
        confidence=values.confidence,
    )
    assert torch.equal(frozen.input_ids, refreshed.input_ids)
    assert not frozen.anchor_state.reliable_anchor_mask.any()
    assert refreshed.anchor_state.reliable_anchor_mask[0, 2]
    assert frozen.global_step_index == refreshed.global_step_index == 1


def test_tiny_llada_dependency_snapshot_retains_anchor_history_and_fixed_cheap_support():
    sampler = EntropyDropSampler(model=_make_tiny_llada(), tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=4, block_size=4, steps=2, temperature=0.0,
        return_dict=True, proposal_strategy="dependency", dependency_commit_k=2,
        dependency_last_n_layers=2, dependency_sink_filter_enabled=False,
        dependency_candidate_selector="max_confidence", diagnostic_metadata=True,
        dependency_anchor_confidence_threshold=0.0,
    )
    reference = sampler.sample([[3, 4]], config)
    state = sampler.initialize_state([[3, 4]], config)
    prepared = sampler.prepare_step(state)
    selected = sampler.select_step(prepared)
    sampler.commit_step(state, prepared, selected.best_mask, selection=selected)
    restored = DecodeState.from_state_dict(state.state_dict())
    assert restored.anchor_state.committed_position_mask.sum() == 2
    replay = sampler.continue_from_state(restored)
    _assert_same_trajectory(reference, replay)
    assert all(step["lookahead_model_calls"] == 0 for step in replay.diagnostics[0])


def test_threshold_sampler_uses_one_base_forward_and_guarantees_progress():
    sampler, config = _case(
        proposal_strategy="confidence_threshold", confidence_threshold=1.0,
        dependency_max_action_size=2, dependency_action_sizes="1|2",
        diagnostic_metadata=True,
    )
    output = sampler.sample([[0]], config)
    assert len(sampler.model.seen_states) == 4
    assert all(step["commit_k"] == 1 for step in output.diagnostics[0])
    assert all(step["candidate_selector"] == "direct" for step in output.diagnostics[0])
    assert not (output.sequences[:, 1:] == 7).any()
