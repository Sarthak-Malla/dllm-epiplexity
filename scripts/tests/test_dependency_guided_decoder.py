"""Test fixed-k dependency decoding on a tiny CPU LLaDA.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py -v
"""

import math
from types import SimpleNamespace

import pytest
import torch

from dllm.core.samplers.dependency import resolve_llada_attention_structure
from dllm.core.samplers.dependency_non_lookahead import (
    DependencyNonLookaheadSampler,
    DependencyNonLookaheadSamplerConfig,
)
from dllm.core.samplers.dependency_guided import (
    DependencyGuidedSamplerConfig,
    SUPPORTED_PROPOSAL_STRATEGIES,
    resolve_dependency_guided_config,
    validate_dependency_guided_config,
    validate_fixed_k_schedule,
)
from dllm.core.samplers.entropy_drop import (
    EntropyDropSampler,
    EntropyDropSamplerConfig,
)
from dllm.core.samplers.risk_reduction import (
    RiskReductionSampler,
    RiskReductionSamplerConfig,
)
from dllm.pipelines.llada.models.configuration_llada import (
    ActivationType,
    BlockType,
    LLaDAConfig,
)
from dllm.pipelines.llada.models.modeling_llada import (
    LLaDAModel,
    LLaDAModelLM,
    create_model_config_from_pretrained_config,
)


def _make_tiny_llada() -> LLaDAModelLM:
    """Construct a deterministic two-layer CPU model for decoder tests."""
    torch.manual_seed(1234)
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=32,
        activation_type=ActivationType.silu,
        block_type=BlockType.llama,
        block_group_size=1,
        attention_layer_norm=False,
        rope=True,
        max_sequence_length=16,
        vocab_size=32,
        embedding_size=32,
        weight_tying=False,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        init_device="cpu",
    )
    model_config = create_model_config_from_pretrained_config(config)
    model_config.init_device = "cpu"
    core = LLaDAModel(model_config, init_params=True)
    model = LLaDAModelLM(config, model=core, init_params=False)
    model.eval()
    return model


def _tokenizer() -> SimpleNamespace:
    """Return the token IDs required by the sampler interface."""
    return SimpleNamespace(mask_token_id=31, bos_token_id=1, eos_token_id=2)


def test_feature_flag_defaults_to_legacy_and_unknown_strategy_fails_early():
    config = DependencyGuidedSamplerConfig()
    assert config.proposal_strategy == "legacy"
    assert "dependency" in SUPPORTED_PROPOSAL_STRATEGIES
    validate_dependency_guided_config(config)

    config.proposal_strategy = "not_a_strategy"
    with pytest.raises(ValueError, match="Available:.*dependency"):
        validate_dependency_guided_config(config)

    with pytest.raises(ValueError, match="requires return_dict"):
        validate_dependency_guided_config(
            DependencyGuidedSamplerConfig(diagnostic_metadata=True)
        )


def test_fixed_k_schedule_rejects_multi_token_commits():
    validate_fixed_k_schedule(torch.tensor([[1, 1, 0]]))
    with pytest.raises(ValueError, match="exactly one token"):
        validate_fixed_k_schedule(torch.tensor([[1, 2]]))


@pytest.mark.parametrize(
    ("commit_k", "schedule"),
    (
        (2, torch.tensor([[2, 2, 1, 0]])),
        (4, torch.tensor([[4, 4, 3, 0]])),
    ),
)
def test_parallel_fixed_k_schedule_allows_only_a_clipped_final_action(
    commit_k,
    schedule,
):
    validate_fixed_k_schedule(schedule, commit_k)
    invalid = schedule.clone()
    invalid[0, 0] = commit_k - 1
    with pytest.raises(ValueError, match=f"exactly {commit_k} tokens"):
        validate_fixed_k_schedule(invalid, commit_k)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dependency_commit_k", 0, "positive integer"),
        ("dependency_parallel_variant", "unknown", "parallel_variant"),
        ("dependency_conflict_normalization", "unknown", "normalization"),
        ("dependency_conflict_penalty", -1.0, "nonnegative"),
        ("dependency_anchor_confidence_threshold", 1.1, "in \\[0,1\\]"),
    ),
)
def test_parallel_configuration_rejects_invalid_values(field, value, message):
    config = DependencyGuidedSamplerConfig()
    setattr(config, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_dependency_guided_config(config)


def test_fixed_k_defaults_to_sequential_candidate_width_without_changing_legacy():
    legacy = resolve_dependency_guided_config(
        DependencyGuidedSamplerConfig(),
        {},
    )
    fixed_k = resolve_dependency_guided_config(
        DependencyGuidedSamplerConfig(
            proposal_strategy="dependency",
            return_dict=True,
        ),
        {},
    )

    assert legacy.candidate_chunk_size is None
    assert fixed_k.candidate_chunk_size == 1


@pytest.mark.parametrize(
    ("sampler_class", "config_class", "metric"),
    (
        (EntropyDropSampler, EntropyDropSamplerConfig, "entropy_drop"),
        (RiskReductionSampler, RiskReductionSamplerConfig, "risk_reduction"),
    ),
)
def test_dependency_k1_path_completes_and_emits_auditable_steps(
    sampler_class,
    config_class,
    metric,
):
    model = _make_tiny_llada()
    structure = resolve_llada_attention_structure(model, last_n_layers=2)
    baseline_hook_counts = tuple(
        (len(layer.q_proj._forward_hooks), len(layer.k_proj._forward_hooks))
        for layer in structure.selected_layers
    )
    sampler = sampler_class(model=model, tokenizer=_tokenizer())
    config = config_class(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=3,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.7,
        dependency_generation_seed=19,
        dependency_sink_filter_enabled=False,
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.sequences.shape == (1, 6)
    assert not torch.any(output.sequences[:, 2:] == _tokenizer().mask_token_id)
    assert output.histories is not None and len(output.histories) == 5
    assert output.diagnostics is not None and len(output.diagnostics[0]) == 4
    assert output.selected_candidates
    for step, record in enumerate(output.diagnostics[0]):
        assert record["verifier_metric"] == metric
        assert record["global_step_index"] == step
        assert record["commit_k"] == 1
        assert record["immediate_token_consistency_total"] == (0 if step == 0 else 1)
        assert record["candidate_count_realized"] == min(3, 4 - step)
        assert record["captured_base_forward_count"] == 1
        assert record["lookahead_capture_forward_count"] == 0
        assert record["capture_active_before_lookahead"] is False
        assert record["capture_tensors_released_before_lookahead"] is True
        assert record["dependency_capture_source"] == "conditional"
        assert record["capture_layers"] == [0, 1]
        assert record["selected_candidate"] is not None
        assert len(record["selected_candidate"]["positions"]) == 1
        assert all(
            candidate["source"] is None
            or "dependency" in candidate["source"]
            for candidate in record["candidates"]
        )

    final_hook_counts = tuple(
        (len(layer.q_proj._forward_hooks), len(layer.k_proj._forward_hooks))
        for layer in structure.selected_layers
    )
    assert final_hook_counts == baseline_hook_counts


def test_phase6_controls_do_not_change_the_k1_dependency_path():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    common = dict(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=3,
        dependency_commit_k=1,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.7,
        dependency_generation_seed=31,
        dependency_sink_filter_enabled=False,
        diagnostic_metadata=True,
    )
    reference = sampler.sample([[3, 4]], config=EntropyDropSamplerConfig(**common))
    configured = sampler.sample(
        [[3, 4]],
        config=EntropyDropSamplerConfig(
            **common,
            dependency_parallel_variant="correlated_together",
            dependency_conflict_normalization="mean_positive",
            dependency_conflict_penalty=9.0,
            dependency_hard_conflict_threshold=0.0,
            dependency_anchor_support_weight=7.0,
            dependency_anchor_confidence_threshold=0.0,
        ),
    )

    assert torch.equal(reference.sequences, configured.sequences)
    assert reference.diagnostics is not None
    assert configured.diagnostics is not None
    assert [
        step["selected_candidate"]["positions"]
        for step in reference.diagnostics[0]
    ] == [
        step["selected_candidate"]["positions"]
        for step in configured.diagnostics[0]
    ]


def test_parallel_anchor_history_resets_at_each_decoder_block():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=4,
        block_size=2,
        steps=2,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=2,
        dependency_commit_k=2,
        dependency_parallel_variant="soft_full",
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_sink_filter_enabled=False,
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    assert [step["block_index"] for step in output.diagnostics[0]] == [0, 1]
    assert [
        step["committed_anchor_count_before"] for step in output.diagnostics[0]
    ] == [0, 0]
    assert [
        step["committed_anchor_count_after"] for step in output.diagnostics[0]
    ] == [2, 2]


@pytest.mark.parametrize(
    ("sampler_class", "config_class", "metric"),
    (
        (EntropyDropSampler, EntropyDropSamplerConfig, "entropy_drop"),
        (RiskReductionSampler, RiskReductionSamplerConfig, "risk_reduction"),
    ),
)
@pytest.mark.parametrize(
    ("commit_k", "steps", "variant"),
    (
        (2, 2, "soft_full"),
        (4, 1, "hard_low_conflict"),
    ),
)
def test_parallel_dependency_path_is_exact_valid_and_reproducible(
    sampler_class,
    config_class,
    metric,
    commit_k,
    steps,
    variant,
):
    model = _make_tiny_llada()
    sampler = sampler_class(model=model, tokenizer=_tokenizer())
    config = config_class(
        max_new_tokens=4,
        block_size=4,
        steps=steps,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=3,
        dependency_commit_k=commit_k,
        dependency_parallel_variant=variant,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.7,
        dependency_generation_seed=71,
        dependency_sink_filter_enabled=False,
        dependency_anchor_confidence_threshold=0.0,
        diagnostic_metadata=True,
    )

    first = sampler.sample([[3, 4]], config=config)
    second = sampler.sample([[3, 4]], config=config)

    assert torch.equal(first.sequences, second.sequences)
    assert first.histories is not None and len(first.histories) == steps + 1
    assert first.diagnostics is not None
    assert second.diagnostics is not None
    assert len(first.diagnostics[0]) == steps
    response_positions = {2, 3, 4, 5}
    revealed_positions: set[int] = set()
    remaining = 4
    first_selected_positions = []
    second_selected_positions = []
    for record_index, (first_record, second_record) in enumerate(zip(
        first.diagnostics[0], second.diagnostics[0], strict=True
    )):
        expected_candidates = min(
            config.candidate_budget,
            math.comb(remaining, commit_k),
        )
        assert first_record["verifier_metric"] == metric
        assert first_record["dependency_parallel_variant"] == variant
        assert first_record["commit_k"] == commit_k
        assert first_record["immediate_token_consistency_total"] == (
            0 if record_index == 0 else commit_k
        )
        assert first_record["candidate_count_realized"] == expected_candidates
        assert first_record["candidate_collapse"] is False
        assert first_record["committed_anchor_count_after"] == 4 - remaining + commit_k
        assert first_record["reliable_anchor_count_after"] == 4 - remaining + commit_k
        selected = first_record["selected_candidate"]
        assert selected is not None
        assert len(selected["positions"]) == commit_k
        assert len(set(selected["positions"])) == commit_k
        assert set(selected["positions"]) <= response_positions
        if remaining < 4:
            assert selected["anchor_support_sum"] > 0
        for candidate in first_record["candidates"]:
            if not candidate["valid"]:
                continue
            assert len(candidate["positions"]) == commit_k
            assert candidate["heldout_count"] == remaining - commit_k
            assert set(candidate["positions"]).isdisjoint(revealed_positions)
        first_selected_positions.append(tuple(selected["positions"]))
        second_selected_positions.append(
            tuple(second_record["selected_candidate"]["positions"])
        )
        revealed_positions.update(selected["positions"])
        remaining -= commit_k
    assert first_selected_positions == second_selected_positions


def test_cfg_records_conditional_half_as_dependency_source():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        cfg_scale=0.5,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=2,
        dependency_last_n_layers=1,
        dependency_position_temperature=0.0,
        dependency_sink_filter_enabled=False,
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    assert all(
        record["dependency_capture_source"] == "conditional_cfg_half"
        and record["capture_batch_size"] == 2
        and record["capture_layers"] == [1]
        for record in output.diagnostics[0]
    )


@pytest.mark.parametrize(
    "strategy",
    (
        "baseline_random",
        "baseline_current_mixed",
        "baseline_confidence_gumbel",
    ),
)
def test_comparison_controls_are_explicit_and_do_not_capture_attention(strategy):
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        return_dict=True,
        proposal_strategy=strategy,
        candidate_budget=2,
        dependency_position_temperature=0.7,
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3]], config=config)

    assert output.diagnostics is not None
    assert all(
        record["proposal_strategy"].startswith("baseline_")
        and record["dependency_capture_source"] is None
        and record["capture_layers"] == []
        for record in output.diagnostics[0]
    )


def test_adaptive_configuration_requires_dependency_and_size_aware_scoring():
    with pytest.raises(ValueError, match="proposal_strategy='dependency'"):
        validate_dependency_guided_config(
            DependencyGuidedSamplerConfig(
                dependency_cardinality_strategy="joint_k",
                dependency_size_scoring="per_token",
            )
        )
    with pytest.raises(ValueError, match="size-aware scoring"):
        validate_dependency_guided_config(
            DependencyGuidedSamplerConfig(
                proposal_strategy="dependency",
                dependency_cardinality_strategy="joint_k",
            )
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_dependency_guided_config(
            DependencyGuidedSamplerConfig(
                dependency_max_action_size=2,
                dependency_action_sizes="1|2|4",
            )
        )


@pytest.mark.parametrize(
    ("sampler_class", "config_class", "metric"),
    (
        (EntropyDropSampler, EntropyDropSamplerConfig, "entropy_drop"),
        (RiskReductionSampler, RiskReductionSamplerConfig, "risk_reduction"),
    ),
)
def test_marginal_utility_path_guarantees_single_position_progress(
    sampler_class,
    config_class,
    metric,
):
    model = _make_tiny_llada()
    sampler = sampler_class(model=model, tokenizer=_tokenizer())
    config = config_class(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=2,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_generation_seed=91,
        dependency_sink_filter_enabled=False,
        dependency_anchor_confidence_threshold=0.0,
        dependency_cardinality_strategy="marginal_utility",
        dependency_max_action_size=4,
        dependency_utility_threshold=1e9,
        dependency_size_scoring="per_token",
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    assert len(output.diagnostics[0]) == 4
    assert not torch.any(output.sequences[:, 2:] == _tokenizer().mask_token_id)
    for step, record in enumerate(output.diagnostics[0]):
        assert record["verifier_metric"] == metric
        assert record["cardinality_strategy"] == "marginal_utility"
        assert record["size_scoring_rule"] == "per_token"
        assert record["commit_k"] == 1
        assert record["candidate_collapse"] is False
        assert record["immediate_token_consistency_total"] == (0 if step == 0 else 1)
        selected = record["selected_candidate"]
        assert selected["action_size"] == 1
        assert selected["raw_verifier_score"] is not None
        assert selected["size_aware_verifier_score"] is not None


def test_joint_k_path_compares_sizes_and_finishes_without_invalid_actions():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=1,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_generation_seed=101,
        dependency_sink_filter_enabled=False,
        dependency_anchor_confidence_threshold=0.0,
        dependency_cardinality_strategy="joint_k",
        dependency_max_action_size=4,
        dependency_action_sizes="1|2|4",
        dependency_size_scoring="per_token",
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    records = output.diagnostics[0]
    assert records
    assert sum(record["commit_k"] for record in records) == 4
    assert not torch.any(output.sequences[:, 2:] == _tokenizer().mask_token_id)
    for record in records:
        assert record["cardinality_strategy"] == "joint_k"
        assert record["candidate_budget_semantics"] == "per_action_size"
        assert record["commit_k"] in {1, 2, 4}
        assert record["candidate_collapse"] is False
        assert record["selected_candidate"]["action_size"] == record["commit_k"]
        assert all(
            candidate["action_size"] in {0, 1, 2, 4}
            for candidate in record["candidates"]
        )


def test_existing_scheduler_control_uses_soft_full_positions_at_requested_size():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=2,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=2,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_sink_filter_enabled=False,
        dependency_cardinality_strategy="scheduler",
        dependency_size_scoring="raw",
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    assert [record["commit_k"] for record in output.diagnostics[0]] == [2, 2]
    assert all(
        record["cardinality_strategy"] == "scheduler"
        and record["dependency_parallel_variant"] == "soft_full"
        and record["candidate_collapse"] is False
        for record in output.diagnostics[0]
    )


@pytest.mark.parametrize(
    "candidate_selector",
    ("max_confidence", "min_entropy", "min_top2_margin"),
)
def test_non_lookahead_dependency_selector_uses_only_base_forwards(
    candidate_selector,
):
    model = _make_tiny_llada()
    forward_calls = 0

    def count_forward_calls(_module, _inputs, _output):
        nonlocal forward_calls
        forward_calls += 1

    hook = model.register_forward_hook(count_forward_calls)
    sampler = DependencyNonLookaheadSampler(model=model, tokenizer=_tokenizer())
    config = DependencyNonLookaheadSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=2,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=3,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_sink_filter_enabled=False,
        dependency_cardinality_strategy="entropy_budget",
        dependency_max_action_size=4,
        dependency_entropy_budget=2.0,
        dependency_size_scoring="per_token",
        dependency_candidate_selector=candidate_selector,
        diagnostic_metadata=False,
    )

    try:
        output = sampler.sample([[3, 4]], config=config)
    finally:
        hook.remove()

    assert output.histories is not None
    assert forward_calls == len(output.histories) - 1
    assert 1 <= forward_calls <= 4
    assert output.selected_candidates
    assert not torch.any(output.sequences[:, 2:] == _tokenizer().mask_token_id)


def test_entropy_budget_path_is_wired_into_decoder_and_guarantees_progress():
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    config = EntropyDropSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        temperature=0.0,
        return_dict=True,
        proposal_strategy="dependency",
        candidate_budget=2,
        dependency_last_n_layers=2,
        dependency_position_temperature=0.0,
        dependency_sink_filter_enabled=False,
        dependency_cardinality_strategy="entropy_budget",
        dependency_max_action_size=4,
        dependency_entropy_budget=0.0,
        dependency_size_scoring="per_token",
        diagnostic_metadata=True,
    )

    output = sampler.sample([[3, 4]], config=config)

    assert output.diagnostics is not None
    assert [record["commit_k"] for record in output.diagnostics[0]] == [1, 1, 1, 1]
    assert all(
        record["cardinality_strategy"] == "entropy_budget"
        and record["candidate_collapse"] is False
        and record["selected_candidate"]["stopping_reason"]
        in {"entropy_budget", "eligible_exhausted", "maximum_action_size"}
        for record in output.diagnostics[0]
    )
    assert not torch.any(output.sequences[:, 2:] == _tokenizer().mask_token_id)
