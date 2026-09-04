"""
Test proxy snapshot collection across deterministic masking stages.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_proxy_state_collection.py" -v
"""

import pytest
import torch

from dllm.core.samplers.proxy_diagnostics import (
    ProxyStateCollectionConfig,
    collect_llada_proxy_states,
    select_proxy_evaluation_pool,
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


def _make_zero_tiny_llada() -> LLaDAModelLM:
    """Construct a deterministic zero-logit four-layer LLaDA model."""
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_kv_heads=4,
        n_layers=4,
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
        mask_token_id=31,
        init_device="cpu",
    )
    model_config = create_model_config_from_pretrained_config(config)
    model_config.init_device = "cpu"
    core = LLaDAModel(model_config, init_params=True)
    model = LLaDAModelLM(config, model=core, init_params=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model


def _collect_tiny_states():
    model = _make_zero_tiny_llada()
    input_ids = torch.tensor([[1, 2, 31, 31, 31, 31, 31, 31, 31, 31]])
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.tensor(
        [[False, False, True, True, True, True, True, True, True, True]]
    )
    config = ProxyStateCollectionConfig(
        last_n_layers=2,
        top_confidence_pool_size=2,
        random_pool_size=2,
        seed=7,
        sink_quantile=0.99,
    )
    return collect_llada_proxy_states(
        model,
        input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        mask_token_id=31,
        example_id="tiny-0",
        checkpoint="constructed-zero-llada",
        config=config,
    )


def test_collects_reproducible_complete_snapshots_at_four_mask_ratios():
    first = _collect_tiny_states()
    second = _collect_tiny_states()

    assert first.to_dict() == second.to_dict()
    assert first.example_id == "tiny-0"
    assert first.response_length == 8
    assert first.prompt_token_count == 2
    assert [state.masked_count for state in first.states] == [8, 6, 4, 2]
    assert [state.actual_mask_ratio for state in first.states] == [
        1.0,
        0.75,
        0.5,
        0.25,
    ]

    for state in first.states:
        active_count = state.masked_count
        assert state.active_positions.numel() == active_count
        assert torch.equal(state.dependency_positions, state.active_positions)
        assert state.active_confidence.shape == (active_count,)
        assert state.active_entropy.shape == (active_count,)
        assert state.dependency_before_sink.shape == (
            active_count,
            active_count,
        )
        assert state.dependency_after_sink.shape == (
            active_count,
            active_count,
        )
        assert torch.isfinite(state.dependency_after_sink).all()
        assert state.capture_layer_ids == (2, 3)
        assert state.sink_mask is not None
        assert not state.sink_mask.any()

        pool_positions = state.evaluation_positions.tolist()
        assert len(pool_positions) == len(set(pool_positions))
        assert set(pool_positions).issubset(set(state.active_positions.tolist()))
        assert state.oracle_revealed_token_ids.shape == (
            state.evaluation_positions.shape
        )
        assert state.oracle_heldout_counts.tolist() == [
            active_count - 1
        ] * len(pool_positions)
        assert torch.count_nonzero(state.oracle_entropy_drop_sum) == 0
        assert torch.count_nonzero(state.oracle_risk_reduction_sum) == 0

    assert first.states[0].committed_response_positions.numel() == 0
    assert first.states[-1].committed_response_positions.numel() == 6
    serialized = first.to_dict()
    assert serialized["configuration"]["token_value_policy"] == "base_argmax"
    assert serialized["configuration"]["trajectory_policy"] == (
        "stagewise_highest_confidence"
    )
    assert serialized["states"][0]["dependency"]["direction_convention"]


def test_evaluation_pool_is_top_plus_seeded_random_without_duplicates():
    confidence = torch.tensor([0.0, 0.9, 0.2, 0.8, 0.1, 0.7])
    active_mask = torch.tensor([False, True, True, True, True, True])
    first_generator = torch.Generator(device="cpu").manual_seed(19)
    second_generator = torch.Generator(device="cpu").manual_seed(19)

    first = select_proxy_evaluation_pool(
        confidence,
        active_mask,
        top_confidence_pool_size=2,
        random_pool_size=2,
        generator=first_generator,
    )
    second = select_proxy_evaluation_pool(
        confidence,
        active_mask,
        top_confidence_pool_size=2,
        random_pool_size=2,
        generator=second_generator,
    )

    assert torch.equal(first.top_confidence_positions, torch.tensor([1, 3]))
    assert torch.equal(first.top_confidence_positions, second.top_confidence_positions)
    assert torch.equal(first.random_positions, second.random_positions)
    assert not set(first.top_confidence_positions.tolist()).intersection(
        first.random_positions.tolist()
    )
    assert first.positions.numel() == 4


def test_collection_requires_a_fully_masked_initial_response():
    model = _make_zero_tiny_llada()
    input_ids = torch.tensor([[1, 2, 31, 0, 31, 31, 31, 31]])
    response_mask = torch.tensor(
        [[False, False, True, True, True, True, True, True]]
    )

    with pytest.raises(ValueError, match="Every initial response position"):
        collect_llada_proxy_states(
            model,
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            response_mask=response_mask,
            mask_token_id=31,
            example_id="invalid",
            checkpoint="constructed-zero-llada",
            config=ProxyStateCollectionConfig(
                last_n_layers=2,
                top_confidence_pool_size=2,
                random_pool_size=2,
            ),
        )
