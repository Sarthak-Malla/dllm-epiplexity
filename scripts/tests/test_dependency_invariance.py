"""
Test exact LLaDA logit invariance around Q/K capture hooks.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    DEPENDENCY_TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${DEPENDENCY_TEST_ROOT}/test_dependency_invariance.py" -v
"""

import torch

from dllm.core.samplers.dependency import check_llada_capture_invariance
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


def _make_deterministic_llada() -> LLaDAModelLM:
    """Construct and initialize a deterministic four-layer LLaDA on CPU."""
    torch.manual_seed(1234)
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
        init_device="cpu",
    )
    model_config = create_model_config_from_pretrained_config(config)
    model_config.init_device = "cpu"
    core = LLaDAModel(model_config, init_params=True)
    model = LLaDAModelLM(config, model=core, init_params=False)
    model.eval()
    return model


def test_capture_is_bitwise_logit_invariant_and_leaves_model_state_unchanged():
    model = _make_deterministic_llada()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    result = check_llada_capture_invariance(
        model,
        input_ids,
        attention_mask=attention_mask,
        last_n_layers=2,
    )

    assert result.passed
    assert result.disabled_enabled_max_abs_diff == 0.0
    assert result.disabled_post_removal_max_abs_diff == 0.0
    assert result.layer_ids == (2, 3)
    assert tuple(item["layer_id"] for item in result.capture_metadata) == (2, 3)
    assert all(item["q_shape"] == (1, 4, 16) for item in result.capture_metadata)
    assert all(item["k_shape"] == (1, 4, 16) for item in result.capture_metadata)
    assert all(item["q_dtype"] == "torch.float32" for item in result.capture_metadata)
    assert all(item["q_device"] == "cpu" for item in result.capture_metadata)
