"""
Test LLaDA runtime-structure validation for dependency attention capture.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_structure.py -v
"""

import pytest
import torch.nn as nn

from dllm.core.samplers.dependency import (
    UnsupportedDependencyModelError,
    resolve_llada_attention_structure,
)
from dllm.pipelines.llada.models.configuration_llada import (
    BlockType,
    LLaDAConfig,
)
from dllm.pipelines.llada.models.modeling_llada import (
    LLaDALlamaBlock,
    LLaDAModel,
    LLaDAModelLM,
    create_model_config_from_pretrained_config,
)


def _make_tiny_llada() -> LLaDAModelLM:
    """Construct a supported four-layer LLaDA wrapper entirely on CPU."""
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_kv_heads=4,
        n_layers=4,
        mlp_hidden_size=32,
        block_type=BlockType.llama,
        block_group_size=1,
        attention_layer_norm=False,
        rope=True,
        max_sequence_length=16,
        vocab_size=32,
        embedding_size=32,
        weight_tying=False,
        init_device="cpu",
    )
    model_config = create_model_config_from_pretrained_config(config)
    model_config.init_device = "cpu"
    core = LLaDAModel(model_config, init_params=False)
    return LLaDAModelLM(config, model=core, init_params=False)


def test_resolves_supported_llada_structure_and_last_layers():
    model = _make_tiny_llada()

    structure = resolve_llada_attention_structure(model, last_n_layers=2)

    assert isinstance(structure.wrapper, LLaDAModelLM)
    assert isinstance(structure.core, LLaDAModel)
    assert structure.layer_ids == (2, 3)
    assert len(structure.selected_layers) == 2
    assert all(
        isinstance(layer, LLaDALlamaBlock) for layer in structure.selected_layers
    )
    assert all(layer.q_proj is not layer.k_proj for layer in structure.selected_layers)
    assert structure.q_norm_present == (False, False)
    assert structure.k_norm_present == (False, False)
    assert structure.rotary_emb_present == (True, True)
    assert structure.n_heads == 4
    assert structure.n_kv_heads == 4
    assert structure.head_dim == 4
    assert structure.kv_head_repeat == 1
    assert structure.q_projection_size == 16
    assert structure.k_projection_size == 16


def test_rejects_unsupported_model_with_useful_message():
    class UnsupportedModel(nn.Module):
        pass

    with pytest.raises(
        UnsupportedDependencyModelError,
        match="Dependency capture supports only LLaDAModelLM; got UnsupportedModel",
    ):
        resolve_llada_attention_structure(UnsupportedModel())


@pytest.mark.parametrize("last_n_layers", [0, 5, True, 1.5])
def test_rejects_invalid_layer_selection(last_n_layers):
    model = _make_tiny_llada()

    with pytest.raises(ValueError, match="last_n_layers"):
        resolve_llada_attention_structure(model, last_n_layers=last_n_layers)
