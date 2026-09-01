"""
Test active-response dependency restriction and aggregation.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest \
        /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_aggregation.py \
        -v
"""

import torch

from dllm.core.samplers.dependency import (
    build_active_dependency_matrix,
    reconstruct_llada_attention,
    resolve_llada_attention_structure,
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
    """Construct a four-layer LLaDA wrapper for CPU aggregation tests."""
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
    core = LLaDAModel(model_config, init_params=False)
    model = LLaDAModelLM(config, model=core, init_params=False)
    model.eval()
    return model


def _make_captures(structure, batch_size: int, sequence_length: int, *, random=False):
    generator = torch.Generator().manual_seed(17)
    captures = {}
    for layer in structure.selected_layers:
        shape = (batch_size, sequence_length, 16)
        if random:
            q_projected = torch.randn(shape, generator=generator)
            k_projected = torch.randn(shape, generator=generator)
        else:
            q_projected = torch.zeros(shape)
            k_projected = torch.zeros(shape)
        captures[layer.layer_id] = {"q": q_projected, "k": k_projected}
    return captures


def test_absolute_mappings_exclude_prompt_and_padding_with_ragged_batches():
    structure = resolve_llada_attention_structure(
        _make_tiny_llada(),
        last_n_layers=2,
    )
    captures = _make_captures(structure, batch_size=2, sequence_length=6)
    active_mask = torch.tensor(
        [
            [1, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 0],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
        ]
    )

    output = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        attention_mask=attention_mask,
    )

    assert output.directed.shape == (2, 2, 2)
    assert torch.equal(output.query_positions, torch.tensor([[2, 4], [3, -1]]))
    assert torch.equal(output.key_positions, output.query_positions)
    assert torch.equal(
        output.query_valid_mask,
        torch.tensor([[True, True], [True, False]]),
    )
    assert torch.equal(output.key_valid_mask, output.query_valid_mask)
    assert torch.equal(output.directed[0], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.count_nonzero(output.directed[1]) == 0
    assert output.layer_ids == (2, 3)


def test_selected_key_renormalization_is_explicit():
    structure = resolve_llada_attention_structure(
        _make_tiny_llada(),
        last_n_layers=2,
    )
    captures = _make_captures(structure, batch_size=1, sequence_length=4)
    active_mask = torch.tensor([[0, 0, 1, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1]])

    renormalized = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        zero_diagonal=False,
        renormalize_selected_keys=True,
    )
    preserved_mass = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        zero_diagonal=False,
        renormalize_selected_keys=False,
    )

    assert torch.equal(renormalized.directed, torch.full((1, 2, 2), 0.5))
    assert torch.equal(preserved_mass.directed, torch.full((1, 2, 2), 0.25))
    assert renormalized.renormalized_selected_keys
    assert not preserved_mass.renormalized_selected_keys


def test_diagonal_zeroing_preserves_absolute_position_mappings():
    structure = resolve_llada_attention_structure(
        _make_tiny_llada(),
        last_n_layers=2,
    )
    captures = _make_captures(structure, batch_size=1, sequence_length=4)
    active_mask = torch.tensor([[0, 0, 1, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1]])

    with_diagonal = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        zero_diagonal=False,
        renormalize_selected_keys=False,
    )
    without_diagonal = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        zero_diagonal=True,
        renormalize_selected_keys=False,
    )

    assert torch.equal(with_diagonal.query_positions, without_diagonal.query_positions)
    assert torch.equal(with_diagonal.key_positions, without_diagonal.key_positions)
    assert torch.equal(
        torch.diagonal(without_diagonal.directed, dim1=-2, dim2=-1),
        torch.zeros((1, 2)),
    )
    assert torch.equal(
        without_diagonal.directed[0, [0, 1], [1, 0]],
        with_diagonal.directed[0, [0, 1], [1, 0]],
    )


def test_empty_and_singleton_active_sets_are_finite_and_well_shaped():
    structure = resolve_llada_attention_structure(
        _make_tiny_llada(),
        last_n_layers=2,
    )
    captures = _make_captures(structure, batch_size=1, sequence_length=4)
    response_mask = torch.tensor([[0, 0, 1, 1]])

    empty = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=torch.zeros((1, 4), dtype=torch.bool),
        response_mask=response_mask,
    )
    singleton = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=torch.tensor([[0, 0, 0, 1]]),
        response_mask=response_mask,
    )

    assert empty.directed.shape == (1, 0, 0)
    assert empty.query_positions.shape == (1, 0)
    assert empty.key_positions.shape == (1, 0)
    assert singleton.directed.shape == (1, 1, 1)
    assert singleton.directed.item() == 0.0
    assert singleton.query_positions.item() == 3
    assert torch.isfinite(singleton.directed).all()


def test_aggregation_matches_direct_head_and_layer_means():
    structure = resolve_llada_attention_structure(
        _make_tiny_llada(),
        last_n_layers=2,
    )
    captures = _make_captures(
        structure,
        batch_size=1,
        sequence_length=4,
        random=True,
    )
    positions = torch.tensor([2, 3])
    active_mask = torch.tensor([[0, 0, 1, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1]])

    output = build_active_dependency_matrix(
        structure,
        captures,
        active_mask=active_mask,
        response_mask=response_mask,
        zero_diagonal=False,
        renormalize_selected_keys=False,
    )
    layer_means = []
    for layer in structure.selected_layers:
        reconstructed = reconstruct_llada_attention(
            layer,
            captures[layer.layer_id]["q"],
            captures[layer.layer_id]["k"],
            query_positions=positions,
        )
        selected = reconstructed.probabilities.index_select(-1, positions)
        layer_means.append(selected.mean(dim=1).squeeze(0))
    expected = torch.stack(layer_means).mean(dim=0)

    assert torch.allclose(output.directed.squeeze(0), expected, atol=1e-7)
