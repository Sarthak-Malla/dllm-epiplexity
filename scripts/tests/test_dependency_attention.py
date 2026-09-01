"""
Test LLaDA attention reconstruction against direct tensor references.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_attention.py -v
"""

import math

import torch

from dllm.core.samplers.dependency import (
    LLaDAQKCapture,
    reconstruct_llada_attention,
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


def _make_tiny_llada(
    *,
    n_kv_heads: int = 4,
    attention_layer_norm: bool = False,
    rope: bool = True,
) -> LLaDAModelLM:
    """Construct a four-layer LLaDA wrapper for CPU attention tests."""
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        n_layers=4,
        mlp_hidden_size=32,
        activation_type=ActivationType.silu,
        block_type=BlockType.llama,
        block_group_size=1,
        attention_layer_norm=attention_layer_norm,
        rope=rope,
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


def _manual_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the local LLaDA rotary convention without calling rotary_emb."""
    original_q_dtype = q.dtype
    original_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    head_dim = q.shape[-1]
    positions = torch.arange(k.shape[-2], dtype=torch.float32, device=q.device)
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device)
            / head_dim
        )
    )
    frequencies = torch.outer(positions, inverse_frequency)
    angles = torch.cat((frequencies, frequencies), dim=-1)[None, None, :, :]

    def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.reshape(*tensor.shape[:-1], 2, head_dim // 2).unbind(
            dim=-2
        )
        return torch.cat((-second, first), dim=-1)

    q = q * angles.cos() + rotate_half(q) * angles.sin()
    k = k * angles.cos() + rotate_half(k) * angles.sin()
    return q.to(original_q_dtype), k.to(original_k_dtype)


def test_reconstruction_matches_manual_norm_rope_gqa_bias_reference():
    model = _make_tiny_llada(n_kv_heads=2, attention_layer_norm=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    with LLaDAQKCapture(model, last_n_layers=1) as capture:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        assert capture.structure is not None
        layer = capture.structure.selected_layers[0]
        q_projected = capture.captures[layer.layer_id]["q"]
        k_projected = capture.captures[layer.layer_id]["k"]

        query_positions = torch.tensor([1, 3])
        key_positions = torch.tensor([0, 2, 3])
        attention_bias = torch.tensor(
            [
                [0.0, -0.2, -0.4, -0.6],
                [0.3, 0.0, -0.1, -0.5],
                [-0.2, 0.4, 0.0, -0.3],
                [0.1, -0.4, 0.2, 0.0],
            ]
        )
        reconstructed = reconstruct_llada_attention(
            layer,
            q_projected,
            k_projected,
            attention_bias=attention_bias,
            attention_mask=attention_mask,
            query_positions=query_positions,
            key_positions=key_positions,
        )

        q = layer.q_norm(q_projected).to(k_projected.dtype)
        k = layer.k_norm(k_projected).to(k_projected.dtype)
        q = q.reshape(1, 4, 4, 4).transpose(1, 2)
        k = k.reshape(1, 4, 2, 4).transpose(1, 2)
        q, k = _manual_rope(q, k, theta=layer.config.rope_theta)
        k = k.repeat_interleave(2, dim=1)
        q = q.index_select(-2, query_positions)
        k = k.index_select(-2, key_positions)
        manual_scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        manual_scores *= 1.0 / math.sqrt(4)
        selected_bias = attention_bias[query_positions][:, key_positions]
        selected_bias = selected_bias[None, None, :, :].clone()
        selected_bias[:, :, :, -1] = torch.finfo(torch.float32).min
        expected = torch.softmax(manual_scores + selected_bias, dim=-1)

    assert reconstructed.layer_id == 3
    assert torch.equal(reconstructed.query_positions, query_positions)
    assert torch.equal(reconstructed.key_positions, key_positions)
    assert reconstructed.probabilities.shape == (1, 4, 2, 3)
    assert reconstructed.probabilities.dtype == torch.float32
    assert torch.allclose(reconstructed.probabilities, expected, atol=1e-6)
    assert torch.allclose(
        reconstructed.probabilities.sum(dim=-1),
        torch.ones((1, 4, 2)),
        atol=1e-6,
    )
    assert torch.count_nonzero(reconstructed.probabilities[..., -1]) == 0


def test_reconstruction_uses_query_then_key_axis_convention_without_rope():
    model = _make_tiny_llada(rope=False)
    layer = model.model.transformer["blocks"][-1]
    q_projected = torch.zeros((1, 2, 16))
    k_projected = torch.zeros((1, 2, 16))
    q_projected[0, 0, 0] = 2.0
    k_projected[0, 1, 0] = 2.0

    reconstructed = reconstruct_llada_attention(
        layer,
        q_projected,
        k_projected,
        query_positions=torch.tensor([0]),
        key_positions=torch.tensor([0, 1]),
    )

    assert reconstructed.probabilities.shape == (1, 4, 1, 2)
    assert reconstructed.probabilities[0, 0, 0, 1] > 0.85
    assert torch.equal(
        reconstructed.probabilities[0, 1:, 0],
        torch.full((3, 2), 0.5),
    )


def test_boolean_attention_bias_uses_true_as_allowed():
    model = _make_tiny_llada(rope=False)
    layer = model.model.transformer["blocks"][-1]
    q_projected = torch.zeros((1, 2, 16))
    k_projected = torch.zeros((1, 2, 16))
    allowed = torch.tensor([[True, False], [True, True]])

    reconstructed = reconstruct_llada_attention(
        layer,
        q_projected,
        k_projected,
        attention_bias=allowed,
    )

    assert torch.equal(
        reconstructed.probabilities[:, :, 0, :],
        torch.tensor([[[1.0, 0.0]]]).expand(1, 4, 2),
    )
    assert torch.equal(
        reconstructed.probabilities[:, :, 1, :],
        torch.full((1, 4, 2), 0.5),
    )
