"""
Test lifecycle and tensor metadata for scoped LLaDA Q/K projection hooks.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_capture.py -v
"""

import pytest
import torch

from dllm.core.samplers.dependency import (
    LLaDAQKCapture,
    UnsupportedDependencyModelError,
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


def _make_tiny_llada(n_kv_heads: int = 4) -> LLaDAModelLM:
    """Construct a deterministic four-layer LLaDA wrapper on CPU."""
    config = LLaDAConfig(
        d_model=16,
        n_heads=4,
        n_kv_heads=n_kv_heads,
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


def _selected_layers(model: LLaDAModelLM):
    return tuple(model.model.transformer["blocks"][-2:])


def _hook_counts(model: LLaDAModelLM) -> tuple[int, tuple[tuple[int, int], ...]]:
    projection_counts = tuple(
        (len(layer.q_proj._forward_hooks), len(layer.k_proj._forward_hooks))
        for layer in _selected_layers(model)
    )
    return len(model._forward_pre_hooks), projection_counts


def _forward(model: LLaDAModelLM, token_offset: int = 0) -> torch.Tensor:
    input_ids = torch.tensor([[1, 2, 3, 4]]) + token_offset
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


@pytest.mark.parametrize(
    ("last_n_layers", "n_kv_heads", "expected_layer_ids"),
    [
        (2, 4, (2, 3)),
        (4, 2, (0, 1, 2, 3)),
    ],
)
def test_captured_qk_match_selected_layer_metadata(
    last_n_layers,
    n_kv_heads,
    expected_layer_ids,
):
    model = _make_tiny_llada(n_kv_heads=n_kv_heads)

    with LLaDAQKCapture(model, last_n_layers=last_n_layers) as capture:
        _forward(model)
        assert capture.structure is not None
        assert capture.structure.layer_ids == expected_layer_ids
        assert capture.structure.n_heads == 4
        assert capture.structure.n_kv_heads == n_kv_heads
        assert capture.structure.head_dim == 4
        assert capture.structure.kv_head_repeat == 4 // n_kv_heads
        assert tuple(capture.captures) == expected_layer_ids

        for layer in capture.structure.selected_layers:
            q_projected = capture.captures[layer.layer_id]["q"]
            k_projected = capture.captures[layer.layer_id]["k"]
            assert q_projected.shape == (1, 4, 16)
            assert k_projected.shape == (1, 4, n_kv_heads * 4)
            assert q_projected.dtype == layer.q_proj.weight.dtype
            assert k_projected.dtype == layer.k_proj.weight.dtype
            assert q_projected.device == layer.q_proj.weight.device
            assert k_projected.device == layer.k_proj.weight.device
            assert not q_projected.requires_grad
            assert not k_projected.requires_grad
            assert q_projected.grad_fn is None
            assert k_projected.grad_fn is None


def test_capture_detaches_projection_outputs_when_gradients_are_enabled():
    model = _make_tiny_llada()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with LLaDAQKCapture(model, last_n_layers=2) as capture:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        assert logits.requires_grad
        for projections in capture.captures.values():
            assert all(not tensor.requires_grad for tensor in projections.values())
            assert all(tensor.grad_fn is None for tensor in projections.values())


def test_rejects_non_divisible_grouped_query_heads():
    model = _make_tiny_llada(n_kv_heads=3)

    with pytest.raises(
        UnsupportedDependencyModelError,
        match="n_heads=4 must be divisible by positive n_kv_heads=3",
    ):
        with LLaDAQKCapture(model, last_n_layers=2):
            pass


def test_hooks_capture_only_while_active_and_are_removed_on_exit():
    model = _make_tiny_llada()
    baseline_hook_counts = _hook_counts(model)
    capture = LLaDAQKCapture(model, last_n_layers=2)

    assert capture.captures == {}
    with capture:
        assert capture.active
        assert _hook_counts(model) == (1, ((1, 1), (1, 1)))
        _forward(model)
        assert tuple(capture.captures) == (2, 3)
        assert all(tuple(values) == ("q", "k") for values in capture.captures.values())

    assert not capture.active
    assert capture.captures == {}
    assert _hook_counts(model) == baseline_hook_counts
    _forward(model)
    assert capture.captures == {}


def test_hooks_are_removed_and_captures_cleared_on_exception():
    model = _make_tiny_llada()
    baseline_hook_counts = _hook_counts(model)
    capture = LLaDAQKCapture(model, last_n_layers=2)

    with pytest.raises(RuntimeError, match="test failure"):
        with capture:
            _forward(model)
            assert capture.captures
            raise RuntimeError("test failure")

    assert not capture.active
    assert capture.captures == {}
    assert _hook_counts(model) == baseline_hook_counts


def test_each_forward_replaces_previous_captures():
    model = _make_tiny_llada()

    with LLaDAQKCapture(model, last_n_layers=2) as capture:
        _forward(model, token_offset=0)
        first_q = capture.captures[2]["q"]
        _forward(model, token_offset=1)
        second_q = capture.captures[2]["q"]

        assert first_q is not second_q
        assert not torch.equal(first_q, second_q)
        assert tuple(capture.captures) == (2, 3)


def test_two_capture_sessions_do_not_share_tensors():
    model = _make_tiny_llada()
    first_session = LLaDAQKCapture(model, last_n_layers=1)
    second_session = LLaDAQKCapture(model, last_n_layers=1)

    with first_session:
        _forward(model)
        first_q = first_session.captures[3]["q"]
    with second_session:
        assert second_session.captures == {}
        _forward(model)
        second_q = second_session.captures[3]["q"]

    assert first_q is not second_q
    assert first_session.captures == {}
    assert second_session.captures == {}


def test_disabled_capture_is_a_no_op():
    model = _make_tiny_llada()
    baseline_hook_counts = _hook_counts(model)
    baseline_logits = _forward(model)

    with LLaDAQKCapture(model, last_n_layers=2, enabled=False) as capture:
        captured_logits = _forward(model)
        assert capture.active
        assert capture.structure is None
        assert capture.captures == {}
        assert _hook_counts(model) == baseline_hook_counts

    assert torch.equal(baseline_logits, captured_logits)
    assert _hook_counts(model) == baseline_hook_counts


@pytest.mark.parametrize("projection_name", ["q_proj", "k_proj"])
def test_missing_projection_fails_without_leaking_hooks(projection_name):
    model = _make_tiny_llada()
    setattr(model.model.transformer["blocks"][-1], projection_name, None)
    baseline_pre_hook_count = len(model._forward_pre_hooks)

    with pytest.raises(
        UnsupportedDependencyModelError,
        match=f"layer 3 is missing a linear {projection_name} module",
    ):
        with LLaDAQKCapture(model, last_n_layers=1):
            pass

    assert len(model._forward_pre_hooks) == baseline_pre_hook_count
