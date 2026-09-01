"""
Validate LLaDA structure and scope dependency-guided Q/K capture hooks.

Run the focused tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest \
        /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_structure.py \
        /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_capture.py \
        /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_attention.py \
        /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_aggregation.py \
        -v
"""

import math
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Callable

import torch
import torch.nn as nn

from dllm.pipelines.llada.models.configuration_llada import BlockType
from dllm.pipelines.llada.models.modeling_llada import (
    LLaDALlamaBlock,
    LLaDAModel,
    LLaDAModelLM,
)


class UnsupportedDependencyModelError(ValueError):
    """Raised when a model cannot support the planned LLaDA Q/K capture path."""


@dataclass(frozen=True)
class LLaDAAttentionStructure:
    """Validated LLaDA modules and dimensions needed by later capture tasks."""

    wrapper: LLaDAModelLM
    core: LLaDAModel
    selected_layers: tuple[LLaDALlamaBlock, ...]
    layer_ids: tuple[int, ...]
    q_norm_present: tuple[bool, ...]
    k_norm_present: tuple[bool, ...]
    rotary_emb_present: tuple[bool, ...]
    n_heads: int
    n_kv_heads: int
    head_dim: int
    kv_head_repeat: int
    q_projection_size: int
    k_projection_size: int


@dataclass(frozen=True)
class LLaDAReconstructedAttention:
    """Float32 probabilities shaped [batch, head, query, key] for one layer."""

    layer_id: int
    probabilities: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor


@dataclass(frozen=True)
class DependencyCaptureOutput:
    """Padded active-position dependency matrices and absolute mappings."""

    directed: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    query_valid_mask: torch.Tensor
    key_valid_mask: torch.Tensor
    layer_ids: tuple[int, ...]
    renormalized_selected_keys: bool
    diagonal_zeroed: bool
    sink_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LLaDALogitInvarianceResult:
    """Exact disabled/enabled/post-removal capture comparison."""

    disabled_enabled_equal: bool
    disabled_post_removal_equal: bool
    enabled_post_removal_equal: bool
    disabled_enabled_argmax_equal: bool
    disabled_post_removal_argmax_equal: bool
    disabled_enabled_max_abs_diff: float
    disabled_post_removal_max_abs_diff: float
    model_state_unchanged: bool
    disabled_hooks_unchanged: bool
    enabled_hooks_active: bool
    hooks_removed: bool
    captures_complete: bool
    layer_ids: tuple[int, ...]
    capture_metadata: tuple[dict[str, object], ...]

    @property
    def passed(self) -> bool:
        """Whether every exact-invariance and lifecycle condition passed."""
        return all(
            (
                self.disabled_enabled_equal,
                self.disabled_post_removal_equal,
                self.enabled_post_removal_equal,
                self.disabled_enabled_argmax_equal,
                self.disabled_post_removal_argmax_equal,
                self.model_state_unchanged,
                self.disabled_hooks_unchanged,
                self.enabled_hooks_active,
                self.hooks_removed,
                self.captures_complete,
            )
        )


class LLaDAQKCapture:
    """Scope Q/K projection hooks to an explicit, reusable-safe context."""

    def __init__(
        self,
        model: nn.Module,
        last_n_layers: int = 4,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.last_n_layers = last_n_layers
        self.enabled = enabled
        self.structure: LLaDAAttentionStructure | None = None
        self._active = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captures: dict[int, dict[str, torch.Tensor]] = {}

    @property
    def active(self) -> bool:
        """Whether this capture context is currently entered."""
        return self._active

    @property
    def captures(self) -> dict[int, dict[str, torch.Tensor]]:
        """Return a shallow snapshot of projections from the latest forward."""
        return {
            layer_id: projections.copy()
            for layer_id, projections in self._captures.items()
        }

    def clear(self) -> None:
        """Drop all tensor references captured by the current session."""
        self._captures.clear()

    def _clear_before_forward(self, _module: nn.Module, _inputs: tuple) -> None:
        self.clear()

    def _projection_hook(
        self,
        layer_id: int,
        projection_name: str,
        expected_width: int,
    ) -> Callable[[nn.Module, tuple, torch.Tensor], None]:
        def record_projection(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            if not self._active:
                return
            if not isinstance(output, torch.Tensor):
                raise UnsupportedDependencyModelError(
                    "Unsupported LLaDA projection output: "
                    f"layer {layer_id} {projection_name}_proj returned "
                    f"{type(output).__name__}, expected torch.Tensor."
                )
            if output.ndim != 3:
                raise UnsupportedDependencyModelError(
                    "Unsupported LLaDA projection shape: "
                    f"layer {layer_id} {projection_name}_proj returned "
                    f"shape {tuple(output.shape)}, expected [B, T, {expected_width}]."
                )
            if output.shape[-1] != expected_width:
                raise UnsupportedDependencyModelError(
                    "Unsupported LLaDA projection width: "
                    f"layer {layer_id} {projection_name}_proj returned "
                    f"width {output.shape[-1]}, expected {expected_width}."
                )
            self._captures.setdefault(layer_id, {})[projection_name] = output.detach()

        return record_projection

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "LLaDAQKCapture":
        if self._active:
            raise RuntimeError("LLaDAQKCapture cannot be entered while already active.")

        self.clear()
        self.structure = None
        self._active = True
        if not self.enabled:
            return self

        try:
            self.structure = resolve_llada_attention_structure(
                self.model,
                last_n_layers=self.last_n_layers,
            )
            self._handles.append(
                self.structure.wrapper.register_forward_pre_hook(
                    self._clear_before_forward
                )
            )
            for layer in self.structure.selected_layers:
                self._handles.append(
                    layer.q_proj.register_forward_hook(
                        self._projection_hook(
                            layer.layer_id,
                            "q",
                            self.structure.q_projection_size,
                        )
                    )
                )
                self._handles.append(
                    layer.k_proj.register_forward_hook(
                        self._projection_hook(
                            layer.layer_id,
                            "k",
                            self.structure.k_projection_size,
                        )
                    )
                )
        except Exception:
            self._remove_hooks()
            self.clear()
            self._active = False
            self.structure = None
            raise
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        self._remove_hooks()
        self.clear()
        self._active = False
        return False


def _model_state_signature(model: nn.Module) -> tuple[tuple[object, ...], ...]:
    """Record state identity and mutation versions without cloning model weights."""
    signature = []
    for state_kind, named_tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named_tensors:
            signature.append(
                (
                    state_kind,
                    name,
                    id(tensor),
                    tensor.data_ptr(),
                    tensor._version,
                    tuple(tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
    signature.extend(
        ("training", name, module.training) for name, module in model.named_modules()
    )
    return tuple(signature)


def _capture_hook_signature(
    structure: LLaDAAttentionStructure,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Count wrapper pre-hooks and selected-layer Q/K projection hooks."""
    projection_hooks = tuple(
        (len(layer.q_proj._forward_hooks), len(layer.k_proj._forward_hooks))
        for layer in structure.selected_layers
    )
    return len(structure.wrapper._forward_pre_hooks), projection_hooks


def _expected_active_hook_signature(
    baseline: tuple[int, tuple[tuple[int, int], ...]],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return hook counts expected inside one active capture context."""
    wrapper_hooks, projection_hooks = baseline
    return wrapper_hooks + 1, tuple(
        (q_hooks + 1, k_hooks + 1) for q_hooks, k_hooks in projection_hooks
    )


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a scalar maximum absolute difference for equal-shaped tensors."""
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max().item())


@torch.inference_mode()
def check_llada_capture_invariance(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    last_n_layers: int = 4,
) -> LLaDALogitInvarianceResult:
    """Compare exact logits and model state around one Q/K capture session."""
    structure = resolve_llada_attention_structure(
        model,
        last_n_layers=last_n_layers,
    )
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T].")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape.")

    def forward_logits() -> torch.Tensor:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits.detach().clone()

    forward_logits()
    baseline_state = _model_state_signature(model)
    baseline_hooks = _capture_hook_signature(structure)

    with LLaDAQKCapture(
        model,
        last_n_layers=last_n_layers,
        enabled=False,
    ):
        disabled_hooks_unchanged = (
            _capture_hook_signature(structure) == baseline_hooks
        )
        disabled_logits = forward_logits()
    disabled_hooks_unchanged = disabled_hooks_unchanged and (
        _capture_hook_signature(structure) == baseline_hooks
    )
    disabled_state_unchanged = _model_state_signature(model) == baseline_state

    capture_metadata = []
    with LLaDAQKCapture(
        model,
        last_n_layers=last_n_layers,
        enabled=True,
    ) as capture:
        enabled_hooks_active = _capture_hook_signature(
            structure
        ) == _expected_active_hook_signature(baseline_hooks)
        enabled_logits = forward_logits()
        captured = capture.captures
        captures_complete = tuple(captured) == structure.layer_ids and all(
            set(captured[layer_id]) == {"q", "k"}
            for layer_id in structure.layer_ids
        )
        for layer_id in structure.layer_ids:
            projections = captured.get(layer_id, {})
            q_projected = projections.get("q")
            k_projected = projections.get("k")
            capture_metadata.append(
                {
                    "layer_id": layer_id,
                    "q_shape": (
                        tuple(q_projected.shape)
                        if isinstance(q_projected, torch.Tensor)
                        else None
                    ),
                    "k_shape": (
                        tuple(k_projected.shape)
                        if isinstance(k_projected, torch.Tensor)
                        else None
                    ),
                    "q_dtype": (
                        str(q_projected.dtype)
                        if isinstance(q_projected, torch.Tensor)
                        else None
                    ),
                    "k_dtype": (
                        str(k_projected.dtype)
                        if isinstance(k_projected, torch.Tensor)
                        else None
                    ),
                    "q_device": (
                        str(q_projected.device)
                        if isinstance(q_projected, torch.Tensor)
                        else None
                    ),
                    "k_device": (
                        str(k_projected.device)
                        if isinstance(k_projected, torch.Tensor)
                        else None
                    ),
                }
            )
        enabled_state_unchanged = _model_state_signature(model) == baseline_state

    hooks_removed = _capture_hook_signature(structure) == baseline_hooks
    post_removal_logits = forward_logits()
    post_removal_state_unchanged = _model_state_signature(model) == baseline_state

    disabled_enabled_equal = torch.equal(disabled_logits, enabled_logits)
    disabled_post_removal_equal = torch.equal(
        disabled_logits,
        post_removal_logits,
    )
    return LLaDALogitInvarianceResult(
        disabled_enabled_equal=disabled_enabled_equal,
        disabled_post_removal_equal=disabled_post_removal_equal,
        enabled_post_removal_equal=torch.equal(
            enabled_logits,
            post_removal_logits,
        ),
        disabled_enabled_argmax_equal=torch.equal(
            disabled_logits.argmax(dim=-1),
            enabled_logits.argmax(dim=-1),
        ),
        disabled_post_removal_argmax_equal=torch.equal(
            disabled_logits.argmax(dim=-1),
            post_removal_logits.argmax(dim=-1),
        ),
        disabled_enabled_max_abs_diff=_max_abs_difference(
            disabled_logits,
            enabled_logits,
        ),
        disabled_post_removal_max_abs_diff=_max_abs_difference(
            disabled_logits,
            post_removal_logits,
        ),
        model_state_unchanged=(
            disabled_state_unchanged
            and enabled_state_unchanged
            and post_removal_state_unchanged
        ),
        disabled_hooks_unchanged=disabled_hooks_unchanged,
        enabled_hooks_active=enabled_hooks_active,
        hooks_removed=hooks_removed,
        captures_complete=captures_complete,
        layer_ids=structure.layer_ids,
        capture_metadata=tuple(capture_metadata),
    )


def _normalize_positions(
    positions: torch.Tensor | None,
    *,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Validate a shared, one-dimensional sequence-position selection."""
    if positions is None:
        return torch.arange(sequence_length, device=device, dtype=torch.long)
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if positions.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {positions.shape}.")
    if positions.dtype == torch.bool or torch.is_floating_point(positions):
        raise TypeError(f"{name} must contain integer indices.")

    normalized = positions.detach().to(device=device, dtype=torch.long)
    if normalized.numel() == 0:
        raise ValueError(f"{name} must contain at least one position.")
    if normalized.min().item() < 0 or normalized.max().item() >= sequence_length:
        raise IndexError(
            f"{name} must be within [0, {sequence_length - 1}], "
            f"got {normalized.tolist()}."
        )
    if torch.unique(normalized).numel() != normalized.numel():
        raise ValueError(f"{name} must not contain duplicate positions.")
    return normalized


def _selected_additive_attention_bias(
    *,
    attention_bias: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    n_heads: int,
    sequence_length: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert and select the additive bias used before attention softmax."""
    additive_bias = None
    minimum = torch.finfo(torch.float32).min

    if attention_bias is not None:
        if not isinstance(attention_bias, torch.Tensor):
            raise TypeError("attention_bias must be a torch.Tensor or None.")
        if attention_bias.ndim == 2:
            attention_bias = attention_bias[None, None, :, :]
        if attention_bias.ndim != 4:
            raise ValueError(
                "attention_bias must have shape [T, T] or [B, H, T, T], "
                f"got {tuple(attention_bias.shape)}."
            )
        if attention_bias.shape[-2:] != (sequence_length, sequence_length):
            raise ValueError(
                "attention_bias sequence dimensions must both equal "
                f"{sequence_length}, got {tuple(attention_bias.shape[-2:])}."
            )
        if attention_bias.shape[0] not in (1, batch_size):
            raise ValueError(
                f"attention_bias batch size must be 1 or {batch_size}, "
                f"got {attention_bias.shape[0]}."
            )
        if attention_bias.shape[1] not in (1, n_heads):
            raise ValueError(
                f"attention_bias head count must be 1 or {n_heads}, "
                f"got {attention_bias.shape[1]}."
            )

        if torch.is_floating_point(attention_bias):
            additive_bias = attention_bias.to(device=device, dtype=torch.float32)
        else:
            allowed = attention_bias.to(device=device) != 0
            additive_bias = torch.zeros_like(allowed, dtype=torch.float32)
            additive_bias.masked_fill_(~allowed, minimum)
        additive_bias = additive_bias.index_select(-2, query_positions)
        additive_bias = additive_bias.index_select(-1, key_positions)

    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor or None.")
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape [B, T], "
                f"got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape[0] not in (1, batch_size):
            raise ValueError(
                f"attention_mask batch size must be 1 or {batch_size}, "
                f"got {attention_mask.shape[0]}."
            )
        if attention_mask.shape[1] != sequence_length:
            raise ValueError(
                f"attention_mask length must be {sequence_length}, "
                f"got {attention_mask.shape[1]}."
            )

        valid_keys = (attention_mask.to(device=device) != 0).index_select(
            -1,
            key_positions,
        )
        padding_bias = torch.zeros(
            (valid_keys.shape[0], 1, 1, valid_keys.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        padding_bias.masked_fill_(~valid_keys[:, None, None, :], minimum)
        additive_bias = (
            padding_bias if additive_bias is None else additive_bias + padding_bias
        )

    if additive_bias is None:
        return None
    if torch.isnan(additive_bias).any() or torch.isposinf(additive_bias).any():
        raise ValueError("attention bias must not contain NaN or positive infinity.")
    return torch.where(
        torch.isneginf(additive_bias),
        torch.full_like(additive_bias, minimum),
        additive_bias,
    )


@torch.no_grad()
def reconstruct_llada_attention(
    layer: LLaDALlamaBlock,
    q_projected: torch.Tensor,
    k_projected: torch.Tensor,
    *,
    attention_bias: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> LLaDAReconstructedAttention:
    """Reconstruct selected query-to-key probabilities for one LLaDA layer."""
    if not isinstance(layer, LLaDALlamaBlock):
        raise UnsupportedDependencyModelError(
            "Attention reconstruction requires LLaDALlamaBlock; "
            f"got {type(layer).__name__}."
        )
    if q_projected.ndim != 3 or k_projected.ndim != 3:
        raise ValueError("q_projected and k_projected must have shape [B, T, C].")
    if q_projected.shape[:2] != k_projected.shape[:2]:
        raise ValueError(
            "q_projected and k_projected must have matching batch and sequence "
            f"dimensions, got {q_projected.shape[:2]} and {k_projected.shape[:2]}."
        )
    if q_projected.device != k_projected.device:
        raise ValueError("q_projected and k_projected must be on the same device.")
    if q_projected.dtype != k_projected.dtype:
        raise ValueError("q_projected and k_projected must have the same dtype.")

    batch_size, sequence_length, q_width = q_projected.shape
    config = layer.config
    n_heads = int(config.n_heads)
    n_kv_heads = int(config.effective_n_kv_heads)
    d_model = int(config.d_model)
    if n_heads <= 0 or d_model % n_heads != 0:
        raise UnsupportedDependencyModelError(
            f"d_model={d_model} must be divisible by positive n_heads={n_heads}."
        )
    if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise UnsupportedDependencyModelError(
            f"n_heads={n_heads} must be divisible by positive "
            f"n_kv_heads={n_kv_heads}."
        )
    head_dim = d_model // n_heads
    expected_k_width = n_kv_heads * head_dim
    if q_width != d_model or k_projected.shape[-1] != expected_k_width:
        raise ValueError(
            "Projection widths do not match the layer configuration: "
            f"q={q_width} versus {d_model}, k={k_projected.shape[-1]} "
            f"versus {expected_k_width}."
        )

    dtype = k_projected.dtype
    q = q_projected
    k = k_projected
    if (layer.q_norm is None) != (layer.k_norm is None):
        raise UnsupportedDependencyModelError(
            f"Layer {layer.layer_id} must have both q_norm and k_norm or neither."
        )
    if layer.q_norm is not None and layer.k_norm is not None:
        q = layer.q_norm(q).to(dtype=dtype)
        k = layer.k_norm(k).to(dtype=dtype)

    q = q.reshape(batch_size, sequence_length, n_heads, head_dim).transpose(1, 2)
    k = k.reshape(
        batch_size,
        sequence_length,
        n_kv_heads,
        head_dim,
    ).transpose(1, 2)
    if config.rope:
        rotary_emb = getattr(layer, "rotary_emb", None)
        if rotary_emb is None:
            raise UnsupportedDependencyModelError(
                f"Layer {layer.layer_id} has RoPE enabled but no rotary_emb module."
            )
        q, k = rotary_emb(q, k)
    if n_kv_heads != n_heads:
        k = k.repeat_interleave(
            n_heads // n_kv_heads,
            dim=1,
            output_size=n_heads,
        )

    normalized_query_positions = _normalize_positions(
        query_positions,
        sequence_length=sequence_length,
        device=q.device,
        name="query_positions",
    )
    normalized_key_positions = _normalize_positions(
        key_positions,
        sequence_length=sequence_length,
        device=k.device,
        name="key_positions",
    )
    q = q.index_select(-2, normalized_query_positions)
    k = k.index_select(-2, normalized_key_positions)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores.mul_(1.0 / math.sqrt(head_dim))
    selected_bias = _selected_additive_attention_bias(
        attention_bias=attention_bias,
        attention_mask=attention_mask,
        batch_size=batch_size,
        n_heads=n_heads,
        sequence_length=sequence_length,
        query_positions=normalized_query_positions,
        key_positions=normalized_key_positions,
        device=q.device,
    )
    if selected_bias is not None:
        scores = scores + selected_bias
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)

    return LLaDAReconstructedAttention(
        layer_id=layer.layer_id,
        probabilities=probabilities,
        query_positions=normalized_query_positions,
        key_positions=normalized_key_positions,
    )


def _validate_selection_mask(
    mask: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate and move a batch-by-sequence selection mask."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must have shape {(batch_size, sequence_length)}, "
            f"got {tuple(mask.shape)}."
        )
    return mask.detach().to(device=device) != 0


def _capture_batch_shape(
    structure: LLaDAAttentionStructure,
    captures: dict[int, dict[str, torch.Tensor]],
) -> tuple[int, int, torch.device]:
    """Validate selected-layer captures and return their shared batch shape."""
    expected_shape = None
    expected_device = None
    for layer in structure.selected_layers:
        projections = captures.get(layer.layer_id)
        if not isinstance(projections, dict) or set(projections) != {"q", "k"}:
            raise ValueError(
                f"Missing complete Q/K captures for selected layer {layer.layer_id}."
            )
        q_projected = projections["q"]
        k_projected = projections["k"]
        if q_projected.ndim != 3 or k_projected.ndim != 3:
            raise ValueError(
                f"Layer {layer.layer_id} captures must have shape [B, T, C]."
            )
        if q_projected.shape[:2] != k_projected.shape[:2]:
            raise ValueError(
                f"Layer {layer.layer_id} Q/K batch shapes do not match."
            )
        if q_projected.device != k_projected.device:
            raise ValueError(f"Layer {layer.layer_id} Q/K devices do not match.")
        if q_projected.dtype != k_projected.dtype:
            raise ValueError(f"Layer {layer.layer_id} Q/K dtypes do not match.")
        if q_projected.shape[-1] != structure.q_projection_size:
            raise ValueError(
                f"Layer {layer.layer_id} Q width is {q_projected.shape[-1]}, "
                f"expected {structure.q_projection_size}."
            )
        if k_projected.shape[-1] != structure.k_projection_size:
            raise ValueError(
                f"Layer {layer.layer_id} K width is {k_projected.shape[-1]}, "
                f"expected {structure.k_projection_size}."
            )
        if expected_shape is None:
            expected_shape = q_projected.shape[:2]
            expected_device = q_projected.device
        elif q_projected.shape[:2] != expected_shape:
            raise ValueError("Selected layers do not share one batch/sequence shape.")
        elif q_projected.device != expected_device:
            raise ValueError("Selected layers do not share one capture device.")

    if expected_shape is None or expected_device is None:
        raise ValueError("At least one selected layer capture is required.")
    return int(expected_shape[0]), int(expected_shape[1]), expected_device


def _select_attention_bias_batch(
    attention_bias: torch.Tensor | None,
    *,
    batch_index: int,
    batch_size: int,
) -> torch.Tensor | None:
    """Select a batch row while preserving shared 2D or singleton biases."""
    if attention_bias is None or not isinstance(attention_bias, torch.Tensor):
        return attention_bias
    if attention_bias.ndim == 4 and attention_bias.shape[0] == batch_size:
        return attention_bias[batch_index : batch_index + 1]
    return attention_bias


@torch.no_grad()
def build_active_dependency_matrix(
    structure: LLaDAAttentionStructure,
    captures: dict[int, dict[str, torch.Tensor]],
    *,
    active_mask: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    attention_bias: torch.Tensor | None = None,
    zero_diagonal: bool = True,
    renormalize_selected_keys: bool = True,
) -> DependencyCaptureOutput:
    """Aggregate selected-layer attention over active response positions."""
    if not isinstance(structure, LLaDAAttentionStructure):
        raise TypeError("structure must be a LLaDAAttentionStructure.")
    if not isinstance(captures, dict):
        raise TypeError("captures must be a dictionary keyed by layer ID.")
    if not isinstance(zero_diagonal, bool):
        raise TypeError("zero_diagonal must be a bool.")
    if not isinstance(renormalize_selected_keys, bool):
        raise TypeError("renormalize_selected_keys must be a bool.")

    batch_size, sequence_length, device = _capture_batch_shape(
        structure,
        captures,
    )
    active = _validate_selection_mask(
        active_mask,
        name="active_mask",
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    response = _validate_selection_mask(
        response_mask,
        name="response_mask",
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    if attention_mask is None:
        valid_tokens = torch.ones_like(active)
        normalized_attention_mask = None
    else:
        valid_tokens = _validate_selection_mask(
            attention_mask,
            name="attention_mask",
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        normalized_attention_mask = valid_tokens

    eligible = active & response & valid_tokens
    positions_by_batch = tuple(
        torch.nonzero(row, as_tuple=False).flatten() for row in eligible
    )
    maximum_active = max(
        (positions.numel() for positions in positions_by_batch),
        default=0,
    )
    directed = torch.zeros(
        (batch_size, maximum_active, maximum_active),
        device=device,
        dtype=torch.float32,
    )
    query_positions = torch.full(
        (batch_size, maximum_active),
        -1,
        device=device,
        dtype=torch.long,
    )
    key_positions = query_positions.clone()
    query_valid_mask = torch.zeros(
        (batch_size, maximum_active),
        device=device,
        dtype=torch.bool,
    )
    key_valid_mask = query_valid_mask.clone()

    for batch_index, positions in enumerate(positions_by_batch):
        active_count = positions.numel()
        if active_count == 0:
            continue
        query_positions[batch_index, :active_count] = positions
        key_positions[batch_index, :active_count] = positions
        query_valid_mask[batch_index, :active_count] = True
        key_valid_mask[batch_index, :active_count] = True

        layer_sum = torch.zeros(
            (active_count, active_count),
            device=device,
            dtype=torch.float32,
        )
        for layer in structure.selected_layers:
            projections = captures[layer.layer_id]
            reconstructed = reconstruct_llada_attention(
                layer,
                projections["q"][batch_index : batch_index + 1],
                projections["k"][batch_index : batch_index + 1],
                attention_bias=_select_attention_bias_batch(
                    attention_bias,
                    batch_index=batch_index,
                    batch_size=batch_size,
                ),
                attention_mask=(
                    None
                    if normalized_attention_mask is None
                    else normalized_attention_mask[
                        batch_index : batch_index + 1
                    ]
                ),
                query_positions=positions,
                key_positions=(positions if renormalize_selected_keys else None),
            )
            probabilities = reconstructed.probabilities
            if not renormalize_selected_keys:
                probabilities = probabilities.index_select(-1, positions)
            layer_sum.add_(probabilities.mean(dim=1).squeeze(0))

        matrix = layer_sum / len(structure.selected_layers)
        if zero_diagonal:
            same_position = positions[:, None] == positions[None, :]
            matrix.masked_fill_(same_position, 0.0)
        if renormalize_selected_keys:
            row_sums = matrix.sum(dim=-1, keepdim=True)
            denominators = torch.where(
                row_sums > 0,
                row_sums,
                torch.ones_like(row_sums),
            )
            matrix = matrix / denominators
        directed[batch_index, :active_count, :active_count] = matrix

    return DependencyCaptureOutput(
        directed=directed,
        query_positions=query_positions,
        key_positions=key_positions,
        query_valid_mask=query_valid_mask,
        key_valid_mask=key_valid_mask,
        layer_ids=structure.layer_ids,
        renormalized_selected_keys=renormalize_selected_keys,
        diagonal_zeroed=zero_diagonal,
    )


def _validate_dependency_output(
    output: DependencyCaptureOutput,
) -> tuple[int, int, int]:
    """Validate dependency and padding shapes used by sink filtering."""
    if not isinstance(output, DependencyCaptureOutput):
        raise TypeError("output must be a DependencyCaptureOutput.")
    if output.directed.ndim != 3:
        raise ValueError("output.directed must have shape [B, M_query, M_key].")
    batch_size, query_count, key_count = output.directed.shape
    if output.query_valid_mask.shape != (batch_size, query_count):
        raise ValueError("query_valid_mask does not match output.directed.")
    if output.key_valid_mask.shape != (batch_size, key_count):
        raise ValueError("key_valid_mask does not match output.directed.")
    if output.query_positions.shape != (batch_size, query_count):
        raise ValueError("query_positions does not match output.directed.")
    if output.key_positions.shape != (batch_size, key_count):
        raise ValueError("key_positions does not match output.directed.")
    if output.query_valid_mask.device != output.directed.device:
        raise ValueError("query_valid_mask must share output.directed's device.")
    if output.key_valid_mask.device != output.directed.device:
        raise ValueError("key_valid_mask must share output.directed's device.")
    if not torch.is_floating_point(output.directed):
        raise TypeError("output.directed must use a floating-point dtype.")
    if not torch.isfinite(output.directed).all():
        raise ValueError("output.directed must contain only finite values.")
    if (output.directed < 0).any():
        raise ValueError("output.directed must contain nonnegative dependencies.")
    return batch_size, query_count, key_count


@torch.no_grad()
def detect_dependency_sinks(
    output: DependencyCaptureOutput,
    *,
    sink_quantile: float | None = 0.99,
    sink_threshold: float | None = None,
) -> torch.Tensor:
    """Detect high-incoming-mass key outliers, excluding padded positions."""
    batch_size, _, key_count = _validate_dependency_output(output)
    if sink_threshold is not None:
        if isinstance(sink_threshold, bool) or not isinstance(
            sink_threshold,
            (int, float),
        ):
            raise TypeError("sink_threshold must be a nonnegative number or None.")
        if not math.isfinite(float(sink_threshold)) or sink_threshold < 0:
            raise ValueError("sink_threshold must be finite and nonnegative.")
    elif sink_quantile is None:
        raise ValueError("Set sink_quantile or sink_threshold for sink detection.")
    elif isinstance(sink_quantile, bool) or not isinstance(
        sink_quantile,
        (int, float),
    ):
        raise TypeError("sink_quantile must be a number in [0, 1] or None.")
    elif not 0 <= sink_quantile <= 1:
        raise ValueError("sink_quantile must be in [0, 1].")

    query_valid = output.query_valid_mask.to(dtype=torch.bool)
    key_valid = output.key_valid_mask.to(dtype=torch.bool)
    incoming_sum = (
        output.directed * query_valid[:, :, None].to(output.directed.dtype)
    ).sum(dim=-2)
    valid_query_counts = query_valid.sum(dim=-1, keepdim=True)
    denominators = torch.where(
        valid_query_counts > 0,
        valid_query_counts,
        torch.ones_like(valid_query_counts),
    ).to(output.directed.dtype)
    mean_incoming = incoming_sum / denominators
    mean_incoming.masked_fill_(~key_valid, 0.0)

    sink_mask = torch.zeros(
        (batch_size, key_count),
        device=output.directed.device,
        dtype=torch.bool,
    )
    for batch_index in range(batch_size):
        valid_masses = mean_incoming[batch_index, key_valid[batch_index]]
        if valid_masses.numel() == 0:
            continue
        if sink_threshold is not None:
            cutoff = torch.as_tensor(
                sink_threshold,
                device=valid_masses.device,
                dtype=valid_masses.dtype,
            )
        else:
            cutoff = torch.quantile(valid_masses.float(), float(sink_quantile))
            cutoff = cutoff.to(valid_masses.dtype)
        sink_mask[batch_index, key_valid[batch_index]] = valid_masses > cutoff
    return sink_mask


@torch.no_grad()
def filter_dependency_sinks(
    output: DependencyCaptureOutput,
    *,
    enabled: bool = True,
    sink_quantile: float | None = 0.99,
    sink_threshold: float | None = None,
    renormalize_rows: bool = True,
) -> DependencyCaptureOutput:
    """Zero detected sink columns and safely renormalize valid dependency rows."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool.")
    if not isinstance(renormalize_rows, bool):
        raise TypeError("renormalize_rows must be a bool.")
    if not enabled:
        return output

    _validate_dependency_output(output)
    sink_mask = detect_dependency_sinks(
        output,
        sink_quantile=sink_quantile,
        sink_threshold=sink_threshold,
    )
    query_valid = output.query_valid_mask.to(dtype=torch.bool)
    key_valid = output.key_valid_mask.to(dtype=torch.bool)
    filtered = output.directed.clone()
    filtered.masked_fill_(~query_valid[:, :, None], 0.0)
    filtered.masked_fill_(~key_valid[:, None, :], 0.0)
    filtered.masked_fill_(sink_mask[:, None, :], 0.0)

    if renormalize_rows:
        row_sums = filtered.sum(dim=-1, keepdim=True)
        denominators = torch.where(
            row_sums > 0,
            row_sums,
            torch.ones_like(row_sums),
        )
        filtered = filtered / denominators
        filtered.masked_fill_(~query_valid[:, :, None], 0.0)

    return replace(
        output,
        directed=filtered,
        sink_mask=sink_mask,
    )


def _resolve_llada_layers(core: LLaDAModel) -> tuple[nn.Module, ...]:
    """Resolve flat transformer layers for grouped and ungrouped LLaDA layouts."""
    transformer = getattr(core, "transformer", None)
    if not isinstance(transformer, nn.ModuleDict):
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA structure: LLaDAModel.transformer must be an "
            f"nn.ModuleDict, got {type(transformer).__name__}."
        )

    block_group_size = int(core.config.block_group_size)
    if block_group_size == 1:
        if "blocks" not in transformer:
            raise UnsupportedDependencyModelError(
                "Unsupported LLaDA structure: block_group_size=1 requires "
                "model.transformer.blocks."
            )
        return tuple(transformer["blocks"])

    if "block_groups" not in transformer:
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA structure: grouped blocks require "
            "model.transformer.block_groups."
        )
    return tuple(
        block for block_group in transformer["block_groups"] for block in block_group
    )


def resolve_llada_attention_structure(
    model: nn.Module,
    last_n_layers: int = 4,
) -> LLaDAAttentionStructure:
    """Validate and resolve the LLaDA layers used by future Q/K capture."""
    if not isinstance(model, LLaDAModelLM):
        raise UnsupportedDependencyModelError(
            "Dependency capture supports only LLaDAModelLM; "
            f"got {type(model).__name__}."
        )

    core = getattr(model, "model", None)
    if not isinstance(core, LLaDAModel):
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA structure: LLaDAModelLM.model must be an "
            f"LLaDAModel, got {type(core).__name__}."
        )

    block_type = str(core.config.block_type)
    if block_type != BlockType.llama.value:
        raise UnsupportedDependencyModelError(
            "Dependency capture requires block_type='llama'; "
            f"got {block_type!r}."
        )

    layers = _resolve_llada_layers(core)
    expected_layer_count = int(core.config.n_layers)
    if len(layers) != expected_layer_count:
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA structure: resolved "
            f"{len(layers)} layers but config declares {expected_layer_count}."
        )
    if isinstance(last_n_layers, bool) or not isinstance(last_n_layers, int):
        raise ValueError("last_n_layers must be an integer.")
    if not 1 <= last_n_layers <= len(layers):
        raise ValueError(
            f"last_n_layers must be between 1 and {len(layers)}, "
            f"got {last_n_layers}."
        )

    for expected_layer_id, layer in enumerate(layers):
        if not isinstance(layer, LLaDALlamaBlock):
            raise UnsupportedDependencyModelError(
                "Dependency capture requires LLaDALlamaBlock layers; "
                f"layer {expected_layer_id} is {type(layer).__name__}."
            )
        if layer.layer_id != expected_layer_id:
            raise UnsupportedDependencyModelError(
                "Unsupported LLaDA structure: layer ordering mismatch at index "
                f"{expected_layer_id}, which reports layer_id={layer.layer_id}."
            )
        if not isinstance(getattr(layer, "q_proj", None), nn.Linear):
            raise UnsupportedDependencyModelError(
                f"Unsupported LLaDA structure: layer {expected_layer_id} is "
                "missing a linear q_proj module."
            )
        if not isinstance(getattr(layer, "k_proj", None), nn.Linear):
            raise UnsupportedDependencyModelError(
                f"Unsupported LLaDA structure: layer {expected_layer_id} is "
                "missing a linear k_proj module."
            )
        if layer.q_proj is layer.k_proj:
            raise UnsupportedDependencyModelError(
                f"Unsupported LLaDA structure: layer {expected_layer_id} does "
                "not have separate q_proj and k_proj modules."
            )
        if (layer.q_norm is None) != (layer.k_norm is None):
            raise UnsupportedDependencyModelError(
                f"Unsupported LLaDA structure: layer {expected_layer_id} has "
                "only one of q_norm and k_norm."
            )
        if core.config.rope and not hasattr(layer, "rotary_emb"):
            raise UnsupportedDependencyModelError(
                f"Unsupported LLaDA structure: layer {expected_layer_id} has "
                "RoPE enabled but no rotary_emb module."
            )

    selected_layers = tuple(layers[-last_n_layers:])
    selected_layer_ids = tuple(layer.layer_id for layer in selected_layers)
    block_config = selected_layers[0].config
    d_model = int(block_config.d_model)
    n_heads = int(block_config.n_heads)
    n_kv_heads = int(block_config.effective_n_kv_heads)
    if n_heads <= 0 or d_model % n_heads != 0:
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA head configuration: "
            f"d_model={d_model} must be divisible by positive n_heads={n_heads}."
        )
    if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise UnsupportedDependencyModelError(
            "Unsupported LLaDA grouped-query configuration: "
            f"n_heads={n_heads} must be divisible by positive "
            f"n_kv_heads={n_kv_heads}."
        )

    head_dim = d_model // n_heads
    q_projection_size = n_heads * head_dim
    k_projection_size = n_kv_heads * head_dim
    for layer in selected_layers:
        layer_config = layer.config
        layer_dimensions = (
            int(layer_config.d_model),
            int(layer_config.n_heads),
            int(layer_config.effective_n_kv_heads),
        )
        if layer_dimensions != (d_model, n_heads, n_kv_heads):
            raise UnsupportedDependencyModelError(
                "Unsupported LLaDA structure: selected layer "
                f"{layer.layer_id} has inconsistent (d_model, n_heads, "
                f"n_kv_heads)={layer_dimensions}."
            )
        if layer.q_proj.out_features != q_projection_size:
            raise UnsupportedDependencyModelError(
                "Unsupported LLaDA q_proj width: "
                f"layer {layer.layer_id} has {layer.q_proj.out_features}, "
                f"expected {q_projection_size}."
            )
        if layer.k_proj.out_features != k_projection_size:
            raise UnsupportedDependencyModelError(
                "Unsupported LLaDA k_proj width: "
                f"layer {layer.layer_id} has {layer.k_proj.out_features}, "
                f"expected {k_projection_size}."
            )

    return LLaDAAttentionStructure(
        wrapper=model,
        core=core,
        selected_layers=selected_layers,
        layer_ids=selected_layer_ids,
        q_norm_present=tuple(layer.q_norm is not None for layer in selected_layers),
        k_norm_present=tuple(layer.k_norm is not None for layer in selected_layers),
        rotary_emb_present=tuple(
            hasattr(layer, "rotary_emb") for layer in selected_layers
        ),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        kv_head_repeat=n_heads // n_kv_heads,
        q_projection_size=q_projection_size,
        k_projection_size=k_projection_size,
    )
