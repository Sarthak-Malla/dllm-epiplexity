"""
Collect reproducible dependency-proxy snapshots across masking stages.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_proxy_state_collection.py" -v
"""

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn as nn

from dllm.core.samplers.counterfactual import (
    decoding_risk_per_token,
    entropy_per_token,
    evaluate_one_position_counterfactuals,
)
from dllm.core.samplers.dependency import (
    LLaDAQKCapture,
    build_active_dependency_matrix,
    filter_dependency_sinks,
)


@dataclass(frozen=True)
class ProxyStateCollectionConfig:
    """Configuration frozen across one proxy-state trajectory."""

    mask_ratios: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    last_n_layers: int = 4
    top_confidence_pool_size: int = 16
    random_pool_size: int = 16
    oracle_candidate_chunk_size: int | None = None
    seed: int = 42
    renormalize_selected_keys: bool = True
    zero_diagonal: bool = True
    sink_filter_enabled: bool = True
    sink_quantile: float | None = 0.99
    sink_threshold: float | None = None


@dataclass(frozen=True)
class ProxyEvaluationPool:
    """Deterministic top-confidence and seeded-random evaluation positions."""

    top_confidence_positions: torch.Tensor
    random_positions: torch.Tensor
    evaluation_mask: torch.Tensor

    @property
    def positions(self) -> torch.Tensor:
        """Return all selected absolute positions in sequence order."""
        return torch.nonzero(self.evaluation_mask, as_tuple=False).flatten()


@dataclass(frozen=True)
class ProxyStateSnapshot:
    """One serializable proxy diagnostic state and its oracle targets."""

    example_id: str
    state_index: int
    seed: int
    target_mask_ratio: float
    actual_mask_ratio: float
    response_length: int
    masked_count: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    masked_active_mask: torch.Tensor
    active_positions: torch.Tensor
    committed_response_positions: torch.Tensor
    base_argmax_token_ids: torch.Tensor
    active_confidence: torch.Tensor
    active_entropy: torch.Tensor
    dependency_positions: torch.Tensor
    dependency_before_sink: torch.Tensor
    dependency_after_sink: torch.Tensor
    sink_mask: torch.Tensor | None
    capture_layer_ids: tuple[int, ...]
    top_confidence_positions: torch.Tensor
    random_positions: torch.Tensor
    evaluation_positions: torch.Tensor
    oracle_revealed_token_ids: torch.Tensor
    oracle_heldout_counts: torch.Tensor
    oracle_base_entropy_sum: torch.Tensor
    oracle_lookahead_entropy_sum: torch.Tensor
    oracle_entropy_drop_sum: torch.Tensor
    oracle_entropy_drop_per_heldout: torch.Tensor
    oracle_base_risk_sum: torch.Tensor
    oracle_lookahead_risk_sum: torch.Tensor
    oracle_risk_reduction_sum: torch.Tensor
    oracle_risk_reduction_per_heldout: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        """Convert the snapshot to JSON-compatible Python values."""
        return {
            "example_id": self.example_id,
            "state_index": self.state_index,
            "seed": self.seed,
            "target_mask_ratio": self.target_mask_ratio,
            "actual_mask_ratio": self.actual_mask_ratio,
            "response_length": self.response_length,
            "masked_count": self.masked_count,
            "input_ids": self.input_ids.tolist(),
            "attention_mask": self.attention_mask.tolist(),
            "response_mask": self.response_mask.tolist(),
            "masked_active_mask": self.masked_active_mask.tolist(),
            "active_positions": self.active_positions.tolist(),
            "committed_response_positions": (
                self.committed_response_positions.tolist()
            ),
            "base_argmax_token_ids": self.base_argmax_token_ids.tolist(),
            "active_confidence": self.active_confidence.tolist(),
            "active_entropy": self.active_entropy.tolist(),
            "dependency": {
                "direction_convention": (
                    "matrix[query, key]: query uses key as context"
                ),
                "positions": self.dependency_positions.tolist(),
                "before_sink": self.dependency_before_sink.tolist(),
                "after_sink": self.dependency_after_sink.tolist(),
                "sink_mask": (
                    None if self.sink_mask is None else self.sink_mask.tolist()
                ),
                "capture_layer_ids": list(self.capture_layer_ids),
            },
            "evaluation_pool": {
                "top_confidence_positions": (
                    self.top_confidence_positions.tolist()
                ),
                "random_positions": self.random_positions.tolist(),
                "positions": self.evaluation_positions.tolist(),
            },
            "oracle": {
                "positions": self.evaluation_positions.tolist(),
                "revealed_token_ids": self.oracle_revealed_token_ids.tolist(),
                "heldout_counts": self.oracle_heldout_counts.tolist(),
                "base_entropy_sum": self.oracle_base_entropy_sum.tolist(),
                "lookahead_entropy_sum": (
                    self.oracle_lookahead_entropy_sum.tolist()
                ),
                "entropy_drop_sum": self.oracle_entropy_drop_sum.tolist(),
                "entropy_drop_per_heldout": (
                    self.oracle_entropy_drop_per_heldout.tolist()
                ),
                "base_risk_sum": self.oracle_base_risk_sum.tolist(),
                "lookahead_risk_sum": self.oracle_lookahead_risk_sum.tolist(),
                "risk_reduction_sum": self.oracle_risk_reduction_sum.tolist(),
                "risk_reduction_per_heldout": (
                    self.oracle_risk_reduction_per_heldout.tolist()
                ),
            },
        }


@dataclass(frozen=True)
class ProxyStateCollection:
    """Configuration metadata and ordered snapshots for one example."""

    example_id: str
    checkpoint: str
    model_class: str
    mask_token_id: int
    sequence_length: int
    valid_token_count: int
    prompt_token_count: int
    response_length: int
    config: ProxyStateCollectionConfig
    states: tuple[ProxyStateSnapshot, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert the complete in-memory collection to a JSON-ready object."""
        return {
            "configuration": {
                "example_id": self.example_id,
                "checkpoint": self.checkpoint,
                "model_class": self.model_class,
                "mask_token_id": self.mask_token_id,
                "sequence_length": self.sequence_length,
                "valid_token_count": self.valid_token_count,
                "prompt_token_count": self.prompt_token_count,
                "response_length": self.response_length,
                "mask_ratios": list(self.config.mask_ratios),
                "last_n_layers": self.config.last_n_layers,
                "top_confidence_pool_size": (
                    self.config.top_confidence_pool_size
                ),
                "random_pool_size": self.config.random_pool_size,
                "oracle_candidate_chunk_size": (
                    self.config.oracle_candidate_chunk_size
                ),
                "seed": self.config.seed,
                "renormalize_selected_keys": (
                    self.config.renormalize_selected_keys
                ),
                "zero_diagonal": self.config.zero_diagonal,
                "sink_filter_enabled": self.config.sink_filter_enabled,
                "sink_quantile": self.config.sink_quantile,
                "sink_threshold": self.config.sink_threshold,
                "token_value_policy": "base_argmax",
                "trajectory_policy": "stagewise_highest_confidence",
                "dependency_direction": (
                    "directed[query, key] means query uses key as context"
                ),
            },
            "states": [state.to_dict() for state in self.states],
        }


def _validate_nonnegative_integer(value: int, *, name: str) -> None:
    """Reject booleans and negative/non-integer configuration values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _validate_config(config: ProxyStateCollectionConfig) -> None:
    """Validate one immutable collection configuration."""
    if not isinstance(config, ProxyStateCollectionConfig):
        raise TypeError("config must be a ProxyStateCollectionConfig.")
    if not config.mask_ratios:
        raise ValueError("mask_ratios must not be empty.")
    if not math.isclose(config.mask_ratios[0], 1.0):
        raise ValueError("mask_ratios must start at 1.0.")
    for ratio in config.mask_ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("mask_ratios must contain numbers.")
        if not math.isfinite(float(ratio)) or not 0 < ratio <= 1:
            raise ValueError("mask_ratios must be finite and in (0, 1].")
    if any(
        left <= right
        for left, right in zip(config.mask_ratios, config.mask_ratios[1:])
    ):
        raise ValueError("mask_ratios must be strictly decreasing.")
    if isinstance(config.last_n_layers, bool) or not isinstance(
        config.last_n_layers,
        int,
    ):
        raise ValueError("last_n_layers must be a positive integer.")
    if config.last_n_layers <= 0:
        raise ValueError("last_n_layers must be a positive integer.")
    _validate_nonnegative_integer(
        config.top_confidence_pool_size,
        name="top_confidence_pool_size",
    )
    _validate_nonnegative_integer(
        config.random_pool_size,
        name="random_pool_size",
    )
    if config.top_confidence_pool_size + config.random_pool_size == 0:
        raise ValueError("At least one evaluation-pool size must be positive.")
    if config.oracle_candidate_chunk_size is not None and (
        isinstance(config.oracle_candidate_chunk_size, bool)
        or not isinstance(config.oracle_candidate_chunk_size, int)
        or config.oracle_candidate_chunk_size <= 0
    ):
        raise ValueError(
            "oracle_candidate_chunk_size must be a positive integer or None."
        )
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError("seed must be an integer.")
    for name, value in (
        ("renormalize_selected_keys", config.renormalize_selected_keys),
        ("zero_diagonal", config.zero_diagonal),
        ("sink_filter_enabled", config.sink_filter_enabled),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool.")


def validate_proxy_state_collection_config(
    config: ProxyStateCollectionConfig,
    *,
    response_length: int,
) -> None:
    """Validate configuration and ratio realizability before model loading."""
    _validate_config(config)
    if isinstance(response_length, bool) or not isinstance(response_length, int):
        raise ValueError("response_length must be a positive integer.")
    if response_length <= 0:
        raise ValueError("response_length must be a positive integer.")
    _target_mask_counts(response_length, config.mask_ratios)


def _target_mask_counts(
    response_length: int,
    ratios: tuple[float, ...],
) -> tuple[int, ...]:
    """Map ratios to distinct nearest mask counts for one response length."""
    counts = tuple(
        max(1, int(math.floor(response_length * float(ratio) + 0.5)))
        for ratio in ratios
    )
    if any(left <= right for left, right in zip(counts, counts[1:])):
        raise ValueError(
            "response length is too short to realize distinct mask-ratio stages."
        )
    return counts


def _rank_positions(
    confidence: torch.Tensor,
    active_positions: torch.Tensor,
) -> torch.Tensor:
    """Rank active absolute positions by confidence with position tie-breaking."""
    ranked = sorted(
        zip(
            active_positions.detach().cpu().tolist(),
            confidence.index_select(0, active_positions).float().cpu().tolist(),
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return torch.tensor(
        [position for position, _ in ranked],
        device=active_positions.device,
        dtype=torch.long,
    )


def select_proxy_evaluation_pool(
    confidence: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    top_confidence_pool_size: int,
    random_pool_size: int,
    generator: torch.Generator,
) -> ProxyEvaluationPool:
    """Select distinct top-confidence and seeded-random absolute positions."""
    if not isinstance(confidence, torch.Tensor) or confidence.ndim != 1:
        raise ValueError("confidence must be a one-dimensional torch.Tensor.")
    if not isinstance(active_mask, torch.Tensor) or (
        active_mask.shape != confidence.shape
    ):
        raise ValueError("active_mask must be one-dimensional and match confidence.")
    _validate_nonnegative_integer(
        top_confidence_pool_size,
        name="top_confidence_pool_size",
    )
    _validate_nonnegative_integer(random_pool_size, name="random_pool_size")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator.")

    active = active_mask.detach().to(device=confidence.device) != 0
    active_positions = torch.nonzero(active, as_tuple=False).flatten()
    ranked_positions = _rank_positions(confidence, active_positions)
    top_count = min(top_confidence_pool_size, ranked_positions.numel())
    top_positions = ranked_positions[:top_count]

    top_set = set(top_positions.detach().cpu().tolist())
    remaining = [
        position
        for position in active_positions.detach().cpu().tolist()
        if position not in top_set
    ]
    random_count = min(random_pool_size, len(remaining))
    if random_count:
        permutation = torch.randperm(len(remaining), generator=generator)
        random_values = [remaining[index] for index in permutation[:random_count]]
    else:
        random_values = []
    random_positions = torch.tensor(
        random_values,
        device=confidence.device,
        dtype=torch.long,
    )

    evaluation_mask = torch.zeros_like(active, dtype=torch.bool)
    evaluation_mask[top_positions] = True
    evaluation_mask[random_positions] = True
    return ProxyEvaluationPool(
        top_confidence_positions=top_positions,
        random_positions=random_positions,
        evaluation_mask=evaluation_mask,
    )


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and clone one result tensor onto CPU for durable snapshots."""
    return tensor.detach().cpu().clone()


@torch.no_grad()
def collect_llada_proxy_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    mask_token_id: int,
    example_id: str,
    checkpoint: str,
    config: ProxyStateCollectionConfig | None = None,
    snapshot_callback: Callable[[ProxyStateSnapshot], None] | None = None,
) -> ProxyStateCollection:
    """Collect a confidence-guided trajectory at configured mask ratios."""
    config = config or ProxyStateCollectionConfig()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a torch.Tensor shaped [1, T].")
    if input_ids.shape[0] != 1:
        raise ValueError("P2.2 state collection currently requires batch size 1.")
    if input_ids.dtype == torch.bool or torch.is_floating_point(input_ids):
        raise TypeError("input_ids must contain integer token IDs.")
    if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int):
        raise TypeError("mask_token_id must be an integer.")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("example_id must be a nonempty string.")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint must be a nonempty string.")
    if snapshot_callback is not None and not callable(snapshot_callback):
        raise TypeError("snapshot_callback must be callable or None.")

    shape = tuple(input_ids.shape)
    device = input_ids.device
    for mask, name in (
        (attention_mask, "attention_mask"),
        (response_mask, "response_mask"),
    ):
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != shape:
            raise ValueError(f"{name} must match input_ids shape {shape}.")
    normalized_attention = attention_mask.detach().to(device=device)
    valid = normalized_attention != 0
    response = response_mask.detach().to(device=device) != 0
    if torch.any(response & ~valid):
        raise ValueError("response_mask must be a subset of attention_mask.")
    response_length = int(response.sum().item())
    if response_length <= 0:
        raise ValueError("response_mask must contain at least one position.")
    if not torch.all(input_ids[response] == mask_token_id):
        raise ValueError("Every initial response position must contain mask_token_id.")
    validate_proxy_state_collection_config(
        config,
        response_length=response_length,
    )
    target_counts = _target_mask_counts(response_length, config.mask_ratios)

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    state = input_ids.detach().clone()
    active = response.clone()
    snapshots = []
    for state_index, (target_ratio, target_count) in enumerate(
        zip(config.mask_ratios, target_counts)
    ):
        masked_count = int(active.sum().item())
        if masked_count != target_count:
            raise RuntimeError(
                f"Trajectory has {masked_count} masks at stage {state_index}, "
                f"expected {target_count}."
            )
        active_positions = torch.nonzero(active[0], as_tuple=False).flatten()

        with LLaDAQKCapture(
            model,
            last_n_layers=config.last_n_layers,
        ) as capture:
            base_logits = model(
                input_ids=state,
                attention_mask=normalized_attention,
            ).logits
            if capture.structure is None:
                raise RuntimeError("Capture structure is unavailable.")
            dependency_before_sink = build_active_dependency_matrix(
                capture.structure,
                capture.captures,
                active_mask=active,
                response_mask=response,
                attention_mask=normalized_attention,
                zero_diagonal=config.zero_diagonal,
                renormalize_selected_keys=(
                    config.renormalize_selected_keys
                ),
            )

        dependency_after_sink = filter_dependency_sinks(
            dependency_before_sink,
            enabled=config.sink_filter_enabled,
            sink_quantile=config.sink_quantile,
            sink_threshold=config.sink_threshold,
            renormalize_rows=True,
        )
        base_argmax = base_logits.argmax(dim=-1)
        if torch.any(base_argmax[active] == mask_token_id):
            raise RuntimeError(
                "Base argmax returned mask_token_id for an active position; "
                "the argmax reveal would not create a counterfactual state."
            )
        base_confidence = 1.0 - decoding_risk_per_token(base_logits)
        base_entropy = entropy_per_token(base_logits)

        pool = select_proxy_evaluation_pool(
            base_confidence[0],
            active[0],
            top_confidence_pool_size=config.top_confidence_pool_size,
            random_pool_size=config.random_pool_size,
            generator=generator,
        )
        oracle = evaluate_one_position_counterfactuals(
            model,
            state,
            base_logits,
            masked_active_mask=active,
            attention_mask=normalized_attention,
            evaluation_mask=pool.evaluation_mask.unsqueeze(0),
            candidate_chunk_size=config.oracle_candidate_chunk_size,
        )
        if torch.any(oracle.batch_indices != 0):
            raise RuntimeError("Batch-one oracle returned an invalid batch index.")

        dependency_valid = dependency_before_sink.query_valid_mask[0]
        dependency_positions = dependency_before_sink.query_positions[
            0,
            dependency_valid,
        ]
        if not torch.equal(dependency_positions, active_positions):
            raise RuntimeError("Dependency positions do not match active positions.")
        active_count = active_positions.numel()
        sink_mask = dependency_after_sink.sink_mask
        snapshot = ProxyStateSnapshot(
            example_id=example_id,
            state_index=state_index,
            seed=config.seed,
            target_mask_ratio=float(target_ratio),
            actual_mask_ratio=masked_count / response_length,
            response_length=response_length,
            masked_count=masked_count,
            input_ids=_cpu(state[0]),
            attention_mask=_cpu(valid[0]),
            response_mask=_cpu(response[0]),
            masked_active_mask=_cpu(active[0]),
            active_positions=_cpu(active_positions),
            committed_response_positions=_cpu(
                torch.nonzero(
                    response[0] & ~active[0],
                    as_tuple=False,
                ).flatten()
            ),
            base_argmax_token_ids=_cpu(
                base_argmax[0].index_select(0, active_positions)
            ),
            active_confidence=_cpu(
                base_confidence[0].index_select(0, active_positions)
            ),
            active_entropy=_cpu(
                base_entropy[0].index_select(0, active_positions)
            ),
            dependency_positions=_cpu(dependency_positions),
            dependency_before_sink=_cpu(
                dependency_before_sink.directed[0, :active_count, :active_count]
            ),
            dependency_after_sink=_cpu(
                dependency_after_sink.directed[0, :active_count, :active_count]
            ),
            sink_mask=(
                None
                if sink_mask is None
                else _cpu(sink_mask[0, :active_count])
            ),
            capture_layer_ids=dependency_before_sink.layer_ids,
            top_confidence_positions=_cpu(pool.top_confidence_positions),
            random_positions=_cpu(pool.random_positions),
            evaluation_positions=_cpu(oracle.positions),
            oracle_revealed_token_ids=_cpu(oracle.revealed_token_ids),
            oracle_heldout_counts=_cpu(oracle.heldout_counts),
            oracle_base_entropy_sum=_cpu(oracle.base_entropy_sum),
            oracle_lookahead_entropy_sum=_cpu(oracle.lookahead_entropy_sum),
            oracle_entropy_drop_sum=_cpu(oracle.entropy_drop_sum),
            oracle_entropy_drop_per_heldout=_cpu(
                oracle.entropy_drop_per_heldout
            ),
            oracle_base_risk_sum=_cpu(oracle.base_risk_sum),
            oracle_lookahead_risk_sum=_cpu(oracle.lookahead_risk_sum),
            oracle_risk_reduction_sum=_cpu(oracle.risk_reduction_sum),
            oracle_risk_reduction_per_heldout=_cpu(
                oracle.risk_reduction_per_heldout
            ),
        )
        snapshots.append(snapshot)
        if snapshot_callback is not None:
            snapshot_callback(snapshot)

        if state_index + 1 < len(target_counts):
            next_count = target_counts[state_index + 1]
            reveal_count = masked_count - next_count
            ranked_positions = _rank_positions(base_confidence[0], active_positions)
            reveal_positions = ranked_positions[:reveal_count]
            state[0, reveal_positions] = base_argmax[0, reveal_positions]
            active[0, reveal_positions] = False

    valid_token_count = int(valid.sum().item())
    return ProxyStateCollection(
        example_id=example_id,
        checkpoint=checkpoint,
        model_class=type(model).__name__,
        mask_token_id=mask_token_id,
        sequence_length=input_ids.shape[1],
        valid_token_count=valid_token_count,
        prompt_token_count=valid_token_count - response_length,
        response_length=response_length,
        config=config,
        states=tuple(snapshots),
    )
