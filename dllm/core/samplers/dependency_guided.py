"""Shared fixed-k dependency proposal and lookahead integration.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py -v
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import math
import time

import torch
import torch.nn as nn

from dllm.core.samplers.adaptive_cardinality import (
    ADAPTIVE_CARDINALITY_STRATEGIES,
    CARDINALITY_STRATEGIES,
    SIZE_SCORING_RULES,
    apply_size_aware_scoring,
    generate_joint_k_dependency_candidates,
    generate_stopped_soft_full_candidates,
    parse_action_sizes,
)
from dllm.core.samplers.batched_lookahead import (
    BatchedLookaheadOutput,
    LookaheadMetric,
    evaluate_batched_lookahead,
)
from dllm.core.samplers.candidates import (
    CandidateBatch,
    DEPENDENCY_DIRECTIONS,
    TARGET_WEIGHTINGS,
    compose_principled_dependency_candidates,
    deduplicate_and_refill_k1_candidates,
    generate_current_mixed_candidates,
    generate_dependency_gumbel_top_n,
    generate_dependency_top_n,
    generate_position_confidence_gumbel_candidates,
    generate_random_candidates,
)
from dllm.core.samplers.dependency import (
    DependencyCaptureOutput,
    LLaDAAttentionStructure,
    LLaDAQKCapture,
    build_active_dependency_matrix,
    filter_dependency_sinks,
)
from dllm.core.samplers.mdlm import MDLMSamplerConfig
from dllm.core.samplers.parallel_candidates import (
    CONFLICT_NORMALIZATIONS,
    PARALLEL_VARIANTS,
    CommittedAnchorState,
    gather_committed_anchor_state,
    generate_parallel_dependency_candidates,
)


PROPOSED_PROPOSAL_STRATEGY = "dependency"
BASELINE_PROPOSAL_STRATEGIES = (
    "baseline_random",
    "baseline_current_mixed",
    "baseline_confidence_gumbel",
)
SUPPORTED_PROPOSAL_STRATEGIES = (
    "legacy",
    PROPOSED_PROPOSAL_STRATEGY,
    *BASELINE_PROPOSAL_STRATEGIES,
)
SUPPORTED_DEPENDENCY_FALLBACKS = ("dependency_only",)


@dataclass
class DependencyGuidedSamplerConfig(MDLMSamplerConfig):
    """Configuration shared by the entropy and risk fixed-k decoders."""

    # Disabled by default so existing sampler behavior is unchanged.
    proposal_strategy: str = "legacy"
    candidate_budget: int = 4
    dependency_last_n_layers: int = 4
    dependency_direction: str = "outgoing"
    dependency_target_weighting: str = "entropy"
    dependency_position_temperature: float = 1.0
    dependency_confidence_exponent: float = 0.0
    dependency_generation_seed: int = 42
    dependency_sink_filter_enabled: bool = True
    dependency_sink_quantile: float = 0.99
    dependency_sink_threshold: float | None = None
    dependency_zero_diagonal: bool = True
    dependency_renormalize_selected_keys: bool = True
    dependency_fallback_strategy: str = "dependency_only"
    dependency_commit_k: int = 1
    dependency_parallel_variant: str = "soft_full"
    dependency_conflict_normalization: str = "max"
    dependency_conflict_penalty: float = 1.0
    dependency_hard_conflict_threshold: float = 0.25
    dependency_anchor_support_weight: float = 1.0
    dependency_anchor_confidence_threshold: float = 0.8
    dependency_cardinality_strategy: str = "fixed"
    dependency_max_action_size: int = 4
    dependency_action_sizes: str = "1|2|4"
    dependency_utility_threshold: float = 0.0
    dependency_entropy_budget: float = 1.0
    dependency_size_scoring: str = "raw"
    dependency_immediate_cost_weight: float = 1.0
    dependency_size_penalty: float = 0.0
    candidate_chunk_size: int | None = None
    diagnostic_metadata: bool = False


@dataclass(frozen=True)
class DependencyBaseForwardOutput:
    """Base logits plus capture artifacts retained after hooks are removed."""

    logits: torch.Tensor
    structure: LLaDAAttentionStructure | None
    captures: Mapping[int, Mapping[str, torch.Tensor]] | None
    dependency_capture_source: str | None
    capture_batch_size: int
    captured_base_forward_count: int
    capture_active_after_forward: bool
    base_forward_seconds: float


@dataclass(frozen=True)
class FixedKSelectionOutput:
    """Candidates, verifier output, dependency data, and timing for one step."""

    candidates: CandidateBatch
    lookahead: BatchedLookaheadOutput
    dependency: DependencyCaptureOutput | None
    attention_reconstruction_seconds: float
    proposal_generation_seconds: float
    candidate_lookahead_seconds: float
    lookahead_capture_forward_count: int = 0
    capture_tensors_released_before_lookahead: bool = True
    raw_lookahead_scores: torch.Tensor | None = None
    immediate_action_costs: torch.Tensor | None = None
    size_scoring_rule: str = "raw"


def resolve_dependency_guided_config(
    config: DependencyGuidedSamplerConfig,
    overrides: Mapping[str, object],
) -> DependencyGuidedSamplerConfig:
    """Apply sampler-call overrides to shared dependency-decoder fields."""
    shared_names = {
        field.name for field in fields(DependencyGuidedSamplerConfig)
    }
    updates = {
        name: value for name, value in overrides.items() if name in shared_names
    }
    resolved = replace(config, **updates)
    if resolved.proposal_strategy != "legacy" and resolved.candidate_chunk_size is None:
        # In bf16, changing verifier batch width changed a real Phase-5 winner.
        # Keep fixed-k comparisons on the sequential candidate-width reference
        # unless a caller explicitly opts into approximate wider batching.
        resolved = replace(resolved, candidate_chunk_size=1)
    validate_dependency_guided_config(resolved)
    return resolved


def validate_dependency_guided_config(config: DependencyGuidedSamplerConfig) -> None:
    """Fail before decoding when fixed-k configuration is unsupported."""
    if config.proposal_strategy not in SUPPORTED_PROPOSAL_STRATEGIES:
        raise ValueError(
            f"Unknown proposal_strategy {config.proposal_strategy!r}. Available: "
            f"{', '.join(SUPPORTED_PROPOSAL_STRATEGIES)}."
        )
    if (
        isinstance(config.candidate_budget, bool)
        or not isinstance(config.candidate_budget, int)
        or config.candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if (
        isinstance(config.dependency_commit_k, bool)
        or not isinstance(config.dependency_commit_k, int)
        or config.dependency_commit_k <= 0
    ):
        raise ValueError("dependency_commit_k must be a positive integer.")
    if config.dependency_parallel_variant not in PARALLEL_VARIANTS:
        raise ValueError(
            "Unknown dependency_parallel_variant "
            f"{config.dependency_parallel_variant!r}. Available: "
            f"{', '.join(PARALLEL_VARIANTS)}."
        )
    if config.dependency_cardinality_strategy not in CARDINALITY_STRATEGIES:
        raise ValueError(
            "Unknown dependency_cardinality_strategy "
            f"{config.dependency_cardinality_strategy!r}. Available: "
            f"{', '.join(CARDINALITY_STRATEGIES)}."
        )
    if config.dependency_size_scoring not in SIZE_SCORING_RULES:
        raise ValueError(
            "Unknown dependency_size_scoring "
            f"{config.dependency_size_scoring!r}. Available: "
            f"{', '.join(SIZE_SCORING_RULES)}."
        )
    if (
        isinstance(config.dependency_max_action_size, bool)
        or not isinstance(config.dependency_max_action_size, int)
        or config.dependency_max_action_size <= 0
    ):
        raise ValueError("dependency_max_action_size must be a positive integer.")
    action_sizes = parse_action_sizes(config.dependency_action_sizes)
    if max(action_sizes) > config.dependency_max_action_size:
        raise ValueError(
            "dependency_action_sizes cannot exceed dependency_max_action_size."
        )
    if config.dependency_cardinality_strategy != "fixed":
        if config.proposal_strategy != PROPOSED_PROPOSAL_STRATEGY:
            raise ValueError(
                "Adaptive cardinality requires proposal_strategy='dependency'."
            )
        if config.dependency_parallel_variant != "soft_full":
            raise ValueError(
                "Adaptive cardinality uses the frozen soft_full construction."
            )
    if (
        config.dependency_cardinality_strategy in ADAPTIVE_CARDINALITY_STRATEGIES
        and config.dependency_size_scoring == "raw"
    ):
        raise ValueError(
            "Variable-size adaptive candidates require explicit size-aware scoring."
        )
    if config.dependency_conflict_normalization not in CONFLICT_NORMALIZATIONS:
        raise ValueError(
            "Unknown dependency_conflict_normalization "
            f"{config.dependency_conflict_normalization!r}. Available: "
            f"{', '.join(CONFLICT_NORMALIZATIONS)}."
        )
    if (
        isinstance(config.dependency_last_n_layers, bool)
        or not isinstance(config.dependency_last_n_layers, int)
        or config.dependency_last_n_layers <= 0
    ):
        raise ValueError("dependency_last_n_layers must be a positive integer.")
    if config.dependency_fallback_strategy not in SUPPORTED_DEPENDENCY_FALLBACKS:
        raise ValueError(
            "Unknown dependency_fallback_strategy "
            f"{config.dependency_fallback_strategy!r}. Available: "
            f"{', '.join(SUPPORTED_DEPENDENCY_FALLBACKS)}."
        )
    if config.dependency_direction not in DEPENDENCY_DIRECTIONS:
        raise ValueError(
            f"Unknown dependency_direction {config.dependency_direction!r}. "
            f"Available: {', '.join(DEPENDENCY_DIRECTIONS)}."
        )
    if config.dependency_target_weighting not in TARGET_WEIGHTINGS:
        raise ValueError(
            "Unknown dependency_target_weighting "
            f"{config.dependency_target_weighting!r}. Available: "
            f"{', '.join(TARGET_WEIGHTINGS)}."
        )
    if config.candidate_chunk_size is not None and (
        isinstance(config.candidate_chunk_size, bool)
        or not isinstance(config.candidate_chunk_size, int)
        or config.candidate_chunk_size <= 0
    ):
        raise ValueError("candidate_chunk_size must be a positive integer or None.")
    for name, value in (
        ("dependency_position_temperature", config.dependency_position_temperature),
        ("dependency_confidence_exponent", config.dependency_confidence_exponent),
        ("dependency_conflict_penalty", config.dependency_conflict_penalty),
        (
            "dependency_hard_conflict_threshold",
            config.dependency_hard_conflict_threshold,
        ),
        (
            "dependency_anchor_support_weight",
            config.dependency_anchor_support_weight,
        ),
        ("dependency_entropy_budget", config.dependency_entropy_budget),
        (
            "dependency_immediate_cost_weight",
            config.dependency_immediate_cost_weight,
        ),
        ("dependency_size_penalty", config.dependency_size_penalty),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    utility_threshold = config.dependency_utility_threshold
    if (
        isinstance(utility_threshold, bool)
        or not isinstance(utility_threshold, (int, float))
        or not math.isfinite(float(utility_threshold))
    ):
        raise ValueError("dependency_utility_threshold must be finite.")
    threshold = config.dependency_anchor_confidence_threshold
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "dependency_anchor_confidence_threshold must be finite and in [0,1]."
        )
    if (
        isinstance(config.dependency_generation_seed, bool)
        or not isinstance(config.dependency_generation_seed, int)
        or config.dependency_generation_seed < 0
    ):
        raise ValueError("dependency_generation_seed must be a nonnegative integer.")
    if not isinstance(config.dependency_sink_filter_enabled, bool):
        raise TypeError("dependency_sink_filter_enabled must be boolean.")
    if not isinstance(config.dependency_zero_diagonal, bool):
        raise TypeError("dependency_zero_diagonal must be boolean.")
    if not isinstance(config.dependency_renormalize_selected_keys, bool):
        raise TypeError("dependency_renormalize_selected_keys must be boolean.")
    if (
        isinstance(config.dependency_sink_quantile, bool)
        or not isinstance(config.dependency_sink_quantile, (int, float))
        or not math.isfinite(float(config.dependency_sink_quantile))
        or not 0 <= config.dependency_sink_quantile <= 1
    ):
        raise ValueError("dependency_sink_quantile must be finite and in [0, 1].")
    if config.dependency_sink_threshold is not None and (
        isinstance(config.dependency_sink_threshold, bool)
        or not isinstance(config.dependency_sink_threshold, (int, float))
        or not math.isfinite(float(config.dependency_sink_threshold))
        or config.dependency_sink_threshold < 0
    ):
        raise ValueError(
            "dependency_sink_threshold must be finite and nonnegative or None."
        )
    if not isinstance(config.diagnostic_metadata, bool):
        raise TypeError("diagnostic_metadata must be boolean.")
    if config.diagnostic_metadata and not config.return_dict:
        raise ValueError("diagnostic_metadata=True requires return_dict=True.")


def is_fixed_k_strategy(strategy: str) -> bool:
    """Return whether a strategy uses a fixed-cardinality CandidateBatch path."""
    return strategy != "legacy"


def validate_fixed_k_schedule(
    num_transfer_tokens: torch.Tensor,
    commit_k: int = 1,
) -> None:
    """Require fixed k on every nonfinal action and feasible clipping at the end."""
    if not isinstance(num_transfer_tokens, torch.Tensor) or num_transfer_tokens.ndim != 2:
        raise ValueError("num_transfer_tokens must have shape [B,S].")
    if num_transfer_tokens.dtype == torch.bool or num_transfer_tokens.is_floating_point():
        raise TypeError("num_transfer_tokens must use an integer dtype.")
    if isinstance(commit_k, bool) or not isinstance(commit_k, int) or commit_k <= 0:
        raise ValueError("commit_k must be a positive integer.")
    if torch.any(num_transfer_tokens < 0):
        raise ValueError("num_transfer_tokens must be nonnegative.")
    for row in num_transfer_tokens:
        positive = row[row > 0]
        if positive.numel() == 0:
            continue
        if torch.any(positive > commit_k) or (
            positive.numel() > 1 and torch.any(positive[:-1] != commit_k)
        ):
            maximum = int(positive.max().item())
            action_description = (
                "exactly one token"
                if commit_k == 1
                else f"exactly {commit_k} tokens"
            )
            raise ValueError(
                f"Fixed-k proposal strategies require {action_description} "
                f"(k={commit_k}) on every "
                "nonfinal action and permit only a smaller final feasible action, "
                f"but the scheduler requested values up to {maximum}. Adjust steps "
                "or use proposal_strategy='legacy'."
            )


def _synchronize_for_timing(tensor: torch.Tensor, enabled: bool) -> None:
    """Synchronize only when diagnostic timings were requested."""
    if enabled and tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _conditional_capture_rows(
    captures: Mapping[int, Mapping[str, torch.Tensor]],
    batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Retain the conditional (first) half of a CFG capture."""
    sliced: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id, projections in captures.items():
        sliced[layer_id] = {}
        for projection_name, tensor in projections.items():
            if tensor.shape[0] < batch_size:
                raise RuntimeError("Captured projection batch is smaller than input batch.")
            sliced[layer_id][projection_name] = tensor[:batch_size]
    return sliced


@torch.no_grad()
def run_base_forward_with_cfg_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    cfg_scale: float,
    unconditional_input_ids: torch.Tensor | None,
    capture_dependency: bool,
    dependency_last_n_layers: int,
    measure_timing: bool,
) -> DependencyBaseForwardOutput:
    """Run the base forward and source CFG dependency from the conditional half."""
    _synchronize_for_timing(input_ids, measure_timing)
    started_at = time.perf_counter()
    structure: LLaDAAttentionStructure | None = None
    captures: Mapping[int, Mapping[str, torch.Tensor]] | None = None
    capture_source: str | None = None
    capture_batch_size = input_ids.shape[0]

    if cfg_scale > 0.0:
        if unconditional_input_ids is None:
            raise ValueError("unconditional_input_ids is required when cfg_scale > 0.")
        forward_ids = torch.cat([input_ids, unconditional_input_ids], dim=0)
        forward_attention = attention_mask.repeat(2, 1)
    else:
        forward_ids = input_ids
        forward_attention = attention_mask

    if capture_dependency:
        capture = LLaDAQKCapture(model, last_n_layers=dependency_last_n_layers)
        with capture:
            raw_logits = model(
                forward_ids,
                attention_mask=forward_attention,
            ).logits
            structure = capture.structure
            captured = capture.captures
            capture_batch_size = forward_ids.shape[0]
            if cfg_scale > 0.0:
                captures = _conditional_capture_rows(captured, input_ids.shape[0])
                capture_source = "conditional_cfg_half"
            else:
                captures = captured
                capture_source = "conditional"
        capture_active_after = capture.active
        if capture_active_after:
            raise RuntimeError("Dependency capture remained active after base forward.")
    else:
        raw_logits = model(forward_ids, attention_mask=forward_attention).logits
        capture_active_after = False

    if cfg_scale > 0.0:
        conditional_logits, unconditional_logits = torch.chunk(raw_logits, 2, dim=0)
        logits = unconditional_logits + (cfg_scale + 1) * (
            conditional_logits - unconditional_logits
        )
    else:
        logits = raw_logits

    _synchronize_for_timing(input_ids, measure_timing)
    return DependencyBaseForwardOutput(
        logits=logits,
        structure=structure,
        captures=captures,
        dependency_capture_source=capture_source,
        capture_batch_size=capture_batch_size,
        captured_base_forward_count=1,
        capture_active_after_forward=capture_active_after,
        base_forward_seconds=time.perf_counter() - started_at,
    )


def _gather_compact_values(
    values: torch.Tensor,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather [B,T] values into a padded [B,M] absolute-position mapping."""
    safe_positions = positions.clamp_min(0)
    gathered = torch.gather(values, dim=1, index=safe_positions)
    return torch.where(valid_mask, gathered, torch.zeros_like(gathered))


def remap_compact_candidates_to_sequence(
    candidates: CandidateBatch,
    dependency: DependencyCaptureOutput,
    *,
    sequence_length: int,
) -> CandidateBatch:
    """Map compact active-position candidate indexes back to absolute tokens."""
    candidate_count, batch_size, _ = candidates.shape
    if dependency.query_positions.shape != candidates.eligible_mask.shape:
        raise ValueError("Dependency positions must match compact candidate space.")
    full_masks = torch.zeros(
        (candidate_count, batch_size, sequence_length),
        device=candidates.candidate_masks.device,
        dtype=torch.bool,
    )
    storage_width = candidates.selected_positions.shape[-1]
    full_positions = torch.full(
        (candidate_count, batch_size, storage_width),
        -1,
        device=candidates.candidate_masks.device,
        dtype=torch.long,
    )
    full_anchors = torch.full_like(candidates.seed_anchors, -1)
    full_eligible = torch.zeros(
        (batch_size, sequence_length),
        device=candidates.candidate_masks.device,
        dtype=torch.bool,
    )
    for batch_index in range(batch_size):
        compact_valid = candidates.eligible_mask[batch_index]
        absolute_valid = dependency.query_positions[batch_index, compact_valid]
        full_eligible[batch_index, absolute_valid] = True
        for candidate_index in range(candidate_count):
            if not bool(candidates.candidate_valid[candidate_index, batch_index]):
                continue
            compact = candidates.selected_positions[candidate_index, batch_index]
            compact = compact[compact >= 0]
            absolute = dependency.query_positions[batch_index, compact]
            absolute = torch.sort(absolute).values
            full_masks[candidate_index, batch_index, absolute] = True
            full_positions[
                candidate_index, batch_index, : absolute.numel()
            ] = absolute
            compact_anchor = candidates.seed_anchors[candidate_index, batch_index]
            full_anchors[candidate_index, batch_index] = dependency.query_positions[
                batch_index, compact_anchor
            ]

    configuration = dict(candidates.configuration)
    configuration.update(
        {
            "candidate_position_space": "absolute_sequence",
            "dependency_matrix_position_space": "compact_active_response",
        }
    )
    return replace(
        candidates,
        candidate_masks=full_masks,
        eligible_mask=full_eligible,
        selected_positions=full_positions,
        seed_anchors=full_anchors,
        metadata=tuple(
            {**metadata, "positions_remapped_to_absolute_sequence": True}
            for metadata in candidates.metadata
        ),
        configuration=configuration,
    )


def _build_dependency_candidates(
    base_forward: DependencyBaseForwardOutput,
    *,
    active_mask: torch.Tensor,
    requested_k: torch.Tensor,
    anchor_state: CommittedAnchorState | None,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    entropy_map: torch.Tensor,
    confidence: torch.Tensor,
    config: DependencyGuidedSamplerConfig,
    generation_seed: int,
) -> tuple[CandidateBatch, DependencyCaptureOutput, float, float]:
    """Reconstruct dependency and build a k=1 or parallel dependency pool."""
    if base_forward.structure is None or base_forward.captures is None:
        raise RuntimeError("Dependency proposal requested without base-forward capture.")
    # A Phase-6 fixed run stays on the parallel path when k > 1. Every Phase-7
    # cardinality policy uses the same frozen soft-full construction, including
    # steps whose realized action happens to contain one position.
    cardinality_strategy = config.dependency_cardinality_strategy
    parallel_action = (
        config.dependency_commit_k > 1 or cardinality_strategy != "fixed"
    )
    dependency_active_mask = active_mask
    if parallel_action and anchor_state is not None:
        if anchor_state.committed_position_mask.shape != active_mask.shape:
            raise ValueError("anchor_state must match the absolute sequence shape.")
        # Low-confidence commits remain in the audit history but are excluded
        # from both support and the reconstructed dependency position space.
        dependency_active_mask = active_mask | anchor_state.reliable_anchor_mask
    _synchronize_for_timing(active_mask, config.diagnostic_metadata)
    reconstruction_started_at = time.perf_counter()
    dependency = build_active_dependency_matrix(
        base_forward.structure,
        dict(base_forward.captures),
        active_mask=dependency_active_mask,
        response_mask=response_mask,
        attention_mask=attention_mask,
        zero_diagonal=config.dependency_zero_diagonal,
        renormalize_selected_keys=config.dependency_renormalize_selected_keys,
    )
    dependency = filter_dependency_sinks(
        dependency,
        enabled=config.dependency_sink_filter_enabled,
        sink_quantile=config.dependency_sink_quantile,
        sink_threshold=config.dependency_sink_threshold,
        renormalize_rows=True,
    )
    _synchronize_for_timing(active_mask, config.diagnostic_metadata)
    reconstruction_seconds = time.perf_counter() - reconstruction_started_at
    proposal_started_at = time.perf_counter()
    compact_entropy = _gather_compact_values(
        entropy_map.float(),
        dependency.query_positions,
        dependency.query_valid_mask,
    )
    compact_confidence = _gather_compact_values(
        confidence.float(),
        dependency.query_positions,
        dependency.query_valid_mask,
    )
    compact_eligible = _gather_compact_values(
        active_mask,
        dependency.query_positions,
        dependency.query_valid_mask,
    ).bool()
    if parallel_action:
        compact_anchor_state = (
            gather_committed_anchor_state(
                anchor_state,
                dependency.query_positions,
                dependency.query_valid_mask,
            )
            if anchor_state is not None
            else None
        )
        shared_parallel_kwargs = {
            "direction": config.dependency_direction,
            "target_weighting": config.dependency_target_weighting,
            "confidence_exponent": config.dependency_confidence_exponent,
            "anchor_state": compact_anchor_state,
            "conflict_normalization": config.dependency_conflict_normalization,
            "conflict_penalty": config.dependency_conflict_penalty,
            "anchor_support_weight": config.dependency_anchor_support_weight,
            "position_temperature": config.dependency_position_temperature,
            "generation_seed": generation_seed,
        }
        if cardinality_strategy == "joint_k":
            compact_candidates = generate_joint_k_dependency_candidates(
                dependency.directed,
                compact_entropy,
                compact_confidence,
                compact_eligible,
                action_sizes=config.dependency_action_sizes,
                candidate_budget_per_size=config.candidate_budget,
                hard_conflict_threshold=(
                    config.dependency_hard_conflict_threshold
                ),
                name_prefix="joint_candidate",
                **shared_parallel_kwargs,
            )
        elif cardinality_strategy in {
            "marginal_utility",
            "entropy_budget",
        }:
            compact_candidates = generate_stopped_soft_full_candidates(
                dependency.directed,
                compact_entropy,
                compact_confidence,
                compact_eligible,
                candidate_budget=config.candidate_budget,
                maximum_action_size=config.dependency_max_action_size,
                stopping_rule=cardinality_strategy,
                utility_threshold=config.dependency_utility_threshold,
                entropy_budget=config.dependency_entropy_budget,
                name_prefix="adaptive_candidate",
                **shared_parallel_kwargs,
            )
        else:
            compact_candidates = generate_parallel_dependency_candidates(
                dependency.directed,
                compact_entropy,
                compact_confidence,
                compact_eligible,
                requested_k=requested_k,
                candidate_budget=config.candidate_budget,
                variant=config.dependency_parallel_variant,
                hard_conflict_threshold=(
                    config.dependency_hard_conflict_threshold
                ),
                name_prefix="parallel_candidate",
                **shared_parallel_kwargs,
            )
    else:
        # Keep the accepted Phase-5 k=1 composer byte-for-byte on its old path.
        deterministic = generate_dependency_top_n(
            dependency.directed,
            compact_entropy,
            compact_eligible,
            candidate_budget=config.candidate_budget,
            direction=config.dependency_direction,
            target_weighting=config.dependency_target_weighting,
            confidence=compact_confidence,
            confidence_exponent=config.dependency_confidence_exponent,
            generation_seed=generation_seed,
            name_prefix="dependency_deterministic",
        )
        diverse = generate_dependency_gumbel_top_n(
            dependency.directed,
            compact_entropy,
            compact_eligible,
            candidate_budget=config.candidate_budget,
            position_temperature=config.dependency_position_temperature,
            generation_seed=generation_seed,
            direction=config.dependency_direction,
            target_weighting=config.dependency_target_weighting,
            confidence=compact_confidence,
            confidence_exponent=config.dependency_confidence_exponent,
            name_prefix="dependency_gumbel",
        )
        compact_candidates = compose_principled_dependency_candidates(
            deterministic,
            diverse,
            candidate_budget=config.candidate_budget,
            generation_seed=generation_seed,
            name_prefix="dependency_candidate",
        )
    candidates = remap_compact_candidates_to_sequence(
        compact_candidates,
        dependency,
        sequence_length=active_mask.shape[1],
    )
    _synchronize_for_timing(active_mask, config.diagnostic_metadata)
    proposal_seconds = time.perf_counter() - proposal_started_at
    return candidates, dependency, reconstruction_seconds, proposal_seconds


def _build_baseline_candidates(
    strategy: str,
    *,
    active_mask: torch.Tensor,
    confidence: torch.Tensor,
    config: DependencyGuidedSamplerConfig,
    generation_seed: int,
) -> CandidateBatch:
    """Build explicitly baseline-only pools used by the Phase-5 comparison."""
    if strategy == "baseline_random":
        return generate_random_candidates(
            active_mask,
            candidate_budget=config.candidate_budget,
            generation_seed=generation_seed,
            name_prefix="baseline_random",
        )
    if strategy == "baseline_confidence_gumbel":
        return generate_position_confidence_gumbel_candidates(
            confidence.float(),
            active_mask,
            candidate_budget=config.candidate_budget,
            position_temperature=config.dependency_position_temperature,
            generation_seed=generation_seed,
            name_prefix="baseline_confidence_gumbel",
        )
    if strategy == "baseline_current_mixed":
        mixed = generate_current_mixed_candidates(
            confidence.float(),
            active_mask,
            generation_seed=generation_seed,
        )
        return deduplicate_and_refill_k1_candidates(
            (mixed,),
            candidate_budget=config.candidate_budget,
            dependency_fallback=None,
            confidence_fallback=None,
            generation_seed=generation_seed,
            name_prefix="baseline_current_mixed",
            allow_random_fallback=True,
        )
    raise ValueError(f"Unsupported baseline strategy: {strategy!r}.")


def _release_capture_tensors(base_forward: DependencyBaseForwardOutput) -> bool:
    """Release retained Q/K references before materializing lookahead logits."""
    captures = base_forward.captures
    if captures is None:
        return True
    if not isinstance(captures, dict):
        raise TypeError("Internal dependency captures must be a mutable dictionary.")
    for projections in captures.values():
        if not isinstance(projections, dict):
            raise TypeError("Internal projection captures must be mutable dictionaries.")
        projections.clear()
    captures.clear()
    return not captures


@torch.no_grad()
def select_fixed_k_candidate(
    model: nn.Module,
    input_ids: torch.Tensor,
    predicted_token_ids: torch.Tensor,
    *,
    base_forward: DependencyBaseForwardOutput,
    base_metric_map: torch.Tensor,
    entropy_map: torch.Tensor,
    confidence: torch.Tensor,
    metric: LookaheadMetric,
    active_mask: torch.Tensor,
    requested_k: torch.Tensor,
    anchor_state: CommittedAnchorState | None,
    masked_active_mask: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    config: DependencyGuidedSamplerConfig,
    generation_seed: int,
) -> FixedKSelectionOutput:
    """Build one shared fixed-k pool and select it with either verifier metric."""
    measure = config.diagnostic_metadata
    _synchronize_for_timing(input_ids, measure)
    proposal_started_at = time.perf_counter()
    dependency: DependencyCaptureOutput | None = None
    if config.proposal_strategy == PROPOSED_PROPOSAL_STRATEGY:
        (
            candidates,
            dependency,
            reconstruction_seconds,
            proposal_seconds,
        ) = _build_dependency_candidates(
            base_forward,
            active_mask=active_mask,
            requested_k=requested_k,
            anchor_state=anchor_state,
            response_mask=response_mask,
            attention_mask=attention_mask,
            entropy_map=entropy_map,
            confidence=confidence,
            config=config,
            generation_seed=generation_seed,
        )
        captures_released = _release_capture_tensors(base_forward)
    else:
        reconstruction_seconds = 0.0
        candidates = _build_baseline_candidates(
            config.proposal_strategy,
            active_mask=active_mask,
            confidence=confidence,
            config=config,
            generation_seed=generation_seed,
        )
        _synchronize_for_timing(input_ids, measure)
        proposal_seconds = time.perf_counter() - proposal_started_at
        captures_released = True

    if base_forward.capture_active_after_forward:
        raise RuntimeError("Lookahead cannot start while dependency capture is active.")
    if not captures_released:
        raise RuntimeError("Captured Q/K tensors remained live before lookahead.")
    _synchronize_for_timing(input_ids, measure)
    lookahead_started_at = time.perf_counter()
    raw_lookahead = evaluate_batched_lookahead(
        model,
        input_ids,
        predicted_token_ids,
        candidates,
        base_metric_map=base_metric_map,
        metric=metric,
        attention_mask=attention_mask,
        masked_active_mask=masked_active_mask,
        candidate_chunk_size=config.candidate_chunk_size,
    )
    size_scoring = apply_size_aware_scoring(
        raw_lookahead,
        candidates,
        base_metric_map,
        rule=config.dependency_size_scoring,
        immediate_cost_weight=config.dependency_immediate_cost_weight,
        size_penalty=config.dependency_size_penalty,
    )
    lookahead = size_scoring.lookahead
    _synchronize_for_timing(input_ids, measure)
    lookahead_seconds = time.perf_counter() - lookahead_started_at
    return FixedKSelectionOutput(
        candidates=candidates,
        lookahead=lookahead,
        dependency=dependency,
        attention_reconstruction_seconds=reconstruction_seconds,
        proposal_generation_seconds=proposal_seconds,
        candidate_lookahead_seconds=lookahead_seconds,
        capture_tensors_released_before_lookahead=captures_released,
        raw_lookahead_scores=size_scoring.raw_scores,
        immediate_action_costs=size_scoring.immediate_costs,
        size_scoring_rule=size_scoring.rule,
    )


def _metadata_value_for_batch(
    metadata: Mapping[str, object],
    name: str,
    batch_index: int,
) -> object | None:
    """Resolve either a scalar metadata value or its per-batch tuple."""
    by_batch = metadata.get(f"{name}_by_batch")
    if isinstance(by_batch, (tuple, list)) and batch_index < len(by_batch):
        return by_batch[batch_index]
    return metadata.get(name)


def build_step_diagnostics(
    selection: FixedKSelectionOutput,
    base_forward: DependencyBaseForwardOutput,
    *,
    config: DependencyGuidedSamplerConfig,
    metric: LookaheadMetric,
    masked_active_mask: torch.Tensor,
    response_mask: torch.Tensor,
    block_index: int,
    step_index: int,
    global_step_index: int,
    generation_seed: int,
    base_metric_map: torch.Tensor | None = None,
    predicted_token_ids: torch.Tensor | None = None,
    anchor_state_before: CommittedAnchorState | None = None,
    anchor_state_after: CommittedAnchorState | None = None,
) -> list[dict[str, object]]:
    """Create JSON-serializable per-example records for one decoder step."""
    candidates = selection.candidates
    lookahead = selection.lookahead
    diagnostics: list[dict[str, object]] = []
    precision_reference = 0.05759 if metric == "entropy_drop" else 0.007165
    for batch_index in range(masked_active_mask.shape[0]):
        valid_indexes = torch.where(candidates.candidate_valid[:, batch_index])[0]
        valid_scores = lookahead.scores[valid_indexes, batch_index]
        finite_scores = valid_scores[torch.isfinite(valid_scores)]
        if finite_scores.numel() >= 2:
            ordered = torch.sort(finite_scores, descending=True).values
            winning_margin = float((ordered[0] - ordered[1]).item())
        else:
            winning_margin = None
        best_index = int(lookahead.best_index[batch_index].item())
        if best_index >= 0:
            heldout_count = int(
                lookahead.heldout_counts[best_index, batch_index].item()
            )
            margin_per_heldout = (
                None
                if winning_margin is None
                else winning_margin / max(heldout_count, 1)
            )
        else:
            heldout_count = 0
            margin_per_heldout = None

        candidate_records = []
        for candidate_index, name in enumerate(candidates.names):
            valid = bool(candidates.candidate_valid[candidate_index, batch_index])
            positions = candidates.selected_positions[candidate_index, batch_index]
            metadata = candidates.metadata[candidate_index]
            score = lookahead.scores[candidate_index, batch_index]
            raw_score = (
                selection.raw_lookahead_scores[candidate_index, batch_index]
                if selection.raw_lookahead_scores is not None
                else score
            )
            immediate_action_cost = (
                selection.immediate_action_costs[candidate_index, batch_index]
                if selection.immediate_action_costs is not None
                else None
            )
            candidate_heldout_count = int(
                lookahead.heldout_counts[candidate_index, batch_index].item()
            )
            proposal_score = candidates.proposal_scores[
                candidate_index, batch_index
            ]
            candidate_records.append(
                {
                    "index": candidate_index,
                    "name": name,
                    "valid": valid,
                    "positions": [
                        int(position)
                        for position in positions[positions >= 0].tolist()
                    ],
                    "action_size": int(
                        candidates.candidate_masks[
                            candidate_index, batch_index
                        ].sum().item()
                    ),
                    "proposal_score": (
                        float(proposal_score.item()) if valid else None
                    ),
                    "verifier_score": float(score.item()) if valid else None,
                    "raw_verifier_score": (
                        float(raw_score.item()) if valid else None
                    ),
                    "size_aware_verifier_score": (
                        float(score.item()) if valid else None
                    ),
                    "immediate_action_cost": (
                        float(immediate_action_cost.item())
                        if valid and immediate_action_cost is not None
                        else None
                    ),
                    "verifier_score_per_heldout": (
                        float(score.item()) / max(candidate_heldout_count, 1)
                        if valid
                        else None
                    ),
                    "heldout_count": candidate_heldout_count,
                    "source": _metadata_value_for_batch(
                        metadata, "source", batch_index
                    ),
                    "fallback": _metadata_value_for_batch(
                        metadata, "is_fallback", batch_index
                    ),
                    "fallback_source": _metadata_value_for_batch(
                        metadata, "fallback_source", batch_index
                    ),
                    "mean_within_set_conflict": float(
                        candidates.mean_within_set_dependency[
                            candidate_index, batch_index
                        ].item()
                    ),
                    "max_within_set_conflict": _metadata_value_for_batch(
                        metadata, "max_within_set_conflict", batch_index
                    ),
                    "anchor_support_sum": _metadata_value_for_batch(
                        metadata, "anchor_support_sum", batch_index
                    ),
                    "hard_fallback_count": _metadata_value_for_batch(
                        metadata, "hard_fallback_count", batch_index
                    ),
                    "stopping_reason": _metadata_value_for_batch(
                        metadata, "stopping_reason", batch_index
                    ),
                    "requested_action_size": _metadata_value_for_batch(
                        metadata, "requested_action_size", batch_index
                    ),
                }
            )
            if (
                candidate_records[-1]["fallback"] is True
                and candidate_records[-1]["fallback_source"] is None
            ):
                candidate_records[-1]["fallback_source"] = candidate_records[-1][
                    "source"
                ]
            elif candidate_records[-1]["fallback"] is not True:
                candidate_records[-1]["fallback_source"] = None

        response_count = int(response_mask[batch_index].sum().item())
        remaining_count = int(
            (masked_active_mask[batch_index] & response_mask[batch_index]).sum().item()
        )
        action_space_size = int(candidates.eligible_mask[batch_index].sum().item())
        requested_commit_k = int(candidates.clipped_k[batch_index].item())
        configured_action_set_counts = candidates.configuration.get(
            "candidate_action_set_count_by_batch"
        )
        if isinstance(configured_action_set_counts, (tuple, list)):
            candidate_action_set_count = int(
                configured_action_set_counts[batch_index]
            )
        else:
            candidate_action_set_count = (
                math.comb(action_space_size, requested_commit_k)
                if requested_commit_k > 0
                else 0
            )
        configured_expected_counts = candidates.configuration.get(
            "expected_candidate_count_by_batch"
        )
        if isinstance(configured_expected_counts, (tuple, list)):
            expected_candidate_count = int(
                configured_expected_counts[batch_index]
            )
        else:
            expected_candidate_count = min(
                config.candidate_budget,
                candidate_action_set_count,
            )
        dependency = selection.dependency
        sink_count = (
            int(dependency.sink_mask[batch_index].sum().item())
            if dependency is not None and dependency.sink_mask is not None
            else 0
        )
        selected_record = (
            candidate_records[best_index] if best_index >= 0 else None
        )
        if base_metric_map is not None:
            current_metric_mask = (
                masked_active_mask[batch_index] & response_mask[batch_index]
            )
            current_metric_values = base_metric_map[batch_index, current_metric_mask]
            current_metric_sum = float(current_metric_values.float().sum().item())
            current_metric_mean = (
                float(current_metric_values.float().mean().item())
                if current_metric_values.numel()
                else 0.0
            )
        else:
            current_metric_sum = None
            current_metric_mean = None
        committed_before = (
            int(anchor_state_before.committed_position_mask[batch_index].sum().item())
            if anchor_state_before is not None
            else 0
        )
        reliable_before = (
            int(anchor_state_before.reliable_anchor_mask[batch_index].sum().item())
            if anchor_state_before is not None
            else 0
        )
        committed_after = (
            int(anchor_state_after.committed_position_mask[batch_index].sum().item())
            if anchor_state_after is not None
            else committed_before
        )
        reliable_after = (
            int(anchor_state_after.reliable_anchor_mask[batch_index].sum().item())
            if anchor_state_after is not None
            else reliable_before
        )
        immediate_consistency_count = 0
        immediate_consistency_total = 0
        immediate_consistency_rate = None
        if anchor_state_before is not None and predicted_token_ids is not None:
            previous_commits = (
                anchor_state_before.committed_position_mask[batch_index]
                & (
                    anchor_state_before.commit_step[batch_index]
                    == global_step_index - 1
                )
            )
            immediate_consistency_total = int(previous_commits.sum().item())
            if immediate_consistency_total:
                stable = (
                    predicted_token_ids[batch_index]
                    == anchor_state_before.token_id_at_commit[batch_index]
                ) & previous_commits
                immediate_consistency_count = int(stable.sum().item())
                immediate_consistency_rate = (
                    immediate_consistency_count / immediate_consistency_total
                )
        if config.proposal_strategy == PROPOSED_PROPOSAL_STRATEGY:
            fallback_policy = config.dependency_fallback_strategy
        elif config.proposal_strategy == "baseline_current_mixed":
            fallback_policy = "baseline_random_refill"
        else:
            fallback_policy = "none"
        diagnostics.append(
            {
                "schema_version": 1,
                "proposal_strategy": config.proposal_strategy,
                "verifier_metric": metric,
                "block_index": block_index,
                "step_index": step_index,
                "global_step_index": global_step_index,
                "generation_seed": generation_seed,
                "mask_ratio": (
                    remaining_count / response_count if response_count else 0.0
                ),
                "remaining_response_masks": remaining_count,
                "response_tokens": response_count,
                "candidate_budget_requested": config.candidate_budget,
                "candidate_budget_semantics": (
                    "per_action_size"
                    if config.dependency_cardinality_strategy == "joint_k"
                    else "per_step"
                ),
                "candidate_count_realized": int(valid_indexes.numel()),
                "candidate_action_space_size": action_space_size,
                "candidate_action_set_count": candidate_action_set_count,
                "candidate_collapse": (
                    int(valid_indexes.numel()) != expected_candidate_count
                ),
                "commit_k": (
                    int(lookahead.best_mask[batch_index].sum().item())
                    if best_index >= 0
                    else 0
                ),
                "cardinality_strategy": (
                    config.dependency_cardinality_strategy
                ),
                "maximum_action_size": config.dependency_max_action_size,
                "allowed_action_sizes": list(
                    parse_action_sizes(config.dependency_action_sizes)
                ),
                "size_scoring_rule": selection.size_scoring_rule,
                "utility_threshold": config.dependency_utility_threshold,
                "entropy_budget": config.dependency_entropy_budget,
                "immediate_cost_weight": (
                    config.dependency_immediate_cost_weight
                ),
                "size_penalty": config.dependency_size_penalty,
                "candidates": candidate_records,
                "selected_candidate": selected_record,
                "selected_set_mean_conflict": (
                    selected_record["mean_within_set_conflict"]
                    if selected_record is not None
                    else None
                ),
                "selected_set_max_conflict": (
                    selected_record["max_within_set_conflict"]
                    if selected_record is not None
                    else None
                ),
                "winning_margin": winning_margin,
                "winning_margin_per_heldout": margin_per_heldout,
                "bf16_per_token_drift_reference": precision_reference,
                "margin_to_bf16_reference_ratio": (
                    None
                    if margin_per_heldout is None
                    else margin_per_heldout / precision_reference
                ),
                "sink_count": sink_count,
                "capture_layers": (
                    list(dependency.layer_ids) if dependency is not None else []
                ),
                "dependency_capture_source": (
                    base_forward.dependency_capture_source
                ),
                "capture_batch_size": base_forward.capture_batch_size,
                "captured_base_forward_count": (
                    base_forward.captured_base_forward_count
                ),
                "base_forward_capture_enabled": (
                    config.proposal_strategy == PROPOSED_PROPOSAL_STRATEGY
                ),
                "capture_active_before_lookahead": (
                    base_forward.capture_active_after_forward
                ),
                "capture_tensors_released_before_lookahead": (
                    selection.capture_tensors_released_before_lookahead
                ),
                "lookahead_capture_forward_count": (
                    selection.lookahead_capture_forward_count
                ),
                "lookahead_model_calls": lookahead.model_calls,
                "candidate_chunk_size": lookahead.candidate_chunk_size,
                "fallback_strategy": fallback_policy,
                "selected_fallback_source": (
                    selected_record["fallback_source"]
                    if selected_record is not None
                    else None
                ),
                "dependency_direction": config.dependency_direction,
                "dependency_target_weighting": (
                    config.dependency_target_weighting
                ),
                "dependency_position_temperature": (
                    config.dependency_position_temperature
                ),
                "dependency_confidence_exponent": (
                    config.dependency_confidence_exponent
                ),
                "dependency_parallel_variant": (
                    config.dependency_parallel_variant
                ),
                "dependency_conflict_normalization": (
                    config.dependency_conflict_normalization
                ),
                "dependency_conflict_penalty": config.dependency_conflict_penalty,
                "dependency_hard_conflict_threshold": (
                    config.dependency_hard_conflict_threshold
                ),
                "dependency_anchor_support_weight": (
                    config.dependency_anchor_support_weight
                ),
                "dependency_anchor_confidence_threshold": (
                    config.dependency_anchor_confidence_threshold
                ),
                "committed_anchor_count_before": committed_before,
                "reliable_anchor_count_before": reliable_before,
                "committed_anchor_count_after": committed_after,
                "reliable_anchor_count_after": reliable_after,
                "immediate_token_consistency_count": (
                    immediate_consistency_count
                ),
                "immediate_token_consistency_total": (
                    immediate_consistency_total
                ),
                "immediate_token_consistency_rate": immediate_consistency_rate,
                "current_state_metric_sum": current_metric_sum,
                "current_state_metric_mean": current_metric_mean,
                "zero_dependency_signal": (
                    candidates.configuration.get(
                        "zero_dependency_signal_by_batch", (None,)
                    )[batch_index]
                    if config.proposal_strategy == PROPOSED_PROPOSAL_STRATEGY
                    else None
                ),
                "constant_dependency_signal": (
                    candidates.configuration.get(
                        "constant_dependency_signal_by_batch", (None,)
                    )[batch_index]
                    if config.proposal_strategy == PROPOSED_PROPOSAL_STRATEGY
                    else None
                ),
                "timing_seconds": {
                    "base_forward": base_forward.base_forward_seconds,
                    "attention_reconstruction": (
                        selection.attention_reconstruction_seconds
                    ),
                    "candidate_proposal_generation": (
                        selection.proposal_generation_seconds
                    ),
                    "candidate_lookahead": selection.candidate_lookahead_seconds,
                },
            }
        )
    return diagnostics
