"""Construct and score adaptive-cardinality dependency actions.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_adaptive_cardinality.py -v
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math

import torch

from dllm.core.samplers.batched_lookahead import BatchedLookaheadOutput
from dllm.core.samplers.candidates import CandidateBatch, dependency_anchor_scores
from dllm.core.samplers.parallel_candidates import (
    CommittedAnchorState,
    _seed_order,
    _set_statistics,
    _stable_best,
    _subset_objective,
    _validate_bool_mask,
    _validate_position_values,
    anchor_support_scores,
    build_symmetric_conflict_matrix,
    generate_parallel_dependency_candidates,
    initialize_committed_anchor_state,
)


CARDINALITY_STRATEGIES = (
    "fixed",
    "scheduler",
    "marginal_utility",
    "joint_k",
    "entropy_budget",
)
ADAPTIVE_CARDINALITY_STRATEGIES = (
    "marginal_utility",
    "joint_k",
    "entropy_budget",
)
SIZE_SCORING_RULES = (
    "raw",
    "per_token",
    "immediate_cost",
    "size_penalty",
)
STOPPING_RULES = ("marginal_utility", "entropy_budget")


def parse_action_sizes(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse a CLI-safe ``1|2|4`` string or an integer iterable."""
    if isinstance(value, str):
        pieces = value.split("|")
        if not value or any(not piece.strip() for piece in pieces):
            raise ValueError("action sizes must use a nonempty form such as '1|2|4'.")
        try:
            sizes = tuple(int(piece.strip()) for piece in pieces)
        except ValueError as error:
            raise ValueError("action sizes must contain integers.") from error
    else:
        if isinstance(value, (bytes, bytearray)):
            raise TypeError("action sizes must be a string or integer iterable.")
        sizes = tuple(value)
    if not sizes:
        raise ValueError("at least one action size is required.")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise TypeError("action sizes must contain integers.")
    if any(size <= 0 for size in sizes):
        raise ValueError("action sizes must be positive.")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("action sizes must be unique and strictly increasing.")
    return sizes


def _validate_generation_inputs(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    maximum_action_size: int,
    generation_seed: int,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate common adaptive generator inputs."""
    if not isinstance(dependency, torch.Tensor) or dependency.ndim != 3:
        raise ValueError("dependency must have shape [B,T,T].")
    if not dependency.is_floating_point():
        raise TypeError("dependency must be floating point.")
    batch_size, query_count, key_count = dependency.shape
    if query_count != key_count:
        raise ValueError("dependency must be square.")
    if not torch.isfinite(dependency).all() or torch.any(dependency < 0):
        raise ValueError("dependency must be finite and nonnegative.")
    shape = (batch_size, query_count)
    eligible = _validate_bool_mask(
        eligible_mask,
        name="eligible_mask",
        shape=shape,
        device=dependency.device,
    )
    entropy = _validate_position_values(
        entropy,
        name="entropy",
        shape=shape,
        device=dependency.device,
    ).float()
    confidence = _validate_position_values(
        confidence,
        name="confidence",
        shape=shape,
        device=dependency.device,
        maximum=1.0,
    ).float()
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if (
        isinstance(maximum_action_size, bool)
        or not isinstance(maximum_action_size, int)
        or maximum_action_size <= 0
    ):
        raise ValueError("maximum_action_size must be a positive integer.")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("generation_seed must be a nonnegative integer.")
    return batch_size, query_count, eligible, entropy, confidence


def _construct_stopped_soft_full_subset(
    *,
    seed: int,
    eligible_positions: list[int],
    maximum_action_size: int,
    utility: torch.Tensor,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    conflict: torch.Tensor,
    conflict_penalty: float,
    stopping_rule: str,
    utility_threshold: float,
    entropy_budget: float,
) -> tuple[list[int], list[float], str]:
    """Grow one frozen soft-full ordering and stop without verifier feedback."""
    selected = [seed]
    accepted_marginals = [float(utility[seed])]
    cumulative_entropy = float(entropy[seed])
    stop_reason = "maximum_action_size"
    while len(selected) < maximum_action_size:
        remaining = [
            position
            for position in eligible_positions
            if position not in selected
        ]
        if not remaining:
            stop_reason = "eligible_exhausted"
            break
        marginal = torch.full_like(utility, -torch.inf)
        for position in remaining:
            risk = 1.0 - torch.minimum(
                confidence[position], confidence[selected]
            )
            penalty = (conflict[position, selected] * risk).sum()
            marginal[position] = utility[position] - conflict_penalty * penalty
        chosen = _stable_best(remaining, marginal)
        chosen_marginal = float(marginal[chosen])
        if stopping_rule == "marginal_utility" and chosen_marginal <= utility_threshold:
            stop_reason = "marginal_utility_threshold"
            break
        if (
            stopping_rule == "entropy_budget"
            and cumulative_entropy + float(entropy[chosen]) > entropy_budget
        ):
            stop_reason = "entropy_budget"
            break
        selected.append(chosen)
        accepted_marginals.append(chosen_marginal)
        cumulative_entropy += float(entropy[chosen])
    return selected, accepted_marginals, stop_reason


def generate_stopped_soft_full_candidates(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    maximum_action_size: int,
    stopping_rule: str,
    utility_threshold: float = 0.0,
    entropy_budget: float = 1.0,
    direction: str = "outgoing",
    target_weighting: str = "entropy",
    confidence_exponent: float = 0.0,
    anchor_state: CommittedAnchorState | None = None,
    conflict_normalization: str = "max",
    conflict_penalty: float = 1.0,
    anchor_support_weight: float = 1.0,
    position_temperature: float = 1.0,
    generation_seed: int = 0,
    name_prefix: str = "adaptive_candidate",
) -> CandidateBatch:
    """Generate variable-size candidates using the frozen soft-full ordering."""
    (
        batch_size,
        query_count,
        eligible,
        entropy,
        confidence,
    ) = _validate_generation_inputs(
        dependency,
        entropy,
        confidence,
        eligible_mask,
        candidate_budget=candidate_budget,
        maximum_action_size=maximum_action_size,
        generation_seed=generation_seed,
    )
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"stopping_rule must be one of {STOPPING_RULES}.")
    for name, value, nonnegative in (
        ("utility_threshold", utility_threshold, False),
        ("entropy_budget", entropy_budget, True),
        ("conflict_penalty", conflict_penalty, True),
        ("anchor_support_weight", anchor_support_weight, True),
        ("position_temperature", position_temperature, True),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or (nonnegative and value < 0):
            qualifier = "finite and nonnegative" if nonnegative else "finite"
            raise ValueError(f"{name} must be {qualifier}.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")

    conflict_output = build_symmetric_conflict_matrix(
        dependency,
        eligible,
        normalization=conflict_normalization,
    )
    base_scores = dependency_anchor_scores(
        dependency,
        entropy,
        eligible,
        direction=direction,
        target_weighting=target_weighting,
        confidence=confidence,
        confidence_exponent=confidence_exponent,
    )
    if anchor_state is None:
        anchor_state = initialize_committed_anchor_state(
            eligible,
            confidence_threshold=1.0,
        )
    support = anchor_support_scores(dependency, eligible, anchor_state)
    utility = base_scores + float(anchor_support_weight) * support
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generation_seed)

    selected_by_batch: list[list[dict[str, object]]] = []
    maximum_output_count = 0
    for batch_index in range(batch_size):
        positions = torch.where(eligible[batch_index])[0].tolist()
        if not positions:
            selected_by_batch.append([])
            continue
        target = min(candidate_budget, len(positions))
        seeds = _seed_order(
            utility[batch_index],
            positions,
            position_temperature=float(position_temperature),
            generator=generator,
        )
        records: list[dict[str, object]] = []
        seen: set[tuple[int, ...]] = set()
        for seed_rank, seed in enumerate(seeds):
            selected, marginals, stop_reason = _construct_stopped_soft_full_subset(
                seed=seed,
                eligible_positions=positions,
                maximum_action_size=min(maximum_action_size, len(positions)),
                utility=utility[batch_index],
                entropy=entropy[batch_index],
                confidence=confidence[batch_index],
                conflict=conflict_output.matrix[batch_index],
                conflict_penalty=float(conflict_penalty),
                stopping_rule=stopping_rule,
                utility_threshold=float(utility_threshold),
                entropy_budget=float(entropy_budget),
            )
            canonical = tuple(sorted(selected))
            if canonical in seen:
                continue
            seen.add(canonical)
            mean_conflict, max_conflict = _set_statistics(
                conflict_output.matrix[batch_index], selected
            )
            records.append(
                {
                    "positions": canonical,
                    "construction_order": tuple(selected),
                    "seed": seed,
                    "score": _subset_objective(
                        selected,
                        utility[batch_index],
                        conflict_output.matrix[batch_index],
                        confidence[batch_index],
                        conflict_penalty=float(conflict_penalty),
                    ),
                    "mean_conflict": mean_conflict,
                    "max_conflict": max_conflict,
                    "support_sum": sum(
                        float(support[batch_index, position])
                        for position in selected
                    ),
                    "seed_rank": seed_rank,
                    "accepted_marginals": tuple(marginals),
                    "stop_reason": stop_reason,
                    "source": f"soft_full_{stopping_rule}",
                    "fallback_source": None,
                }
            )
            if len(records) == target:
                break

        if len(records) < target:
            deterministic = sorted(
                positions,
                key=lambda position: (-float(utility[batch_index, position]), position),
            )
            for seed_rank, seed in enumerate(deterministic):
                canonical = (seed,)
                if canonical in seen:
                    continue
                seen.add(canonical)
                records.append(
                    {
                        "positions": canonical,
                        "construction_order": canonical,
                        "seed": seed,
                        "score": float(utility[batch_index, seed]),
                        "mean_conflict": 0.0,
                        "max_conflict": 0.0,
                        "support_sum": float(support[batch_index, seed]),
                        "seed_rank": seed_rank,
                        "accepted_marginals": (float(utility[batch_index, seed]),),
                        "stop_reason": "singleton_diversity_refill",
                        "source": f"soft_full_{stopping_rule}",
                        "fallback_source": "adaptive_singleton_refill",
                    }
                )
                if len(records) == target:
                    break
        if len(records) != target:
            raise RuntimeError("could not fill the adaptive candidate budget.")
        selected_by_batch.append(records)
        maximum_output_count = max(maximum_output_count, len(records))

    requested = torch.full(
        (batch_size,),
        maximum_action_size,
        device=dependency.device,
        dtype=torch.long,
    )
    eligible_counts = eligible.sum(dim=-1, dtype=torch.long)
    clipped = torch.minimum(requested, eligible_counts)
    storage_width = int(clipped.max().item()) if batch_size else 0
    candidate_masks = torch.zeros(
        (maximum_output_count, batch_size, query_count),
        device=dependency.device,
        dtype=torch.bool,
    )
    candidate_valid = torch.zeros(
        (maximum_output_count, batch_size),
        device=dependency.device,
        dtype=torch.bool,
    )
    proposal_scores = torch.full(
        (maximum_output_count, batch_size),
        -torch.inf,
        device=dependency.device,
        dtype=torch.float32,
    )
    seed_anchors = torch.full(
        (maximum_output_count, batch_size),
        -1,
        device=dependency.device,
        dtype=torch.long,
    )
    selected_positions = torch.full(
        (maximum_output_count, batch_size, storage_width),
        -1,
        device=dependency.device,
        dtype=torch.long,
    )
    mean_conflicts = torch.zeros_like(proposal_scores)
    for batch_index, records in enumerate(selected_by_batch):
        for candidate_index, record in enumerate(records):
            positions = torch.tensor(
                record["positions"],
                device=dependency.device,
                dtype=torch.long,
            )
            candidate_masks[candidate_index, batch_index, positions] = True
            candidate_valid[candidate_index, batch_index] = True
            proposal_scores[candidate_index, batch_index] = float(record["score"])
            seed_anchors[candidate_index, batch_index] = int(record["seed"])
            selected_positions[
                candidate_index, batch_index, : positions.numel()
            ] = positions
            mean_conflicts[candidate_index, batch_index] = float(
                record["mean_conflict"]
            )

    metadata: list[Mapping[str, object]] = []
    for candidate_index in range(maximum_output_count):
        records = [
            (
                selected_by_batch[batch_index][candidate_index]
                if candidate_index < len(selected_by_batch[batch_index])
                else None
            )
            for batch_index in range(batch_size)
        ]
        metadata.append(
            {
                "rank": candidate_index,
                "source": "adaptive_soft_full_by_batch",
                "source_by_batch": tuple(
                    record["source"] if record is not None else None
                    for record in records
                ),
                "action_size_by_batch": tuple(
                    len(record["positions"]) if record is not None else None
                    for record in records
                ),
                "construction_order_by_batch": tuple(
                    record["construction_order"] if record is not None else None
                    for record in records
                ),
                "accepted_marginals_by_batch": tuple(
                    record["accepted_marginals"] if record is not None else None
                    for record in records
                ),
                "stopping_reason_by_batch": tuple(
                    record["stop_reason"] if record is not None else None
                    for record in records
                ),
                "max_within_set_conflict_by_batch": tuple(
                    record["max_conflict"] if record is not None else None
                    for record in records
                ),
                "anchor_support_sum_by_batch": tuple(
                    record["support_sum"] if record is not None else None
                    for record in records
                ),
                "hard_fallback_count_by_batch": tuple(
                    0 if record is not None else None for record in records
                ),
                "fallback_source_by_batch": tuple(
                    record["fallback_source"] if record is not None else None
                    for record in records
                ),
                "is_fallback_by_batch": tuple(
                    bool(record["fallback_source"])
                    if record is not None
                    else None
                    for record in records
                ),
                "seed_rank_by_batch": tuple(
                    record["seed_rank"] if record is not None else None
                    for record in records
                ),
            }
        )

    zero_signal = tuple(
        bool(torch.all(base_scores[row, eligible[row]] == 0))
        if torch.any(eligible[row])
        else True
        for row in range(batch_size)
    )
    constant_signal = tuple(
        bool(
            torch.all(
                base_scores[row, eligible[row]]
                == base_scores[row, eligible[row]][0]
            )
        )
        if torch.any(eligible[row])
        else True
        for row in range(batch_size)
    )
    expected_counts = tuple(len(records) for records in selected_by_batch)
    action_set_counts = tuple(
        sum(
            math.comb(count, size)
            for size in range(1, min(maximum_action_size, count) + 1)
        )
        for count in eligible_counts.tolist()
    )
    return CandidateBatch(
        candidate_masks=candidate_masks,
        names=tuple(
            f"{name_prefix}_{candidate_index}"
            for candidate_index in range(maximum_output_count)
        ),
        proposal_scores=proposal_scores,
        seed_anchors=seed_anchors,
        selected_positions=selected_positions,
        mean_within_set_dependency=mean_conflicts,
        candidate_valid=candidate_valid,
        eligible_mask=eligible,
        requested_k=requested,
        clipped_k=clipped,
        metadata=tuple(metadata),
        generation_seed=generation_seed,
        configuration={
            "proposal": "adaptive_soft_full_pool",
            "cardinality_strategy": stopping_rule,
            "candidate_budget": candidate_budget,
            "maximum_action_size": maximum_action_size,
            "utility_threshold": float(utility_threshold),
            "entropy_budget": float(entropy_budget),
            "direction": direction,
            "target_weighting": target_weighting,
            "confidence_exponent": float(confidence_exponent),
            "conflict_normalization": conflict_normalization,
            "conflict_scale_by_batch": tuple(
                float(value) for value in conflict_output.scale_by_batch.tolist()
            ),
            "conflict_penalty": float(conflict_penalty),
            "interaction_risk": "1 - min(confidence_i, confidence_r)",
            "anchor_support_weight": float(anchor_support_weight),
            "reliable_anchor_count_by_batch": tuple(
                int(value)
                for value in anchor_state.reliable_anchor_mask.sum(dim=-1).tolist()
            ),
            "zero_dependency_signal_by_batch": zero_signal,
            "constant_dependency_signal_by_batch": constant_signal,
            "position_temperature": float(position_temperature),
            "tie_breaking": "ascending compact position",
            "candidate_refill": "adaptive_singleton_diversity",
            "expected_candidate_count_by_batch": expected_counts,
            "candidate_action_set_count_by_batch": action_set_counts,
        },
        action_sizes=candidate_masks.sum(dim=-1, dtype=torch.long),
    )


def generate_joint_k_dependency_candidates(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    action_sizes: str | Iterable[int],
    candidate_budget_per_size: int,
    direction: str = "outgoing",
    target_weighting: str = "entropy",
    confidence_exponent: float = 0.0,
    anchor_state: CommittedAnchorState | None = None,
    conflict_normalization: str = "max",
    conflict_penalty: float = 1.0,
    hard_conflict_threshold: float = 0.25,
    anchor_support_weight: float = 1.0,
    position_temperature: float = 1.0,
    generation_seed: int = 0,
    name_prefix: str = "joint_candidate",
) -> CandidateBatch:
    """Combine equal-budget soft-full pools over explicit action sizes."""
    sizes = parse_action_sizes(action_sizes)
    batch_size, sequence_length, eligible, entropy, confidence = (
        _validate_generation_inputs(
            dependency,
            entropy,
            confidence,
            eligible_mask,
            candidate_budget=candidate_budget_per_size,
            maximum_action_size=max(sizes),
            generation_seed=generation_seed,
        )
    )
    pools = [
        (
            size,
            generate_parallel_dependency_candidates(
                dependency,
                entropy,
                confidence,
                eligible,
                requested_k=size,
                candidate_budget=candidate_budget_per_size,
                variant="soft_full",
                direction=direction,
                target_weighting=target_weighting,
                confidence_exponent=confidence_exponent,
                anchor_state=anchor_state,
                conflict_normalization=conflict_normalization,
                conflict_penalty=conflict_penalty,
                hard_conflict_threshold=hard_conflict_threshold,
                anchor_support_weight=anchor_support_weight,
                position_temperature=position_temperature,
                generation_seed=generation_seed + size,
                name_prefix=f"{name_prefix}_k{size}",
            ),
        )
        for size in sizes
    ]
    candidate_count = sum(pool.shape[0] for _, pool in pools)
    requested = torch.full(
        (batch_size,),
        max(sizes),
        device=dependency.device,
        dtype=torch.long,
    )
    clipped = torch.minimum(
        requested,
        eligible.sum(dim=-1, dtype=torch.long),
    )
    storage_width = int(clipped.max().item()) if batch_size else 0
    masks = torch.zeros(
        (candidate_count, batch_size, sequence_length),
        device=dependency.device,
        dtype=torch.bool,
    )
    valid = torch.zeros(
        (candidate_count, batch_size),
        device=dependency.device,
        dtype=torch.bool,
    )
    scores = torch.full(
        (candidate_count, batch_size),
        -torch.inf,
        device=dependency.device,
        dtype=torch.float32,
    )
    anchors = torch.full(
        (candidate_count, batch_size),
        -1,
        device=dependency.device,
        dtype=torch.long,
    )
    positions = torch.full(
        (candidate_count, batch_size, storage_width),
        -1,
        device=dependency.device,
        dtype=torch.long,
    )
    mean_conflicts = torch.zeros_like(scores)
    names: list[str] = []
    metadata: list[dict[str, object]] = []
    offset = 0
    for requested_size, pool in pools:
        count = pool.shape[0]
        target = slice(offset, offset + count)
        masks[target] = pool.candidate_masks
        valid[target] = pool.candidate_valid
        scores[target] = pool.proposal_scores
        anchors[target] = pool.seed_anchors
        mean_conflicts[target] = pool.mean_within_set_dependency
        for local_index in range(count):
            for batch_index in range(batch_size):
                kept = pool.selected_positions[local_index, batch_index]
                kept = kept[kept >= 0]
                positions[
                    offset + local_index,
                    batch_index,
                    : kept.numel(),
                ] = kept
            names.append(f"{name_prefix}_k{requested_size}_{local_index}")
            metadata.append(
                {
                    **dict(pool.metadata[local_index]),
                    "cardinality_source": "joint_k",
                    "requested_action_size": requested_size,
                    "requested_action_size_by_batch": tuple(
                        requested_size
                        if bool(pool.candidate_valid[local_index, row])
                        else None
                        for row in range(batch_size)
                    ),
                }
            )
        offset += count

    deduplicated = [[False] * batch_size for _ in range(candidate_count)]
    for batch_index in range(batch_size):
        seen: set[tuple[int, ...]] = set()
        for candidate_index in range(candidate_count):
            if not bool(valid[candidate_index, batch_index]):
                continue
            key = tuple(torch.where(masks[candidate_index, batch_index])[0].tolist())
            if key not in seen:
                seen.add(key)
                continue
            deduplicated[candidate_index][batch_index] = True
            masks[candidate_index, batch_index] = False
            valid[candidate_index, batch_index] = False
            scores[candidate_index, batch_index] = -torch.inf
            anchors[candidate_index, batch_index] = -1
            positions[candidate_index, batch_index] = -1
            mean_conflicts[candidate_index, batch_index] = 0.0
    for candidate_index, row in enumerate(metadata):
        row["deduplicated_by_batch"] = tuple(deduplicated[candidate_index])

    realized_counts = tuple(int(value) for value in valid.sum(dim=0).tolist())
    eligible_counts = eligible.sum(dim=-1, dtype=torch.long)
    action_set_counts = []
    for count in eligible_counts.tolist():
        realized_sizes = sorted({min(size, count) for size in sizes if count > 0})
        action_set_counts.append(
            sum(math.comb(count, size) for size in realized_sizes)
        )
    first_configuration = pools[0][1].configuration if pools else {}
    return CandidateBatch(
        candidate_masks=masks,
        names=tuple(names),
        proposal_scores=scores,
        seed_anchors=anchors,
        selected_positions=positions,
        mean_within_set_dependency=mean_conflicts,
        candidate_valid=valid,
        eligible_mask=eligible,
        requested_k=requested,
        clipped_k=clipped,
        metadata=tuple(metadata),
        generation_seed=generation_seed,
        configuration={
            **dict(first_configuration),
            "proposal": "joint_k_soft_full_pool",
            "cardinality_strategy": "joint_k",
            "action_sizes": sizes,
            "candidate_budget_per_size": candidate_budget_per_size,
            "maximum_candidate_count": candidate_count,
            "expected_candidate_count_by_batch": realized_counts,
            "candidate_action_set_count_by_batch": tuple(action_set_counts),
            "deduplication": "exact action mask including realized size",
        },
        action_sizes=masks.sum(dim=-1, dtype=torch.long),
    )


@dataclass(frozen=True)
class SizeAwareScoringOutput:
    """Raw and adjusted lookahead scores for variable-size actions."""

    lookahead: BatchedLookaheadOutput
    raw_scores: torch.Tensor
    immediate_costs: torch.Tensor
    action_sizes: torch.Tensor
    rule: str


def apply_size_aware_scoring(
    lookahead: BatchedLookaheadOutput,
    candidates: CandidateBatch,
    base_metric_map: torch.Tensor,
    *,
    rule: str,
    immediate_cost_weight: float = 1.0,
    size_penalty: float = 0.0,
) -> SizeAwareScoringOutput:
    """Reselect lookahead actions after applying an explicit size correction."""
    if rule not in SIZE_SCORING_RULES:
        raise ValueError(f"rule must be one of {SIZE_SCORING_RULES}.")
    for name, value in (
        ("immediate_cost_weight", immediate_cost_weight),
        ("size_penalty", size_penalty),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if tuple(base_metric_map.shape) != tuple(candidates.eligible_mask.shape):
        raise ValueError("base_metric_map must match the candidate position space.")
    if base_metric_map.device != candidates.candidate_masks.device:
        raise ValueError("base_metric_map must share the candidate device.")
    if tuple(lookahead.scores.shape) != tuple(candidates.candidate_valid.shape):
        raise ValueError("lookahead scores must match candidate validity.")

    raw_scores = lookahead.scores
    action_sizes = candidates.candidate_masks.sum(dim=-1, dtype=torch.long)
    immediate_costs = torch.where(
        candidates.candidate_masks,
        base_metric_map.float().unsqueeze(0),
        torch.zeros_like(candidates.candidate_masks, dtype=torch.float32),
    ).sum(dim=-1)
    if rule == "raw":
        adjusted = raw_scores.clone()
    elif rule == "per_token":
        adjusted = raw_scores / action_sizes.clamp_min(1).float()
    elif rule == "immediate_cost":
        adjusted = raw_scores - float(immediate_cost_weight) * immediate_costs
    else:
        adjusted = raw_scores - float(size_penalty) * action_sizes.float()
    adjusted = torch.where(
        candidates.candidate_valid,
        adjusted,
        torch.full_like(adjusted, -torch.inf),
    )

    batch_size, sequence_length = candidates.eligible_mask.shape
    best_score = torch.full(
        (batch_size,),
        -torch.inf,
        device=adjusted.device,
        dtype=torch.float32,
    )
    best_index = torch.full(
        (batch_size,),
        -1,
        device=adjusted.device,
        dtype=torch.long,
    )
    for candidate_index in range(adjusted.shape[0]):
        improved = (
            candidates.candidate_valid[candidate_index]
            & (adjusted[candidate_index] > best_score)
        )
        best_score = torch.where(improved, adjusted[candidate_index], best_score)
        best_index = torch.where(
            improved,
            torch.full_like(best_index, candidate_index),
            best_index,
        )
    best_mask = torch.zeros(
        (batch_size, sequence_length),
        device=adjusted.device,
        dtype=torch.bool,
    )
    best_names: list[str | None] = []
    best_metadata: list[Mapping[str, object] | None] = []
    for batch_index, candidate_index in enumerate(best_index.tolist()):
        if candidate_index < 0:
            best_names.append(None)
            best_metadata.append(None)
            continue
        best_mask[batch_index] = candidates.candidate_masks[
            candidate_index, batch_index
        ]
        best_names.append(candidates.names[candidate_index])
        best_metadata.append(candidates.metadata[candidate_index])
    rescored = replace(
        lookahead,
        scores=adjusted,
        best_index=best_index,
        best_score=best_score,
        best_mask=best_mask,
        best_names=tuple(best_names),
        best_metadata=tuple(best_metadata),
    )
    return SizeAwareScoringOutput(
        lookahead=rescored,
        raw_scores=raw_scores,
        immediate_costs=immediate_costs,
        action_sizes=action_sizes,
        rule=rule,
    )
