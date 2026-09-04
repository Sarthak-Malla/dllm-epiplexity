"""Construct fixed-k dependency candidates with conflicts and committed anchors.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_parallel_candidates.py -v
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math

import torch

from dllm.core.samplers.candidates import CandidateBatch, dependency_anchor_scores


CONFLICT_NORMALIZATIONS = ("none", "max", "mean_positive")
PARALLEL_VARIANTS = (
    "correlated_together",
    "top_confidence",
    "hard_low_conflict",
    "anchor_support_only",
    "soft_no_anchor",
    "soft_full",
)


def _validate_bool_mask(
    mask: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Validate one [B,T] boolean mask without changing its position space."""
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean.")
    if mask.device != device:
        raise ValueError(f"{name} must share the dependency device.")
    return mask


def _validate_position_values(
    values: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
    maximum: float | None = None,
) -> torch.Tensor:
    """Validate finite nonnegative values aligned to dependency positions."""
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if values.device != device:
        raise ValueError(f"{name} must share the dependency device.")
    invalid = ~torch.isfinite(values) | (values < 0)
    if maximum is not None:
        invalid |= values > maximum
    if torch.any(invalid):
        interval = f"[0, {maximum}]" if maximum is not None else "[0, infinity)"
        raise ValueError(f"{name} must be finite and lie in {interval}.")
    return values


@dataclass(frozen=True)
class SymmetricConflictOutput:
    """Symmetric masked-masked conflicts and their per-row normalization."""

    matrix: torch.Tensor
    eligible_mask: torch.Tensor
    normalization: str
    scale_by_batch: torch.Tensor

    def __post_init__(self) -> None:
        """Enforce symmetry, zero diagonal, and masked-only support."""
        if not isinstance(self.matrix, torch.Tensor) or self.matrix.ndim != 3:
            raise ValueError("matrix must have shape [B, T, T].")
        if not self.matrix.is_floating_point():
            raise TypeError("matrix must be floating point.")
        batch_size, query_count, key_count = self.matrix.shape
        if query_count != key_count:
            raise ValueError("matrix must be square.")
        shape = (batch_size, query_count)
        _validate_bool_mask(
            self.eligible_mask,
            name="eligible_mask",
            shape=shape,
            device=self.matrix.device,
        )
        if self.normalization not in CONFLICT_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {CONFLICT_NORMALIZATIONS}."
            )
        if (
            not isinstance(self.scale_by_batch, torch.Tensor)
            or tuple(self.scale_by_batch.shape) != (batch_size,)
        ):
            raise ValueError("scale_by_batch must have shape [B].")
        if self.scale_by_batch.device != self.matrix.device:
            raise ValueError("scale_by_batch must share the matrix device.")
        if not self.scale_by_batch.is_floating_point():
            raise TypeError("scale_by_batch must be floating point.")
        if not torch.isfinite(self.matrix).all() or torch.any(self.matrix < 0):
            raise ValueError("conflicts must be finite and nonnegative.")
        if not torch.isfinite(self.scale_by_batch).all() or torch.any(
            self.scale_by_batch <= 0
        ):
            raise ValueError("normalization scales must be finite and positive.")
        torch.testing.assert_close(
            self.matrix,
            self.matrix.transpose(-1, -2),
            rtol=1e-6,
            atol=1e-7,
        )
        diagonal = torch.diagonal(self.matrix, dim1=-2, dim2=-1)
        if torch.any(diagonal != 0):
            raise ValueError("conflict diagonal must be zero.")
        pair_mask = self.eligible_mask.unsqueeze(-1) & self.eligible_mask.unsqueeze(-2)
        if torch.any(self.matrix.masked_select(~pair_mask) != 0):
            raise ValueError("conflicts must be restricted to eligible pairs.")


@dataclass(frozen=True)
class CommittedAnchorState:
    """Per-row history containing only positions executed by the decoder."""

    committed_position_mask: torch.Tensor
    confidence_at_commit: torch.Tensor
    commit_step: torch.Tensor
    reliable_anchor_mask: torch.Tensor
    token_id_at_commit: torch.Tensor
    confidence_threshold: float

    def __post_init__(self) -> None:
        """Reject state that could confuse masked predictions with anchors."""
        committed = self.committed_position_mask
        if not isinstance(committed, torch.Tensor) or committed.ndim != 2:
            raise ValueError("committed_position_mask must have shape [B,T].")
        if committed.dtype != torch.bool:
            raise TypeError("committed_position_mask must be boolean.")
        shape = tuple(committed.shape)
        device = committed.device
        _validate_bool_mask(
            self.reliable_anchor_mask,
            name="reliable_anchor_mask",
            shape=shape,
            device=device,
        )
        if torch.any(self.reliable_anchor_mask & ~committed):
            raise ValueError("reliable anchors must already be committed.")
        confidence = _validate_position_values(
            self.confidence_at_commit,
            name="confidence_at_commit",
            shape=shape,
            device=device,
            maximum=1.0,
        )
        for name, tensor in (
            ("commit_step", self.commit_step),
            ("token_id_at_commit", self.token_id_at_commit),
        ):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long.")
            if tensor.device != device:
                raise ValueError(f"{name} must share the state device.")
        if not math.isfinite(float(self.confidence_threshold)) or not (
            0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be finite and in [0,1].")
        if torch.any(confidence.masked_select(~committed) != 0):
            raise ValueError("uncommitted positions must have zero commit confidence.")
        if torch.any(self.commit_step.masked_select(~committed) != -1):
            raise ValueError("uncommitted positions must have commit_step=-1.")
        if torch.any(self.token_id_at_commit.masked_select(~committed) != -1):
            raise ValueError("uncommitted positions must have token_id_at_commit=-1.")
        expected_reliable = committed & (confidence >= self.confidence_threshold)
        if not torch.equal(self.reliable_anchor_mask, expected_reliable):
            raise ValueError("reliable_anchor_mask does not match the threshold.")


def initialize_committed_anchor_state(
    reference: torch.Tensor,
    *,
    confidence_threshold: float,
) -> CommittedAnchorState:
    """Create an empty anchor history aligned to a [B,T] reference tensor."""
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("reference must have shape [B,T].")
    if not math.isfinite(float(confidence_threshold)) or not (
        0 <= confidence_threshold <= 1
    ):
        raise ValueError("confidence_threshold must be finite and in [0,1].")
    shape = tuple(reference.shape)
    device = reference.device
    return CommittedAnchorState(
        committed_position_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        confidence_at_commit=torch.zeros(shape, device=device, dtype=torch.float32),
        commit_step=torch.full(shape, -1, device=device, dtype=torch.long),
        reliable_anchor_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        token_id_at_commit=torch.full(shape, -1, device=device, dtype=torch.long),
        confidence_threshold=float(confidence_threshold),
    )


def record_committed_anchors(
    state: CommittedAnchorState,
    executed_mask: torch.Tensor,
    confidence: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    commit_step: int,
) -> CommittedAnchorState:
    """Return state extended only by the action the decoder actually executed."""
    if not isinstance(state, CommittedAnchorState):
        raise TypeError("state must be a CommittedAnchorState.")
    shape = tuple(state.committed_position_mask.shape)
    executed = _validate_bool_mask(
        executed_mask,
        name="executed_mask",
        shape=shape,
        device=state.committed_position_mask.device,
    )
    confidence = _validate_position_values(
        confidence,
        name="confidence",
        shape=shape,
        device=state.committed_position_mask.device,
        maximum=1.0,
    ).float()
    if not isinstance(token_ids, torch.Tensor) or tuple(token_ids.shape) != shape:
        raise ValueError(f"token_ids must have shape {shape}.")
    if token_ids.dtype != torch.long:
        raise TypeError("token_ids must use torch.long.")
    if token_ids.device != state.committed_position_mask.device:
        raise ValueError("token_ids must share the state device.")
    if isinstance(commit_step, bool) or not isinstance(commit_step, int):
        raise TypeError("commit_step must be an integer.")
    if commit_step < 0:
        raise ValueError("commit_step must be nonnegative.")
    if torch.any(executed & state.committed_position_mask):
        raise ValueError("an executed action cannot recommit an existing anchor.")

    committed = state.committed_position_mask | executed
    confidence_at_commit = torch.where(
        executed,
        confidence,
        state.confidence_at_commit,
    )
    commit_steps = torch.where(
        executed,
        torch.full_like(state.commit_step, commit_step),
        state.commit_step,
    )
    committed_tokens = torch.where(
        executed,
        token_ids,
        state.token_id_at_commit,
    )
    reliable = committed & (
        confidence_at_commit >= state.confidence_threshold
    )
    return CommittedAnchorState(
        committed_position_mask=committed,
        confidence_at_commit=confidence_at_commit,
        commit_step=commit_steps,
        reliable_anchor_mask=reliable,
        token_id_at_commit=committed_tokens,
        confidence_threshold=state.confidence_threshold,
    )


def gather_committed_anchor_state(
    state: CommittedAnchorState,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> CommittedAnchorState:
    """Gather an absolute anchor history into a padded compact position space."""
    if not isinstance(state, CommittedAnchorState):
        raise TypeError("state must be a CommittedAnchorState.")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2:
        raise ValueError("positions must have shape [B,M].")
    shape = tuple(positions.shape)
    valid = _validate_bool_mask(
        valid_mask,
        name="valid_mask",
        shape=shape,
        device=positions.device,
    )
    if positions.dtype != torch.long:
        raise TypeError("positions must use torch.long.")
    if positions.shape[0] != state.committed_position_mask.shape[0]:
        raise ValueError("positions and anchor state must share batch size.")
    if positions.device != state.committed_position_mask.device:
        raise ValueError("positions and anchor state must share a device.")
    safe = positions.clamp_min(0)

    def gather(values: torch.Tensor, fill: int | float | bool) -> torch.Tensor:
        compact = torch.gather(values, dim=1, index=safe)
        return torch.where(valid, compact, torch.full_like(compact, fill))

    return CommittedAnchorState(
        committed_position_mask=gather(state.committed_position_mask, False),
        confidence_at_commit=gather(state.confidence_at_commit, 0.0),
        commit_step=gather(state.commit_step, -1),
        reliable_anchor_mask=gather(state.reliable_anchor_mask, False),
        token_id_at_commit=gather(state.token_id_at_commit, -1),
        confidence_threshold=state.confidence_threshold,
    )


def build_symmetric_conflict_matrix(
    dependency: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    normalization: str = "max",
) -> SymmetricConflictOutput:
    """Build ``0.5 * (D + D^T)`` over masked-masked positions only."""
    if not isinstance(dependency, torch.Tensor) or dependency.ndim != 3:
        raise ValueError("dependency must have shape [B,T,T].")
    if not dependency.is_floating_point():
        raise TypeError("dependency must be floating point.")
    batch_size, query_count, key_count = dependency.shape
    if query_count != key_count:
        raise ValueError("dependency must be square.")
    if not torch.isfinite(dependency).all() or torch.any(dependency < 0):
        raise ValueError("dependency must be finite and nonnegative.")
    eligible = _validate_bool_mask(
        eligible_mask,
        name="eligible_mask",
        shape=(batch_size, query_count),
        device=dependency.device,
    )
    if normalization not in CONFLICT_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {CONFLICT_NORMALIZATIONS}."
        )
    pair_mask = eligible.unsqueeze(-1) & eligible.unsqueeze(-2)
    diagonal = torch.eye(
        query_count,
        device=dependency.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    symmetric = 0.5 * (dependency.float() + dependency.float().transpose(-1, -2))
    symmetric = torch.where(pair_mask & ~diagonal, symmetric, torch.zeros_like(symmetric))
    scales = torch.ones(batch_size, device=dependency.device, dtype=torch.float32)
    for batch_index in range(batch_size):
        values = symmetric[batch_index].masked_select(
            pair_mask[batch_index] & ~diagonal[0]
        )
        if normalization == "max" and values.numel() and torch.any(values > 0):
            scales[batch_index] = values.max()
        elif normalization == "mean_positive":
            positive = values[values > 0]
            if positive.numel():
                scales[batch_index] = positive.mean()
    symmetric = symmetric / scales[:, None, None]
    return SymmetricConflictOutput(
        matrix=symmetric,
        eligible_mask=eligible,
        normalization=normalization,
        scale_by_batch=scales,
    )


def anchor_support_scores(
    dependency: torch.Tensor,
    eligible_mask: torch.Tensor,
    anchor_state: CommittedAnchorState,
) -> torch.Tensor:
    """Compute target-query to committed-anchor-key support ``D[i,u]``."""
    if not isinstance(dependency, torch.Tensor) or dependency.ndim != 3:
        raise ValueError("dependency must have shape [B,T,T].")
    batch_size, query_count, key_count = dependency.shape
    if query_count != key_count:
        raise ValueError("dependency must be square.")
    if not dependency.is_floating_point():
        raise TypeError("dependency must be floating point.")
    if not torch.isfinite(dependency).all() or torch.any(dependency < 0):
        raise ValueError("dependency must be finite and nonnegative.")
    shape = (batch_size, query_count)
    eligible = _validate_bool_mask(
        eligible_mask,
        name="eligible_mask",
        shape=shape,
        device=dependency.device,
    )
    if not isinstance(anchor_state, CommittedAnchorState):
        raise TypeError("anchor_state must be a CommittedAnchorState.")
    if tuple(anchor_state.committed_position_mask.shape) != shape:
        raise ValueError("anchor_state must match dependency position space.")
    if anchor_state.committed_position_mask.device != dependency.device:
        raise ValueError("anchor_state must share the dependency device.")
    weights = torch.where(
        anchor_state.reliable_anchor_mask,
        anchor_state.confidence_at_commit,
        torch.zeros_like(anchor_state.confidence_at_commit),
    ).float()
    support = torch.bmm(dependency.float(), weights.unsqueeze(-1)).squeeze(-1)
    return torch.where(eligible, support, torch.zeros_like(support))


def within_action_conflict_statistics(
    conflict: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and maximum unordered-pair conflict for [N,B,T] actions."""
    if not isinstance(conflict, torch.Tensor) or conflict.ndim != 3:
        raise ValueError("conflict must have shape [B,T,T].")
    if not isinstance(action_mask, torch.Tensor) or action_mask.ndim != 3:
        raise ValueError("action_mask must have shape [N,B,T].")
    candidate_count, batch_size, sequence_length = action_mask.shape
    if tuple(conflict.shape) != (batch_size, sequence_length, sequence_length):
        raise ValueError("conflict and action_mask shapes are not aligned.")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must be boolean.")
    if action_mask.device != conflict.device:
        raise ValueError("conflict and action_mask must share a device.")
    means = torch.zeros(
        (candidate_count, batch_size),
        device=conflict.device,
        dtype=torch.float32,
    )
    maxima = torch.zeros_like(means)
    for candidate_index in range(candidate_count):
        for batch_index in range(batch_size):
            positions = torch.where(action_mask[candidate_index, batch_index])[0]
            if positions.numel() < 2:
                continue
            pairs = torch.combinations(positions, r=2)
            values = conflict[batch_index, pairs[:, 0], pairs[:, 1]].float()
            means[candidate_index, batch_index] = values.mean()
            maxima[candidate_index, batch_index] = values.max()
    return means, maxima


def _normalize_requested_k(
    requested_k: int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a scalar or [B] fixed-cardinality request."""
    if isinstance(requested_k, bool):
        raise TypeError("requested_k must be an integer or integer tensor.")
    if isinstance(requested_k, int):
        values = torch.full(
            (batch_size,), requested_k, device=device, dtype=torch.long
        )
    elif isinstance(requested_k, torch.Tensor):
        if requested_k.dtype == torch.bool or requested_k.is_floating_point():
            raise TypeError("requested_k tensor must use an integer dtype.")
        values = requested_k.to(device=device, dtype=torch.long).flatten()
        if values.numel() == 1 and batch_size > 1:
            values = values.expand(batch_size)
        if values.numel() != batch_size:
            raise ValueError(f"requested_k must be scalar or length {batch_size}.")
    else:
        raise TypeError("requested_k must be an integer or integer tensor.")
    if torch.any(values < 0):
        raise ValueError("requested_k must be nonnegative.")
    return values


def _stable_best(positions: list[int], values: torch.Tensor) -> int:
    """Choose the greatest value with ascending-position tie breaking."""
    return max(positions, key=lambda position: (float(values[position]), -position))


def _set_statistics(
    conflict: torch.Tensor,
    selected: list[int],
) -> tuple[float, float]:
    """Compute unordered-pair conflict statistics for one position list."""
    if len(selected) < 2:
        return 0.0, 0.0
    values = [
        float(conflict[left, right])
        for left, right in combinations(selected, 2)
    ]
    return sum(values) / len(values), max(values)


def _subset_objective(
    selected: list[int],
    utility: torch.Tensor,
    conflict: torch.Tensor,
    confidence: torch.Tensor,
    *,
    conflict_penalty: float,
) -> float:
    """Evaluate the exact soft objective for a completed unordered set."""
    score = sum(float(utility[position]) for position in selected)
    for left, right in combinations(selected, 2):
        risk = 1.0 - min(float(confidence[left]), float(confidence[right]))
        score -= conflict_penalty * float(conflict[left, right]) * risk
    return score


def _construct_subset(
    *,
    seed: int,
    eligible_positions: list[int],
    requested_k: int,
    variant: str,
    base_utility: torch.Tensor,
    support: torch.Tensor,
    confidence: torch.Tensor,
    conflict: torch.Tensor,
    conflict_penalty: float,
    anchor_support_weight: float,
    hard_conflict_threshold: float,
) -> tuple[list[int], float, int]:
    """Construct one fixed-k set from an explicit seed anchor."""
    if variant == "top_confidence":
        utility = confidence
    elif variant in {"anchor_support_only", "soft_full"}:
        utility = base_utility + anchor_support_weight * support
    else:
        utility = base_utility
    selected = [seed]
    hard_fallback_count = 0
    while len(selected) < requested_k:
        remaining = [p for p in eligible_positions if p not in selected]
        if not remaining:
            break
        if variant == "correlated_together":
            values = torch.full_like(base_utility, -torch.inf)
            for position in remaining:
                values[position] = conflict[position, selected].sum()
            chosen = _stable_best(remaining, values)
        elif variant == "hard_low_conflict":
            independent = [
                position
                for position in remaining
                if bool(
                    torch.all(
                        conflict[position, selected] <= hard_conflict_threshold
                    )
                )
            ]
            if independent:
                chosen = _stable_best(independent, utility)
            else:
                hard_fallback_count += 1
                chosen = min(
                    remaining,
                    key=lambda position: (
                        float(conflict[position, selected].max()),
                        float(conflict[position, selected].sum()),
                        -float(utility[position]),
                        position,
                    ),
                )
        elif variant in {"soft_no_anchor", "soft_full"}:
            marginal = torch.full_like(base_utility, -torch.inf)
            for position in remaining:
                risk = 1.0 - torch.minimum(
                    confidence[position], confidence[selected]
                )
                penalty = (conflict[position, selected] * risk).sum()
                marginal[position] = utility[position] - conflict_penalty * penalty
            chosen = _stable_best(remaining, marginal)
        else:
            chosen = _stable_best(remaining, utility)
        selected.append(chosen)

    if len(selected) != requested_k:
        raise RuntimeError("parallel subset construction did not reach fixed k.")
    objective_penalty = (
        conflict_penalty
        if variant in {"soft_no_anchor", "soft_full"}
        else 0.0
    )
    score = _subset_objective(
        selected,
        utility,
        conflict,
        confidence,
        conflict_penalty=objective_penalty,
    )
    return selected, score, hard_fallback_count


def _seed_order(
    scores: torch.Tensor,
    eligible_positions: list[int],
    *,
    position_temperature: float,
    generator: torch.Generator,
) -> list[int]:
    """Return deterministic-best, Gumbel-diverse, then deterministic seeds."""
    deterministic = sorted(
        eligible_positions,
        key=lambda position: (-float(scores[position]), position),
    )
    if position_temperature == 0:
        return deterministic
    uniform = torch.rand(
        len(eligible_positions),
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    info = torch.finfo(uniform.dtype)
    uniform = uniform.clamp(min=info.tiny, max=1.0 - info.eps)
    noise = -torch.log(-torch.log(uniform))
    perturbed = {
        position: float(scores[position]) / position_temperature + float(noise[index])
        for index, position in enumerate(eligible_positions)
    }
    diverse = sorted(
        eligible_positions,
        key=lambda position: (-perturbed[position], position),
    )
    ordered = [deterministic[0], *diverse, *deterministic]
    return list(dict.fromkeys(ordered))


def generate_parallel_dependency_candidates(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    requested_k: int | torch.Tensor,
    candidate_budget: int,
    variant: str = "soft_full",
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
    name_prefix: str = "parallel_candidate",
) -> CandidateBatch:
    """Generate deterministic and seed-diverse fixed-k dependency actions."""
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
    if variant not in PARALLEL_VARIANTS:
        raise ValueError(f"variant must be one of {PARALLEL_VARIANTS}.")
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    for name, value in (
        ("conflict_penalty", conflict_penalty),
        ("hard_conflict_threshold", hard_conflict_threshold),
        ("anchor_support_weight", anchor_support_weight),
        ("position_temperature", position_temperature),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("generation_seed must be a nonnegative integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")

    requested = _normalize_requested_k(
        requested_k,
        batch_size=batch_size,
        device=dependency.device,
    )
    eligible_counts = eligible.sum(dim=-1, dtype=torch.long)
    clipped = torch.minimum(requested, eligible_counts)
    storage_width = int(clipped.max().item()) if batch_size else 0
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
    if variant in {"anchor_support_only", "soft_full"}:
        seed_scores = base_scores + anchor_support_weight * support
    elif variant == "top_confidence":
        seed_scores = torch.where(
            eligible,
            confidence,
            torch.full_like(confidence, -torch.inf),
        )
    else:
        seed_scores = base_scores

    generator = torch.Generator(device="cpu")
    generator.manual_seed(generation_seed)
    selected_by_batch: list[list[dict[str, object]]] = []
    maximum_output_count = 0
    for batch_index in range(batch_size):
        positions = torch.where(eligible[batch_index])[0].tolist()
        k = int(clipped[batch_index].item())
        if k == 0:
            selected_by_batch.append([])
            continue
        target = min(candidate_budget, math.comb(len(positions), k))
        seeds = _seed_order(
            seed_scores[batch_index],
            positions,
            position_temperature=float(position_temperature),
            generator=generator,
        )
        records: list[dict[str, object]] = []
        seen: set[tuple[int, ...]] = set()
        for seed_rank, seed in enumerate(seeds):
            selected, score, hard_fallback_count = _construct_subset(
                seed=seed,
                eligible_positions=positions,
                requested_k=k,
                variant=variant,
                base_utility=base_scores[batch_index],
                support=support[batch_index],
                confidence=confidence[batch_index],
                conflict=conflict_output.matrix[batch_index],
                conflict_penalty=float(conflict_penalty),
                anchor_support_weight=float(anchor_support_weight),
                hard_conflict_threshold=float(hard_conflict_threshold),
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
                    "seed": seed,
                    "score": score,
                    "mean_conflict": mean_conflict,
                    "max_conflict": max_conflict,
                    "support_sum": sum(float(support[batch_index, p]) for p in selected),
                    "hard_fallback_count": hard_fallback_count,
                    "seed_rank": seed_rank,
                    "source": f"parallel_{variant}",
                    "fallback_source": (
                        "hard_lowest_conflict"
                        if hard_fallback_count
                        else None
                    ),
                }
            )
            if len(records) == target:
                break

        if len(records) < target:
            ranked_positions = sorted(
                positions,
                key=lambda position: (-float(seed_scores[batch_index, position]), position),
            )
            for ranked_subset in combinations(ranked_positions, k):
                canonical = tuple(sorted(ranked_subset))
                if canonical in seen:
                    continue
                seen.add(canonical)
                mean_conflict, max_conflict = _set_statistics(
                    conflict_output.matrix[batch_index], list(ranked_subset)
                )
                if variant in {"anchor_support_only", "soft_full"}:
                    utility = base_scores[batch_index] + anchor_support_weight * support[batch_index]
                elif variant == "top_confidence":
                    utility = confidence[batch_index]
                else:
                    utility = base_scores[batch_index]
                penalty = (
                    float(conflict_penalty)
                    if variant in {"soft_no_anchor", "soft_full"}
                    else 0.0
                )
                records.append(
                    {
                        "positions": canonical,
                        "seed": int(ranked_subset[0]),
                        "score": _subset_objective(
                            list(ranked_subset),
                            utility,
                            conflict_output.matrix[batch_index],
                            confidence[batch_index],
                            conflict_penalty=penalty,
                        ),
                        "mean_conflict": mean_conflict,
                        "max_conflict": max_conflict,
                        "support_sum": sum(
                            float(support[batch_index, p]) for p in ranked_subset
                        ),
                        "hard_fallback_count": 0,
                        "seed_rank": None,
                        "source": f"parallel_{variant}",
                        "fallback_source": "dependency_ranked_combination_refill",
                    }
                )
                if len(records) == target:
                    break
        if len(records) != target:
            raise RuntimeError("could not fill the feasible parallel candidate budget.")
        selected_by_batch.append(records)
        maximum_output_count = max(maximum_output_count, len(records))

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
            positions = tuple(record["positions"])
            position_tensor = torch.tensor(
                positions,
                device=dependency.device,
                dtype=torch.long,
            )
            candidate_masks[candidate_index, batch_index, position_tensor] = True
            candidate_valid[candidate_index, batch_index] = True
            proposal_scores[candidate_index, batch_index] = float(record["score"])
            seed_anchors[candidate_index, batch_index] = int(record["seed"])
            selected_positions[
                candidate_index, batch_index, : len(positions)
            ] = position_tensor
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
                "source": "parallel_by_batch",
                "source_by_batch": tuple(
                    record["source"] if record is not None else None
                    for record in records
                ),
                "variant_by_batch": tuple(
                    variant if record is not None else None for record in records
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
                    record["hard_fallback_count"] if record is not None else None
                    for record in records
                ),
                "fallback_source_by_batch": tuple(
                    record["fallback_source"] if record is not None else None
                    for record in records
                ),
                "is_fallback_by_batch": tuple(
                    bool(record["fallback_source"]) if record is not None else None
                    for record in records
                ),
                "seed_rank_by_batch": tuple(
                    record["seed_rank"] if record is not None else None
                    for record in records
                ),
            }
        )

    zero_signal = tuple(
        (
            bool(torch.all(base_scores[row, eligible[row]] == 0))
            if torch.any(eligible[row])
            else True
        )
        for row in range(batch_size)
    )
    constant_signal = tuple(
        (
            bool(
                torch.all(
                    base_scores[row, eligible[row]]
                    == base_scores[row, eligible[row]][0]
                )
            )
            if torch.any(eligible[row])
            else True
        )
        for row in range(batch_size)
    )
    configuration = {
        "proposal": "parallel_dependency_pool",
        "candidate_budget": candidate_budget,
        "requested_k_by_batch": tuple(int(value) for value in requested.tolist()),
        "parallel_variant": variant,
        "direction": direction,
        "target_weighting": target_weighting,
        "confidence_exponent": float(confidence_exponent),
        "conflict_definition": "0.5 * (D[i,j] + D[j,i])",
        "conflict_normalization": conflict_normalization,
        "conflict_scale_by_batch": tuple(
            float(value) for value in conflict_output.scale_by_batch.tolist()
        ),
        "conflict_penalty": float(conflict_penalty),
        "interaction_risk": "1 - min(confidence_i, confidence_r)",
        "hard_conflict_threshold": float(hard_conflict_threshold),
        "hard_fallback": "lowest_max_conflict_then_lowest_sum_then_utility",
        "anchor_support_weight": float(anchor_support_weight),
        "reliable_anchor_count_by_batch": tuple(
            int(value)
            for value in anchor_state.reliable_anchor_mask.sum(dim=-1).tolist()
        ),
        "zero_dependency_signal_by_batch": zero_signal,
        "constant_dependency_signal_by_batch": constant_signal,
        "position_temperature": float(position_temperature),
        "tie_breaking": "ascending compact position",
        "candidate_refill": "dependency_ranked_combinations",
    }
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
        configuration=configuration,
    )
