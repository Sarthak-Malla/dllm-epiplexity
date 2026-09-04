"""Build model-independent dependency-guided candidate batches.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_candidates.py -v
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

import torch


DEPENDENCY_DIRECTIONS = ("incoming", "outgoing", "symmetric")
TARGET_WEIGHTINGS = ("uniform", "entropy")
BASELINE_ONLY_PROPOSALS = (
    "top_confidence",
    "high_entropy",
    "random",
    "spaced",
    "position_confidence_gumbel",
    "current_mixed",
)


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor uses a non-boolean integer dtype."""
    return tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


@dataclass(frozen=True)
class CandidateBatch:
    """Validated candidate actions shaped for a later batched lookahead.

    ``candidate_valid[n, b]`` distinguishes an unavailable action from a valid
    zero-filled action. This matters when rows have different remaining-mask
    counts and therefore support different numbers of distinct candidates.
    """

    candidate_masks: torch.Tensor
    names: tuple[str, ...]
    proposal_scores: torch.Tensor
    seed_anchors: torch.Tensor
    selected_positions: torch.Tensor
    mean_within_set_dependency: torch.Tensor
    candidate_valid: torch.Tensor
    eligible_mask: torch.Tensor
    requested_k: torch.Tensor
    clipped_k: torch.Tensor
    metadata: tuple[Mapping[str, object], ...]
    generation_seed: int | None
    configuration: Mapping[str, object]
    action_sizes: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Reject malformed actions before they reach a verifier."""
        masks = self.candidate_masks
        if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
            raise ValueError("candidate_masks must have shape [N, B, T].")
        if masks.dtype != torch.bool:
            raise TypeError("candidate_masks must be boolean.")
        candidate_count, batch_size, sequence_length = masks.shape

        if len(self.names) != candidate_count:
            raise ValueError("names must have length N.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("candidate names must be unique.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("candidate names must be nonempty strings.")
        if len(self.metadata) != candidate_count:
            raise ValueError("metadata must have length N.")
        if any(not isinstance(row, Mapping) for row in self.metadata):
            raise TypeError("each candidate metadata entry must be a mapping.")

        expected_nb = (candidate_count, batch_size)
        shaped_tensors = {
            "proposal_scores": (self.proposal_scores, expected_nb),
            "seed_anchors": (self.seed_anchors, expected_nb),
            "mean_within_set_dependency": (
                self.mean_within_set_dependency,
                expected_nb,
            ),
            "candidate_valid": (self.candidate_valid, expected_nb),
            "eligible_mask": (
                self.eligible_mask,
                (batch_size, sequence_length),
            ),
            "requested_k": (self.requested_k, (batch_size,)),
            "clipped_k": (self.clipped_k, (batch_size,)),
        }
        for name, (tensor, expected_shape) in shaped_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor.")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}.")
            if tensor.device != masks.device:
                raise ValueError("all CandidateBatch tensors must share one device.")

        if self.eligible_mask.dtype != torch.bool:
            raise TypeError("eligible_mask must be boolean.")
        if self.candidate_valid.dtype != torch.bool:
            raise TypeError("candidate_valid must be boolean.")
        if not self.proposal_scores.is_floating_point():
            raise TypeError("proposal_scores must be floating point.")
        if not self.mean_within_set_dependency.is_floating_point():
            raise TypeError("mean_within_set_dependency must be floating point.")
        for name, tensor in (
            ("seed_anchors", self.seed_anchors),
            ("requested_k", self.requested_k),
            ("clipped_k", self.clipped_k),
        ):
            if not _is_integer_tensor(tensor):
                raise TypeError(f"{name} must use an integer dtype.")

        if self.selected_positions.ndim != 3:
            raise ValueError("selected_positions must have shape [N, B, K].")
        if self.selected_positions.shape[:2] != expected_nb:
            raise ValueError("selected_positions must begin with shape [N, B].")
        if self.selected_positions.device != masks.device:
            raise ValueError("all CandidateBatch tensors must share one device.")
        if not _is_integer_tensor(self.selected_positions):
            raise TypeError("selected_positions must use an integer dtype.")

        if torch.any(self.requested_k < 0) or torch.any(self.clipped_k < 0):
            raise ValueError("requested_k and clipped_k must be nonnegative.")
        eligible_counts = self.eligible_mask.sum(dim=-1).to(self.clipped_k.dtype)
        expected_clipped = torch.minimum(self.requested_k, eligible_counts)
        if not torch.equal(self.clipped_k, expected_clipped):
            raise ValueError("clipped_k must equal min(requested_k, eligible count).")
        storage_width = self.selected_positions.shape[-1]
        expected_width = int(self.clipped_k.max().item()) if batch_size else 0
        if storage_width != expected_width:
            raise ValueError("selected_positions width must equal max(clipped_k).")

        if torch.any(masks & ~self.eligible_mask.unsqueeze(0)):
            raise ValueError("candidate selections must stay inside eligible_mask.")
        inferred_action_sizes = masks.sum(dim=-1).to(self.clipped_k.dtype)
        if self.action_sizes is None:
            action_sizes = inferred_action_sizes
            expected_sizes = torch.where(
                self.candidate_valid,
                self.clipped_k.unsqueeze(0),
                torch.zeros_like(action_sizes),
            )
            if not torch.equal(action_sizes, expected_sizes):
                raise ValueError(
                    "candidate action size is inconsistent with clipped k."
                )
        else:
            action_sizes = self.action_sizes
            if not isinstance(action_sizes, torch.Tensor):
                raise TypeError("action_sizes must be a tensor or None.")
            if tuple(action_sizes.shape) != expected_nb:
                raise ValueError("action_sizes must have shape [N, B].")
            if action_sizes.device != masks.device:
                raise ValueError("action_sizes must share the candidate device.")
            if not _is_integer_tensor(action_sizes):
                raise TypeError("action_sizes must use an integer dtype.")
            if not torch.equal(action_sizes, inferred_action_sizes):
                raise ValueError("action_sizes must equal candidate mask counts.")
            if torch.any(
                self.candidate_valid
                & ((action_sizes <= 0) | (action_sizes > self.clipped_k.unsqueeze(0)))
            ):
                raise ValueError(
                    "valid variable-size candidates must lie in [1, clipped_k]."
                )
            if torch.any(~self.candidate_valid & (action_sizes != 0)):
                raise ValueError("invalid variable-size candidates must have size zero.")
        if torch.any(self.candidate_valid & (self.clipped_k.unsqueeze(0) == 0)):
            raise ValueError("a candidate cannot be valid when its clipped k is zero.")

        if torch.isnan(self.proposal_scores).any():
            raise ValueError("proposal_scores must not contain NaN.")
        if torch.any(
            self.candidate_valid & ~torch.isfinite(self.proposal_scores)
        ):
            raise ValueError("valid candidate proposal scores must be finite.")
        if torch.any(
            ~self.candidate_valid & ~torch.isneginf(self.proposal_scores)
        ):
            raise ValueError("invalid candidate proposal scores must be -inf.")
        if not torch.isfinite(self.mean_within_set_dependency).all():
            raise ValueError("mean within-set dependency must be finite.")

        for candidate_index in range(candidate_count):
            for batch_index in range(batch_size):
                valid = bool(self.candidate_valid[candidate_index, batch_index])
                anchor = int(self.seed_anchors[candidate_index, batch_index].item())
                positions = self.selected_positions[candidate_index, batch_index]
                kept = positions[positions >= 0]
                expected_position_count = (
                    int(action_sizes[candidate_index, batch_index].item())
                    if valid
                    else 0
                )
                if kept.numel() != expected_position_count:
                    raise ValueError(
                        "selected_positions does not match candidate validity and k."
                    )
                if not valid:
                    if anchor != -1 or torch.any(positions != -1):
                        raise ValueError(
                            "invalid candidates require -1 anchor and position padding."
                        )
                    continue
                if anchor < 0 or anchor >= sequence_length:
                    raise ValueError("seed anchor is outside the sequence.")
                if not bool(masks[candidate_index, batch_index, anchor]):
                    raise ValueError("seed anchor must be selected by its candidate.")
                expected_positions = torch.where(
                    masks[candidate_index, batch_index]
                )[0]
                if not torch.equal(kept, expected_positions):
                    raise ValueError(
                        "selected_positions must exactly encode the candidate mask."
                    )

        if self.generation_seed is not None and (
            isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
        ):
            raise TypeError("generation_seed must be an integer or None.")
        if not isinstance(self.configuration, Mapping):
            raise TypeError("configuration must be a mapping.")

    @property
    def shape(self) -> torch.Size:
        """Return the canonical [N, B, T] candidate-mask shape."""
        return self.candidate_masks.shape

    def as_legacy_dict(self) -> dict[str, torch.Tensor]:
        """Expose the current sampler's name-to-[B,T]-mask adapter."""
        return {
            name: self.candidate_masks[index]
            for index, name in enumerate(self.names)
        }


def _validate_score_inputs(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> tuple[int, int]:
    """Validate aligned [B,T,T], [B,T], and [B,T] score inputs."""
    if not isinstance(dependency, torch.Tensor) or dependency.ndim != 3:
        raise ValueError("dependency must have shape [B, T, T].")
    if not dependency.is_floating_point():
        raise TypeError("dependency must be floating point.")
    batch_size, query_length, key_length = dependency.shape
    if query_length != key_length:
        raise ValueError("dependency must be square over sequence positions.")
    expected_shape = (batch_size, query_length)
    if not isinstance(entropy, torch.Tensor) or entropy.shape != expected_shape:
        raise ValueError(f"entropy must have shape {expected_shape}.")
    if not entropy.is_floating_point():
        raise TypeError("entropy must be floating point.")
    if (
        not isinstance(eligible_mask, torch.Tensor)
        or eligible_mask.shape != expected_shape
    ):
        raise ValueError(f"eligible_mask must have shape {expected_shape}.")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean.")
    if entropy.device != dependency.device or eligible_mask.device != dependency.device:
        raise ValueError("dependency, entropy, and eligible_mask must share a device.")
    if not torch.isfinite(dependency).all() or torch.any(dependency < 0):
        raise ValueError("dependency must be finite and nonnegative.")
    if not torch.isfinite(entropy).all() or torch.any(entropy < 0):
        raise ValueError("entropy must be finite and nonnegative.")
    return batch_size, query_length


def dependency_anchor_scores(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    direction: str = "outgoing",
    target_weighting: str = "entropy",
    confidence: torch.Tensor | None = None,
    confidence_exponent: float = 0.0,
) -> torch.Tensor:
    """Score eligible anchors under an explicit D[query,key] convention.

    The G2 working default is outgoing, entropy-weighted, and eta=0:
    ``score[i] = sum_j D[i,j] * entropy[j]`` for eligible ``i != j``.
    Invalid positions receive negative infinity. Valid positions receive zero
    when dependency or the target set is empty, giving a deterministic fallback.
    """
    batch_size, sequence_length = _validate_score_inputs(
        dependency,
        entropy,
        eligible_mask,
    )
    if direction not in DEPENDENCY_DIRECTIONS:
        raise ValueError(f"Unsupported dependency direction: {direction!r}.")
    if target_weighting not in TARGET_WEIGHTINGS:
        raise ValueError(f"Unsupported target weighting: {target_weighting!r}.")
    if not math.isfinite(confidence_exponent) or confidence_exponent < 0:
        raise ValueError("confidence_exponent must be finite and nonnegative.")

    score_dtype = torch.promote_types(dependency.dtype, entropy.dtype)
    matrix = dependency.to(dtype=score_dtype)
    weights = entropy.to(dtype=score_dtype)
    if target_weighting == "uniform":
        weights = torch.ones_like(weights)
    weights = torch.where(eligible_mask, weights, torch.zeros_like(weights))

    pair_mask = eligible_mask.unsqueeze(-1) & eligible_mask.unsqueeze(-2)
    diagonal = torch.eye(
        sequence_length,
        device=dependency.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    matrix = torch.where(
        pair_mask & ~diagonal,
        matrix,
        torch.zeros_like(matrix),
    )
    if direction == "incoming":
        influence = torch.bmm(matrix.transpose(1, 2), weights.unsqueeze(-1))
    elif direction == "outgoing":
        influence = torch.bmm(matrix, weights.unsqueeze(-1))
    else:
        symmetric = 0.5 * (matrix + matrix.transpose(1, 2))
        influence = torch.bmm(symmetric, weights.unsqueeze(-1))
    scores = influence.squeeze(-1)

    if confidence_exponent > 0:
        expected_shape = (batch_size, sequence_length)
        if (
            not isinstance(confidence, torch.Tensor)
            or confidence.shape != expected_shape
        ):
            raise ValueError(f"confidence must have shape {expected_shape}.")
        if not confidence.is_floating_point():
            raise TypeError("confidence must be floating point.")
        if confidence.device != dependency.device:
            raise ValueError("confidence must share the dependency device.")
        if not torch.isfinite(confidence).all() or torch.any(
            (confidence < 0) | (confidence > 1)
        ):
            raise ValueError("confidence must be finite and lie in [0, 1].")
        scores = scores * confidence.to(score_dtype).pow(confidence_exponent)

    negative_infinity = torch.full_like(scores, -torch.inf)
    return torch.where(eligible_mask, scores, negative_infinity)


def _build_ranked_k1_candidate_batch(
    *,
    base_scores: torch.Tensor,
    ranking: torch.Tensor,
    eligible_mask: torch.Tensor,
    candidate_budget: int,
    generation_seed: int | None,
    name_prefix: str,
    metadata_common: Mapping[str, object],
    configuration: Mapping[str, object],
    sampling_scores: torch.Tensor | None = None,
    gumbel_noise: torch.Tensor | None = None,
) -> CandidateBatch:
    """Materialize one ranked position list as validated k=1 actions."""
    batch_size, sequence_length = eligible_mask.shape
    expected_shape = (batch_size, sequence_length)
    if base_scores.shape != expected_shape or ranking.shape != expected_shape:
        raise ValueError("base_scores and ranking must match eligible_mask.")
    valid_counts = eligible_mask.sum(dim=-1, dtype=torch.long)
    max_candidate_count = int(valid_counts.max().item()) if batch_size else 0
    candidate_count = min(candidate_budget, max_candidate_count)
    requested_k = torch.ones(
        batch_size,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    clipped_k = torch.minimum(requested_k, valid_counts)
    storage_width = int(clipped_k.max().item()) if batch_size else 0

    candidate_masks = torch.zeros(
        (candidate_count, batch_size, sequence_length),
        device=eligible_mask.device,
        dtype=torch.bool,
    )
    candidate_valid = torch.zeros(
        (candidate_count, batch_size),
        device=eligible_mask.device,
        dtype=torch.bool,
    )
    proposal_scores = torch.full(
        (candidate_count, batch_size),
        -torch.inf,
        device=eligible_mask.device,
        dtype=base_scores.dtype,
    )
    seed_anchors = torch.full(
        (candidate_count, batch_size),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    selected_positions = torch.full(
        (candidate_count, batch_size, storage_width),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )

    batch_indices = torch.arange(batch_size, device=eligible_mask.device)
    for candidate_index in range(candidate_count):
        row_valid = valid_counts > candidate_index
        anchors = ranking[:, candidate_index]
        selected_batches = batch_indices[row_valid]
        selected_anchors = anchors[row_valid]
        candidate_valid[candidate_index, row_valid] = True
        candidate_masks[
            candidate_index,
            selected_batches,
            selected_anchors,
        ] = True
        proposal_scores[candidate_index, row_valid] = base_scores[
            row_valid,
            selected_anchors,
        ]
        seed_anchors[candidate_index, row_valid] = selected_anchors
        if storage_width:
            selected_positions[candidate_index, row_valid, 0] = selected_anchors

    names = tuple(
        f"{name_prefix}_{candidate_index}" for candidate_index in range(candidate_count)
    )
    metadata_rows = []
    for candidate_index in range(candidate_count):
        row = dict(metadata_common)
        row["rank"] = candidate_index
        if sampling_scores is not None:
            row["base_scores_by_batch"] = tuple(
                (
                    float(proposal_scores[candidate_index, batch_index].item())
                    if candidate_valid[candidate_index, batch_index]
                    else None
                )
                for batch_index in range(batch_size)
            )
            row["sampling_scores_by_batch"] = tuple(
                (
                    float(
                        sampling_scores[
                            batch_index,
                            seed_anchors[candidate_index, batch_index],
                        ].item()
                    )
                    if candidate_valid[candidate_index, batch_index]
                    else None
                )
                for batch_index in range(batch_size)
            )
        if gumbel_noise is not None:
            row["gumbel_noise_by_batch"] = tuple(
                (
                    float(
                        gumbel_noise[
                            batch_index,
                            seed_anchors[candidate_index, batch_index],
                        ].item()
                    )
                    if candidate_valid[candidate_index, batch_index]
                    else None
                )
                for batch_index in range(batch_size)
            )
        metadata_rows.append(row)

    return CandidateBatch(
        candidate_masks=candidate_masks,
        names=names,
        proposal_scores=proposal_scores,
        seed_anchors=seed_anchors,
        selected_positions=selected_positions,
        mean_within_set_dependency=torch.zeros_like(proposal_scores),
        candidate_valid=candidate_valid,
        eligible_mask=eligible_mask,
        requested_k=requested_k,
        clipped_k=clipped_k,
        metadata=tuple(metadata_rows),
        generation_seed=generation_seed,
        configuration=configuration,
    )


def generate_dependency_top_n(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    direction: str = "outgoing",
    target_weighting: str = "entropy",
    confidence: torch.Tensor | None = None,
    confidence_exponent: float = 0.0,
    generation_seed: int | None = None,
    name_prefix: str = "dependency",
) -> CandidateBatch:
    """Generate distinct deterministic k=1 actions by descending anchor score."""
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")
    if generation_seed is not None and (
        isinstance(generation_seed, bool) or not isinstance(generation_seed, int)
    ):
        raise TypeError("generation_seed must be an integer or None.")

    scores = dependency_anchor_scores(
        dependency,
        entropy,
        eligible_mask,
        direction=direction,
        target_weighting=target_weighting,
        confidence=confidence,
        confidence_exponent=confidence_exponent,
    )
    # Stable sorting preserves ascending absolute positions for tied scores.
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)
    zero_signal_by_batch = tuple(
        bool(torch.all(scores[batch_index, eligible_mask[batch_index]] == 0))
        for batch_index in range(eligible_mask.shape[0])
    )
    constant_signal_by_batch = tuple(
        (
            bool(
                torch.all(
                    scores[batch_index, eligible_mask[batch_index]]
                    == scores[batch_index, eligible_mask[batch_index]][0]
                )
            )
            if torch.any(eligible_mask[batch_index])
            else True
        )
        for batch_index in range(eligible_mask.shape[0])
    )
    configuration = {
        "candidate_budget": candidate_budget,
        "requested_k": 1,
        "direction_convention": "D[query,key]: query uses key as context",
        "direction": direction,
        "target_weighting": target_weighting,
        "confidence_exponent": float(confidence_exponent),
        "sampling_strategy": "deterministic_top_n",
        "zero_dependency_signal_by_batch": zero_signal_by_batch,
        "constant_dependency_signal_by_batch": constant_signal_by_batch,
        "tie_breaking": "ascending absolute position",
        "zero_dependency_fallback": "ascending absolute position",
    }
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={
            "source": "dependency",
            "direction": direction,
            "target_weighting": target_weighting,
            "confidence_exponent": float(confidence_exponent),
        },
        configuration=configuration,
    )


def generate_dependency_gumbel_top_n(
    dependency: torch.Tensor,
    entropy: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    position_temperature: float,
    generation_seed: int | None,
    direction: str = "outgoing",
    target_weighting: str = "entropy",
    confidence: torch.Tensor | None = None,
    confidence_exponent: float = 0.0,
    name_prefix: str = "dependency",
) -> CandidateBatch:
    """Sample distinct anchors with standard, locally seeded Gumbel-top-k.

    Positive temperatures rank ``base_score / temperature + Gumbel(0,1)``.
    Gumbel noise is generated by a private CPU generator to keep global RNG
    state unchanged and make a fixed seed device-independent. Temperature zero
    delegates exactly to deterministic top-N generation.
    """
    if isinstance(position_temperature, bool) or not isinstance(
        position_temperature,
        (int, float),
    ):
        raise TypeError("position_temperature must be numeric.")
    position_temperature = float(position_temperature)
    if not math.isfinite(position_temperature) or position_temperature < 0:
        raise ValueError("position_temperature must be finite and nonnegative.")
    if position_temperature == 0:
        return generate_dependency_top_n(
            dependency,
            entropy,
            eligible_mask,
            candidate_budget=candidate_budget,
            direction=direction,
            target_weighting=target_weighting,
            confidence=confidence,
            confidence_exponent=confidence_exponent,
            generation_seed=generation_seed,
            name_prefix=name_prefix,
        )
    if (
        generation_seed is None
        or isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError(
            "positive position_temperature requires a nonnegative integer seed."
        )
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")

    scores = dependency_anchor_scores(
        dependency,
        entropy,
        eligible_mask,
        direction=direction,
        target_weighting=target_weighting,
        confidence=confidence,
        confidence_exponent=confidence_exponent,
    )
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(generation_seed)
    uniform = torch.rand(
        scores.shape,
        generator=local_generator,
        device="cpu",
        dtype=torch.float64,
    )
    floating_info = torch.finfo(uniform.dtype)
    uniform = uniform.clamp(
        min=floating_info.tiny,
        max=1.0 - floating_info.eps,
    )
    gumbel_noise = -torch.log(-torch.log(uniform))
    gumbel_noise = gumbel_noise.to(device=scores.device)
    sampling_scores = scores.to(torch.float64) / position_temperature + gumbel_noise
    sampling_scores = torch.where(
        eligible_mask,
        sampling_scores,
        torch.full_like(sampling_scores, -torch.inf),
    )
    ranking = torch.argsort(
        sampling_scores,
        dim=-1,
        descending=True,
        stable=True,
    )
    configuration = {
        "candidate_budget": candidate_budget,
        "requested_k": 1,
        "direction_convention": "D[query,key]: query uses key as context",
        "direction": direction,
        "target_weighting": target_weighting,
        "confidence_exponent": float(confidence_exponent),
        "sampling_strategy": "gumbel_top_n_without_replacement",
        "sampling_logits": "base_score / position_temperature",
        "position_temperature": position_temperature,
        "rng_device": "cpu",
        "zero_dependency_signal_by_batch": tuple(
            bool(torch.all(scores[batch_index, eligible_mask[batch_index]] == 0))
            for batch_index in range(eligible_mask.shape[0])
        ),
        "tie_breaking": "ascending absolute position after perturbed-score ties",
        "zero_dependency_fallback": "seeded Gumbel ranking",
    }
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={
            "source": "dependency_gumbel",
            "direction": direction,
            "target_weighting": target_weighting,
            "confidence_exponent": float(confidence_exponent),
            "position_temperature": position_temperature,
        },
        configuration=configuration,
        sampling_scores=sampling_scores,
        gumbel_noise=gumbel_noise,
    )


def _validate_compatible_candidate_batches(
    batches: Sequence[CandidateBatch],
) -> CandidateBatch:
    """Return a reference batch after checking composition compatibility."""
    if not batches:
        raise ValueError("at least one candidate batch is required.")
    reference = batches[0]
    if torch.any(reference.requested_k != 1):
        raise ValueError("P3.5 deduplication currently supports only k=1.")
    for batch in batches[1:]:
        if batch.candidate_masks.shape[1:] != reference.candidate_masks.shape[1:]:
            raise ValueError("candidate batches must share batch and sequence shape.")
        if batch.candidate_masks.device != reference.candidate_masks.device:
            raise ValueError("candidate batches must share one device.")
        if not torch.equal(batch.eligible_mask, reference.eligible_mask):
            raise ValueError("candidate batches must share eligible_mask.")
        if not torch.equal(batch.requested_k, reference.requested_k):
            raise ValueError("candidate batches must share requested_k.")
        if not torch.equal(batch.clipped_k, reference.clipped_k):
            raise ValueError("candidate batches must share clipped_k.")
    return reference


def deduplicate_and_refill_k1_candidates(
    primary_batches: Sequence[CandidateBatch],
    *,
    candidate_budget: int,
    dependency_fallback: CandidateBatch | None = None,
    confidence_fallback: CandidateBatch | None = None,
    generation_seed: int,
    name_prefix: str = "candidate",
    allow_random_fallback: bool = True,
) -> CandidateBatch:
    """Stably deduplicate k=1 actions and fill dependency/confidence/random.

    Candidate actions are considered independently for each batch row. Primary
    batches retain their supplied order. Missing unique actions are then taken
    from the deterministic dependency batch, the confidence batch, and finally
    a private-CPU-generator random permutation of remaining eligible positions.
    """
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("generation_seed must be a nonnegative integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")
    if not isinstance(allow_random_fallback, bool):
        raise TypeError("allow_random_fallback must be boolean.")
    if isinstance(primary_batches, CandidateBatch) or not isinstance(
        primary_batches,
        Sequence,
    ):
        raise TypeError("primary_batches must be a sequence of CandidateBatch.")
    if any(not isinstance(batch, CandidateBatch) for batch in primary_batches):
        raise TypeError("primary_batches must contain only CandidateBatch values.")

    ordered_batches: list[tuple[CandidateBatch, str | None, bool]] = [
        (batch, None, False) for batch in primary_batches
    ]
    if dependency_fallback is not None:
        if not isinstance(dependency_fallback, CandidateBatch):
            raise TypeError("dependency_fallback must be a CandidateBatch or None.")
        ordered_batches.append(
            (dependency_fallback, "dependency_fallback", True)
        )
    if confidence_fallback is not None:
        if not isinstance(confidence_fallback, CandidateBatch):
            raise TypeError("confidence_fallback must be a CandidateBatch or None.")
        ordered_batches.append((confidence_fallback, "confidence_fallback", True))

    reference = _validate_compatible_candidate_batches(
        [batch for batch, _, _ in ordered_batches]
    )
    device = reference.candidate_masks.device
    batch_size, sequence_length = reference.eligible_mask.shape
    eligible_counts = reference.eligible_mask.sum(dim=-1, dtype=torch.long)
    maximum_actions = int(eligible_counts.max().item()) if batch_size else 0
    output_count = min(candidate_budget, maximum_actions)
    row_targets = torch.minimum(
        eligible_counts,
        torch.full_like(eligible_counts, candidate_budget),
    )

    selected_by_batch: list[list[dict[str, object]]] = [
        [] for _ in range(batch_size)
    ]
    seen_by_batch: list[set[tuple[int, ...]]] = [
        set() for _ in range(batch_size)
    ]
    duplicates_skipped = [0] * batch_size

    for source_batch, source_override, is_fallback in ordered_batches:
        for candidate_index, origin_name in enumerate(source_batch.names):
            source_metadata = source_batch.metadata[candidate_index]
            source_name = source_override or str(
                source_metadata.get("source", "primary")
            )
            for batch_index in range(batch_size):
                if len(selected_by_batch[batch_index]) >= int(
                    row_targets[batch_index].item()
                ):
                    continue
                if not bool(
                    source_batch.candidate_valid[candidate_index, batch_index]
                ):
                    continue
                positions_tensor = source_batch.selected_positions[
                    candidate_index,
                    batch_index,
                ]
                positions = tuple(
                    int(position)
                    for position in positions_tensor[positions_tensor >= 0].tolist()
                )
                if positions in seen_by_batch[batch_index]:
                    duplicates_skipped[batch_index] += 1
                    continue
                seen_by_batch[batch_index].add(positions)
                selected_by_batch[batch_index].append(
                    {
                        "positions": positions,
                        "score": float(
                            source_batch.proposal_scores[
                                candidate_index,
                                batch_index,
                            ].item()
                        ),
                        "anchor": int(
                            source_batch.seed_anchors[
                                candidate_index,
                                batch_index,
                            ].item()
                        ),
                        "mean_dependency": float(
                            source_batch.mean_within_set_dependency[
                                candidate_index,
                                batch_index,
                            ].item()
                        ),
                        "source": source_name,
                        "origin_name": origin_name,
                        "is_fallback": is_fallback,
                    }
                )

    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(generation_seed)
    for batch_index in range(batch_size):
        target_count = int(row_targets[batch_index].item())
        if len(selected_by_batch[batch_index]) >= target_count:
            continue
        if not allow_random_fallback:
            raise RuntimeError(
                "candidate sources could not fill the k=1 budget without random."
            )
        eligible_positions = torch.where(reference.eligible_mask[batch_index])[0]
        remaining = [
            int(position)
            for position in eligible_positions.tolist()
            if (int(position),) not in seen_by_batch[batch_index]
        ]
        permutation = torch.randperm(
            len(remaining),
            generator=local_generator,
            device="cpu",
        ).tolist()
        for permutation_index in permutation:
            if len(selected_by_batch[batch_index]) >= target_count:
                break
            position = remaining[permutation_index]
            positions = (position,)
            seen_by_batch[batch_index].add(positions)
            selected_by_batch[batch_index].append(
                {
                    "positions": positions,
                    "score": 0.0,
                    "anchor": position,
                    "mean_dependency": 0.0,
                    "source": "random_fallback",
                    "origin_name": None,
                    "is_fallback": True,
                }
            )
        if len(selected_by_batch[batch_index]) != target_count:
            raise RuntimeError("candidate refill did not exhaust the k=1 action space.")

    score_dtype = reference.proposal_scores.dtype
    for source_batch, _, _ in ordered_batches[1:]:
        score_dtype = torch.promote_types(
            score_dtype,
            source_batch.proposal_scores.dtype,
        )
    candidate_masks = torch.zeros(
        (output_count, batch_size, sequence_length),
        device=device,
        dtype=torch.bool,
    )
    candidate_valid = torch.zeros(
        (output_count, batch_size),
        device=device,
        dtype=torch.bool,
    )
    proposal_scores = torch.full(
        (output_count, batch_size),
        -torch.inf,
        device=device,
        dtype=score_dtype,
    )
    seed_anchors = torch.full(
        (output_count, batch_size),
        -1,
        device=device,
        dtype=torch.long,
    )
    selected_positions = torch.full(
        (output_count, batch_size, 1 if maximum_actions else 0),
        -1,
        device=device,
        dtype=torch.long,
    )
    mean_dependency = torch.zeros_like(proposal_scores)

    for batch_index, selected in enumerate(selected_by_batch):
        for candidate_index, record in enumerate(selected):
            positions = record["positions"]
            if not isinstance(positions, tuple) or len(positions) != 1:
                raise RuntimeError("P3.5 encountered a non-k=1 candidate.")
            position = positions[0]
            candidate_masks[candidate_index, batch_index, position] = True
            candidate_valid[candidate_index, batch_index] = True
            proposal_scores[candidate_index, batch_index] = float(record["score"])
            seed_anchors[candidate_index, batch_index] = int(record["anchor"])
            selected_positions[candidate_index, batch_index, 0] = position
            mean_dependency[candidate_index, batch_index] = float(
                record["mean_dependency"]
            )

    metadata = []
    for candidate_index in range(output_count):
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
                "source": "composed_by_batch",
                "source_by_batch": tuple(
                    record["source"] if record is not None else None
                    for record in records
                ),
                "origin_name_by_batch": tuple(
                    record["origin_name"] if record is not None else None
                    for record in records
                ),
                "is_fallback_by_batch": tuple(
                    record["is_fallback"] if record is not None else None
                    for record in records
                ),
            }
        )

    source_counts_by_batch = []
    for selected in selected_by_batch:
        counts: dict[str, int] = {}
        for record in selected:
            source = str(record["source"])
            counts[source] = counts.get(source, 0) + 1
        source_counts_by_batch.append(counts)
    configuration = {
        "candidate_budget": candidate_budget,
        "requested_k": 1,
        "composition": "stable_per_row_deduplication_then_refill",
        "fallback_order": tuple(
            source
            for source, enabled in (
                ("dependency_fallback", dependency_fallback is not None),
                ("confidence_fallback", confidence_fallback is not None),
                ("random_fallback", allow_random_fallback),
            )
            if enabled
        ),
        "random_fallback_enabled": allow_random_fallback,
        "duplicates_skipped_by_batch": tuple(duplicates_skipped),
        "source_counts_by_batch": tuple(source_counts_by_batch),
        "random_rng_device": "cpu",
    }
    return CandidateBatch(
        candidate_masks=candidate_masks,
        names=tuple(
            f"{name_prefix}_{candidate_index}"
            for candidate_index in range(output_count)
        ),
        proposal_scores=proposal_scores,
        seed_anchors=seed_anchors,
        selected_positions=selected_positions,
        mean_within_set_dependency=mean_dependency,
        candidate_valid=candidate_valid,
        eligible_mask=reference.eligible_mask,
        requested_k=reference.requested_k,
        clipped_k=reference.clipped_k,
        metadata=tuple(metadata),
        generation_seed=generation_seed,
        configuration=configuration,
    )


def _candidate_prefix(batch: CandidateBatch, count: int) -> CandidateBatch:
    """Return the first candidate slots while preserving validated metadata."""
    count = max(0, min(count, batch.candidate_masks.shape[0]))
    configuration = dict(batch.configuration)
    configuration["candidate_prefix_count"] = count
    return replace(
        batch,
        candidate_masks=batch.candidate_masks[:count],
        names=batch.names[:count],
        proposal_scores=batch.proposal_scores[:count],
        seed_anchors=batch.seed_anchors[:count],
        selected_positions=batch.selected_positions[:count],
        mean_within_set_dependency=batch.mean_within_set_dependency[:count],
        candidate_valid=batch.candidate_valid[:count],
        metadata=batch.metadata[:count],
        configuration=configuration,
    )


def compose_principled_dependency_candidates(
    deterministic: CandidateBatch,
    diverse: CandidateBatch | None,
    *,
    candidate_budget: int,
    generation_seed: int,
    name_prefix: str = "dependency_candidate",
) -> CandidateBatch:
    """Compose the proposed pool using dependency evidence only.

    The highest deterministic dependency candidate is always first. Diverse
    dependency-Gumbel actions follow, duplicates are removed, and the next
    deterministic dependency ranks fill any gaps. Confidence, entropy, spaced,
    and random baselines are deliberately excluded from this method.
    """
    if not isinstance(deterministic, CandidateBatch):
        raise TypeError("deterministic must be a CandidateBatch.")
    if deterministic.configuration.get("sampling_strategy") != (
        "deterministic_top_n"
    ) or "direction" not in deterministic.configuration:
        raise ValueError("deterministic must be a dependency top-N batch.")
    primary_batches = [_candidate_prefix(deterministic, 1)]
    if diverse is not None:
        if not isinstance(diverse, CandidateBatch):
            raise TypeError("diverse must be a CandidateBatch or None.")
        if diverse.configuration.get("sampling_strategy") not in {
            "gumbel_top_n_without_replacement",
            "deterministic_top_n",
        } or "direction" not in diverse.configuration:
            raise ValueError("diverse must be a dependency-Gumbel batch.")
        for field in ("direction", "target_weighting", "confidence_exponent"):
            if diverse.configuration.get(field) != deterministic.configuration.get(
                field
            ):
                raise ValueError(
                    "deterministic and diverse dependency configurations "
                    f"must share {field}."
                )
        primary_batches.append(diverse)

    composed = deduplicate_and_refill_k1_candidates(
        tuple(primary_batches),
        candidate_budget=candidate_budget,
        dependency_fallback=deterministic,
        confidence_fallback=None,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        allow_random_fallback=False,
    )
    configuration = dict(composed.configuration)
    configuration.update(
        {
            "proposal": "principled_dependency_pool",
            "primary_order": (
                "deterministic_dependency_best",
                "dependency_gumbel_diversity",
            ),
            "unprincipled_fallbacks": (),
            "zero_dependency_signal_by_batch": deterministic.configuration.get(
                "zero_dependency_signal_by_batch"
            ),
            "constant_dependency_signal_by_batch": (
                deterministic.configuration.get(
                    "constant_dependency_signal_by_batch"
                )
            ),
        }
    )
    return replace(composed, configuration=configuration)


def _validated_position_values(
    values: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> torch.Tensor:
    """Validate a finite [B,T] position-score tensor on one device."""
    if not isinstance(eligible_mask, torch.Tensor) or eligible_mask.ndim != 2:
        raise ValueError("eligible_mask must have shape [B, T].")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean.")
    if not isinstance(values, torch.Tensor) or values.shape != eligible_mask.shape:
        raise ValueError(f"{name} must have shape {tuple(eligible_mask.shape)}.")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if values.device != eligible_mask.device:
        raise ValueError(f"{name} and eligible_mask must share a device.")
    invalid = ~torch.isfinite(values) | (values < minimum)
    if maximum is not None:
        invalid |= values > maximum
    if torch.any(invalid):
        interval = (
            f"[{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be finite and lie in {interval}.")
    return values


def _deterministic_score_candidates(
    values: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    source: str,
    generation_seed: int | None,
    name_prefix: str,
    configuration: Mapping[str, object],
) -> CandidateBatch:
    """Build stable descending k=1 candidates from validated position values."""
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")
    scores = torch.where(
        eligible_mask,
        values,
        torch.full_like(values, -torch.inf),
    )
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={"source": source},
        configuration=configuration,
    )


def generate_top_confidence_candidates(
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    generation_seed: int | None = None,
    name_prefix: str = "confidence",
) -> CandidateBatch:
    """Generate deterministic k=1 actions by descending token confidence."""
    confidence = _validated_position_values(
        confidence,
        eligible_mask,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    return _deterministic_score_candidates(
        confidence,
        eligible_mask,
        candidate_budget=candidate_budget,
        source="top_confidence",
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        configuration={
            "candidate_budget": candidate_budget,
            "requested_k": 1,
            "proposal": "top_confidence",
            "sampling_strategy": "deterministic_top_n",
            "tie_breaking": "ascending absolute position",
        },
    )


def generate_high_entropy_candidates(
    entropy: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    generation_seed: int | None = None,
    name_prefix: str = "high_entropy",
) -> CandidateBatch:
    """Generate deterministic k=1 actions by descending measured entropy."""
    entropy = _validated_position_values(
        entropy,
        eligible_mask,
        name="entropy",
        minimum=0.0,
    )
    return _deterministic_score_candidates(
        entropy,
        eligible_mask,
        candidate_budget=candidate_budget,
        source="high_entropy",
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        configuration={
            "candidate_budget": candidate_budget,
            "requested_k": 1,
            "proposal": "high_entropy",
            "sampling_strategy": "deterministic_top_n",
            "tie_breaking": "ascending absolute position",
        },
    )


def generate_random_candidates(
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    generation_seed: int,
    name_prefix: str = "random",
) -> CandidateBatch:
    """Generate a locally seeded uniform permutation of valid k=1 actions."""
    if not isinstance(eligible_mask, torch.Tensor) or eligible_mask.ndim != 2:
        raise ValueError("eligible_mask must have shape [B, T].")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean.")
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("generation_seed must be a nonnegative integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")

    batch_size, sequence_length = eligible_mask.shape
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(generation_seed)
    scores = torch.full(
        (batch_size, sequence_length),
        -torch.inf,
        device=eligible_mask.device,
        dtype=torch.float64,
    )
    for batch_index in range(batch_size):
        positions = torch.where(eligible_mask[batch_index])[0].cpu()
        permutation = torch.randperm(
            positions.numel(),
            generator=local_generator,
            device="cpu",
        )
        ordered_positions = positions[permutation]
        for rank, position in enumerate(ordered_positions.tolist()):
            scores[batch_index, position] = float(positions.numel() - rank)
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={"source": "random"},
        configuration={
            "candidate_budget": candidate_budget,
            "requested_k": 1,
            "proposal": "uniform_random_without_replacement",
            "sampling_strategy": "locally_seeded_randperm",
            "rng_device": "cpu",
        },
    )


def generate_spaced_candidates(
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    generation_seed: int | None = None,
    name_prefix: str = "spaced",
) -> CandidateBatch:
    """Generate up to N distinct positions spread across each valid span."""
    if not isinstance(eligible_mask, torch.Tensor) or eligible_mask.ndim != 2:
        raise ValueError("eligible_mask must have shape [B, T].")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean.")
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string.")

    batch_size, sequence_length = eligible_mask.shape
    scores = torch.full(
        (batch_size, sequence_length),
        -torch.inf,
        device=eligible_mask.device,
        dtype=torch.float64,
    )
    selected_ranks_by_batch = []
    for batch_index in range(batch_size):
        positions = torch.where(eligible_mask[batch_index])[0].tolist()
        selected_count = min(candidate_budget, len(positions))
        if selected_count <= 1:
            selected_ranks = [0] if selected_count else []
        else:
            selected_ranks = [
                (candidate_index * (len(positions) - 1))
                // (selected_count - 1)
                for candidate_index in range(selected_count)
            ]
        selected_ranks_by_batch.append(tuple(selected_ranks))
        selected_rank_set = set(selected_ranks)
        ordered_positions = [positions[index] for index in selected_ranks]
        ordered_positions.extend(
            position
            for index, position in enumerate(positions)
            if index not in selected_rank_set
        )
        for rank, position in enumerate(ordered_positions):
            scores[batch_index, position] = float(len(positions) - rank)
    ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={"source": "spaced"},
        configuration={
            "candidate_budget": candidate_budget,
            "requested_k": 1,
            "proposal": "evenly_spaced_valid_position_ranks",
            "selected_ranks_by_batch": tuple(selected_ranks_by_batch),
            "tie_breaking": "ascending absolute position",
        },
    )


def generate_position_confidence_gumbel_candidates(
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    candidate_budget: int,
    position_temperature: float,
    generation_seed: int | None,
    name_prefix: str = "position_confidence_gumbel",
) -> CandidateBatch:
    """Sample position-confidence candidates, distinct from token Gumbel-Max."""
    confidence = _validated_position_values(
        confidence,
        eligible_mask,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if isinstance(position_temperature, bool) or not isinstance(
        position_temperature,
        (int, float),
    ):
        raise TypeError("position_temperature must be numeric.")
    position_temperature = float(position_temperature)
    if not math.isfinite(position_temperature) or position_temperature < 0:
        raise ValueError("position_temperature must be finite and nonnegative.")
    if position_temperature == 0:
        return generate_top_confidence_candidates(
            confidence,
            eligible_mask,
            candidate_budget=candidate_budget,
            generation_seed=generation_seed,
            name_prefix=name_prefix,
        )
    if (
        generation_seed is None
        or isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError(
            "positive position_temperature requires a nonnegative integer seed."
        )
    if (
        isinstance(candidate_budget, bool)
        or not isinstance(candidate_budget, int)
        or candidate_budget <= 0
    ):
        raise ValueError("candidate_budget must be a positive integer.")

    scores = torch.where(
        eligible_mask,
        confidence,
        torch.full_like(confidence, -torch.inf),
    )
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(generation_seed)
    uniform = torch.rand(
        scores.shape,
        generator=local_generator,
        device="cpu",
        dtype=torch.float64,
    )
    floating_info = torch.finfo(uniform.dtype)
    uniform = uniform.clamp(
        min=floating_info.tiny,
        max=1.0 - floating_info.eps,
    )
    gumbel_noise = -torch.log(-torch.log(uniform)).to(scores.device)
    sampling_scores = scores.to(torch.float64) / position_temperature + gumbel_noise
    sampling_scores = torch.where(
        eligible_mask,
        sampling_scores,
        torch.full_like(sampling_scores, -torch.inf),
    )
    ranking = torch.argsort(
        sampling_scores,
        dim=-1,
        descending=True,
        stable=True,
    )
    return _build_ranked_k1_candidate_batch(
        base_scores=scores,
        ranking=ranking,
        eligible_mask=eligible_mask,
        candidate_budget=candidate_budget,
        generation_seed=generation_seed,
        name_prefix=name_prefix,
        metadata_common={
            "source": "position_confidence_gumbel",
            "position_temperature": position_temperature,
        },
        configuration={
            "candidate_budget": candidate_budget,
            "requested_k": 1,
            "proposal": "position_confidence_gumbel",
            "not_token_logit_gumbel": True,
            "sampling_strategy": "gumbel_top_n_without_replacement",
            "sampling_logits": "confidence / position_temperature",
            "position_temperature": position_temperature,
            "rng_device": "cpu",
        },
        sampling_scores=sampling_scores,
        gumbel_noise=gumbel_noise,
    )


def generate_current_mixed_candidates(
    confidence: torch.Tensor,
    eligible_mask: torch.Tensor,
    *,
    generation_seed: int,
) -> CandidateBatch:
    """Reproduce the current k=1 mixed ensemble in CandidateBatch form.

    Historical behavior calls its fourth candidate ``high_entropy`` but ranks
    by lowest confidence rather than measured entropy. Metadata preserves that
    fact so it cannot be confused with ``generate_high_entropy_candidates``.
    """
    confidence = _validated_position_values(
        confidence,
        eligible_mask,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or generation_seed < 0
    ):
        raise ValueError("generation_seed must be a nonnegative integer.")
    batch_size, sequence_length = eligible_mask.shape
    candidate_count = 4
    candidate_masks = torch.zeros(
        (candidate_count, batch_size, sequence_length),
        device=eligible_mask.device,
        dtype=torch.bool,
    )
    candidate_valid = torch.zeros(
        (candidate_count, batch_size),
        device=eligible_mask.device,
        dtype=torch.bool,
    )
    proposal_scores = torch.full(
        (candidate_count, batch_size),
        -torch.inf,
        device=eligible_mask.device,
        dtype=confidence.dtype,
    )
    seed_anchors = torch.full(
        (candidate_count, batch_size),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    selected_positions = torch.full(
        (candidate_count, batch_size, 1 if torch.any(eligible_mask) else 0),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(generation_seed)

    for batch_index in range(batch_size):
        valid_positions = torch.where(eligible_mask[batch_index])[0]
        valid_count = valid_positions.numel()
        if valid_count == 0:
            continue
        random_offset = int(
            torch.randperm(
                valid_count,
                generator=local_generator,
                device="cpu",
            )[0].item()
        )
        valid_confidence = confidence[batch_index, valid_positions]
        low_confidence_offset = int(
            torch.argsort(valid_confidence, stable=True)[0].item()
        )
        anchors = (
            int(valid_positions[0].item()),
            int(valid_positions[valid_count // 2].item()),
            int(valid_positions[random_offset].item()),
            int(valid_positions[low_confidence_offset].item()),
        )
        for candidate_index, anchor in enumerate(anchors):
            candidate_masks[candidate_index, batch_index, anchor] = True
            candidate_valid[candidate_index, batch_index] = True
            seed_anchors[candidate_index, batch_index] = anchor
            selected_positions[candidate_index, batch_index, 0] = anchor
            proposal_scores[candidate_index, batch_index] = (
                -confidence[batch_index, anchor]
                if candidate_index == 3
                else 0.0
            )

    requested_k = torch.ones(
        batch_size,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    clipped_k = torch.minimum(
        requested_k,
        eligible_mask.sum(dim=-1, dtype=torch.long),
    )
    return CandidateBatch(
        candidate_masks=candidate_masks,
        names=("spaced_0", "spaced_1", "random", "high_entropy"),
        proposal_scores=proposal_scores,
        seed_anchors=seed_anchors,
        selected_positions=selected_positions,
        mean_within_set_dependency=torch.zeros_like(proposal_scores),
        candidate_valid=candidate_valid,
        eligible_mask=eligible_mask,
        requested_k=requested_k,
        clipped_k=clipped_k,
        metadata=(
            {"source": "legacy_spaced", "offset": 0},
            {"source": "legacy_spaced", "offset": 1},
            {"source": "legacy_random"},
            {
                "source": "legacy_low_confidence",
                "historical_name": "high_entropy",
                "uses_measured_entropy": False,
            },
        ),
        generation_seed=generation_seed,
        configuration={
            "requested_k": 1,
            "proposal": "current_mixed_compatibility",
            "candidate_order": (
                "spaced_0",
                "spaced_1",
                "random",
                "high_entropy",
            ),
            "random_rng_device": "cpu",
            "historical_high_entropy_definition": "lowest confidence",
        },
    )
