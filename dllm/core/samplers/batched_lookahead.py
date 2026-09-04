"""Expand and score candidate actions with shared batched lookahead calls.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_batched_lookahead.py" -v
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from dllm.core.samplers.candidates import CandidateBatch
from dllm.core.samplers.counterfactual import (
    decoding_risk_per_token,
    entropy_per_token,
)


LookaheadMetric = Literal["entropy_drop", "risk_reduction"]
SUPPORTED_LOOKAHEAD_METRICS = ("entropy_drop", "risk_reduction")


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor contains non-boolean integer values."""
    return tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _normalize_bool_mask(
    mask: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Validate a batch-by-sequence mask and return a detached bool view."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if mask.device != device:
        raise ValueError(f"{name} must be on {device}, got {mask.device}.")
    return mask.detach() != 0


@dataclass(frozen=True)
class CandidateStateExpansion:
    """Candidate-major successor states and their reversible flat mapping.

    Flat row ``n * B + b`` always corresponds to candidate ``n`` and batch row
    ``b``. Invalid candidate/example pairs remain no-op copies of the input and
    receive empty held-out masks, so they are structurally safe even before the
    evaluator assigns them a score of negative infinity.
    """

    candidate_states: torch.Tensor
    flat_states: torch.Tensor
    flat_attention_mask: torch.Tensor | None
    heldout_masks: torch.Tensor
    candidate_indices: torch.Tensor
    batch_indices: torch.Tensor
    candidate_valid: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the candidate-major representation and inverse indexes."""
        states = self.candidate_states
        if not isinstance(states, torch.Tensor) or states.ndim != 3:
            raise ValueError("candidate_states must have shape [N, B, T].")
        candidate_count, batch_size, sequence_length = states.shape
        flat_count = candidate_count * batch_size
        if tuple(self.flat_states.shape) != (flat_count, sequence_length):
            raise ValueError("flat_states must have shape [N * B, T].")
        if self.flat_states.device != states.device:
            raise ValueError("flat_states and candidate_states must share a device.")
        if self.flat_states.dtype != states.dtype:
            raise TypeError("flat_states and candidate_states must share a dtype.")
        if tuple(self.heldout_masks.shape) != tuple(states.shape):
            raise ValueError("heldout_masks must have shape [N, B, T].")
        if self.heldout_masks.dtype != torch.bool:
            raise TypeError("heldout_masks must be boolean.")
        if tuple(self.candidate_valid.shape) != (candidate_count, batch_size):
            raise ValueError("candidate_valid must have shape [N, B].")
        if self.candidate_valid.dtype != torch.bool:
            raise TypeError("candidate_valid must be boolean.")
        if torch.any(
            self.heldout_masks & ~self.candidate_valid.unsqueeze(-1)
        ):
            raise ValueError("invalid candidates must have empty held-out masks.")

        for name, indexes in (
            ("candidate_indices", self.candidate_indices),
            ("batch_indices", self.batch_indices),
        ):
            if not isinstance(indexes, torch.Tensor):
                raise TypeError(f"{name} must be a tensor.")
            if tuple(indexes.shape) != (flat_count,):
                raise ValueError(f"{name} must have shape [N * B].")
            if indexes.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long.")
            if indexes.device != states.device:
                raise ValueError(f"{name} must share the state device.")

        expected_candidates = torch.arange(
            candidate_count,
            device=states.device,
        ).repeat_interleave(batch_size)
        expected_batches = torch.arange(
            batch_size,
            device=states.device,
        ).repeat(candidate_count)
        if not torch.equal(self.candidate_indices, expected_candidates):
            raise ValueError("candidate_indices do not use candidate-major order.")
        if not torch.equal(self.batch_indices, expected_batches):
            raise ValueError("batch_indices do not use candidate-major order.")

        if self.flat_attention_mask is not None:
            if tuple(self.flat_attention_mask.shape) != (
                flat_count,
                sequence_length,
            ):
                raise ValueError(
                    "flat_attention_mask must have shape [N * B, T]."
                )
            if self.flat_attention_mask.device != states.device:
                raise ValueError("flat_attention_mask must share the state device.")

    @property
    def candidate_count(self) -> int:
        """Return N."""
        return self.candidate_states.shape[0]

    @property
    def batch_size(self) -> int:
        """Return B."""
        return self.candidate_states.shape[1]

    @property
    def sequence_length(self) -> int:
        """Return T."""
        return self.candidate_states.shape[2]

    def flat_row(self, candidate_index: int, batch_index: int) -> int:
        """Map one ``(candidate, batch)`` pair to its candidate-major row."""
        if not 0 <= candidate_index < self.candidate_count:
            raise IndexError("candidate_index is outside the expansion.")
        if not 0 <= batch_index < self.batch_size:
            raise IndexError("batch_index is outside the expansion.")
        return candidate_index * self.batch_size + batch_index

    def unflatten(self, values: torch.Tensor) -> torch.Tensor:
        """Restore a tensor beginning with ``N * B`` to ``[N, B, ...]``."""
        expected_rows = self.candidate_count * self.batch_size
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if values.ndim == 0 or values.shape[0] != expected_rows:
            raise ValueError("values must begin with the N * B flat row count.")
        return values.reshape(
            self.candidate_count,
            self.batch_size,
            *values.shape[1:],
        )


@dataclass(frozen=True)
class BatchedLookaheadOutput:
    """Scores and selected actions for one held-out uncertainty metric."""

    metric: LookaheadMetric
    scores: torch.Tensor
    base_metric_sums: torch.Tensor
    lookahead_metric_sums: torch.Tensor
    heldout_counts: torch.Tensor
    best_index: torch.Tensor
    best_score: torch.Tensor
    best_mask: torch.Tensor
    best_names: tuple[str | None, ...]
    best_metadata: tuple[Mapping[str, object] | None, ...]
    model_calls: int
    candidate_chunk_size: int | None
    flattening_order: str = "candidate_major"

    def __post_init__(self) -> None:
        """Validate aligned score, selection, and accounting fields."""
        if self.metric not in SUPPORTED_LOOKAHEAD_METRICS:
            raise ValueError(f"Unsupported lookahead metric: {self.metric!r}.")
        if not isinstance(self.scores, torch.Tensor) or self.scores.ndim != 2:
            raise ValueError("scores must have shape [N, B].")
        candidate_count, batch_size = self.scores.shape
        for name, tensor in (
            ("base_metric_sums", self.base_metric_sums),
            ("lookahead_metric_sums", self.lookahead_metric_sums),
            ("heldout_counts", self.heldout_counts),
        ):
            if tuple(tensor.shape) != (candidate_count, batch_size):
                raise ValueError(f"{name} must have shape [N, B].")
        if tuple(self.best_index.shape) != (batch_size,):
            raise ValueError("best_index must have shape [B].")
        if tuple(self.best_score.shape) != (batch_size,):
            raise ValueError("best_score must have shape [B].")
        if self.best_mask.ndim != 2 or self.best_mask.shape[0] != batch_size:
            raise ValueError("best_mask must have shape [B, T].")
        if self.best_mask.dtype != torch.bool:
            raise TypeError("best_mask must be boolean.")
        if len(self.best_names) != batch_size:
            raise ValueError("best_names must have length B.")
        if len(self.best_metadata) != batch_size:
            raise ValueError("best_metadata must have length B.")
        if self.model_calls < 0:
            raise ValueError("model_calls must be nonnegative.")
        if self.candidate_chunk_size is not None and self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive or None.")
        if self.flattening_order != "candidate_major":
            raise ValueError("Only candidate-major flattening is supported.")


def expand_candidate_states(
    input_ids: torch.Tensor,
    predicted_token_ids: torch.Tensor,
    candidates: CandidateBatch,
    *,
    attention_mask: torch.Tensor | None = None,
    masked_active_mask: torch.Tensor | None = None,
) -> CandidateStateExpansion:
    """Materialize all candidate successors without mutating source tensors."""
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T].")
    if not _is_integer_tensor(input_ids):
        raise TypeError("input_ids must contain integer token IDs.")
    if (
        not isinstance(predicted_token_ids, torch.Tensor)
        or tuple(predicted_token_ids.shape) != tuple(input_ids.shape)
    ):
        raise ValueError("predicted_token_ids must match input_ids shape [B, T].")
    if not _is_integer_tensor(predicted_token_ids):
        raise TypeError("predicted_token_ids must contain integer token IDs.")
    if predicted_token_ids.device != input_ids.device:
        raise ValueError("predicted_token_ids must share the input_ids device.")
    if not isinstance(candidates, CandidateBatch):
        raise TypeError("candidates must be a CandidateBatch.")

    batch_size, sequence_length = input_ids.shape
    candidate_count, candidate_batch_size, candidate_length = candidates.shape
    if (candidate_batch_size, candidate_length) != (batch_size, sequence_length):
        raise ValueError("CandidateBatch [B, T] must match input_ids.")
    if candidates.candidate_masks.device != input_ids.device:
        raise ValueError("CandidateBatch must share the input_ids device.")

    shape = (batch_size, sequence_length)
    if masked_active_mask is None:
        active = candidates.eligible_mask
    else:
        active = _normalize_bool_mask(
            masked_active_mask,
            name="masked_active_mask",
            shape=shape,
            device=input_ids.device,
        )
    if torch.any(candidates.candidate_masks & ~active.unsqueeze(0)):
        raise ValueError("candidate masks must be a subset of masked_active_mask.")

    flat_attention_mask = None
    if attention_mask is not None:
        valid = _normalize_bool_mask(
            attention_mask,
            name="attention_mask",
            shape=shape,
            device=input_ids.device,
        )
        if torch.any(active & ~valid):
            raise ValueError("masked_active_mask must be a subset of attention_mask.")
        flat_attention_mask = (
            attention_mask.detach()
            .unsqueeze(0)
            .expand(candidate_count, -1, -1)
            .reshape(candidate_count * batch_size, sequence_length)
            .clone()
        )

    candidate_states = (
        input_ids.detach()
        .unsqueeze(0)
        .expand(candidate_count, -1, -1)
        .clone()
    )
    replacement_tokens = predicted_token_ids.to(dtype=input_ids.dtype)
    repeated_replacements = replacement_tokens.unsqueeze(0).expand(
        candidate_count,
        -1,
        -1,
    )
    candidate_states = torch.where(
        candidates.candidate_masks,
        repeated_replacements,
        candidate_states,
    )
    heldout_masks = (
        active.unsqueeze(0)
        & ~candidates.candidate_masks
        & candidates.candidate_valid.unsqueeze(-1)
    )
    flat_states = candidate_states.reshape(
        candidate_count * batch_size,
        sequence_length,
    )
    candidate_indices = torch.arange(
        candidate_count,
        device=input_ids.device,
    ).repeat_interleave(batch_size)
    batch_indices = torch.arange(
        batch_size,
        device=input_ids.device,
    ).repeat(candidate_count)
    return CandidateStateExpansion(
        candidate_states=candidate_states,
        flat_states=flat_states,
        flat_attention_mask=flat_attention_mask,
        heldout_masks=heldout_masks,
        candidate_indices=candidate_indices,
        batch_indices=batch_indices,
        candidate_valid=candidates.candidate_valid,
    )


def candidate_batch_from_mask_mapping(
    candidate_masks: Mapping[str, torch.Tensor],
    *,
    eligible_mask: torch.Tensor,
) -> CandidateBatch:
    """Adapt an ordered legacy name-to-mask mapping to ``CandidateBatch``."""
    if not isinstance(candidate_masks, Mapping):
        raise TypeError("candidate_masks must be a mapping.")
    if not isinstance(eligible_mask, torch.Tensor) or eligible_mask.ndim != 2:
        raise ValueError("eligible_mask must have shape [B, T].")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean.")
    names = tuple(candidate_masks)
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate names must be nonempty strings.")

    batch_size, sequence_length = eligible_mask.shape
    masks = []
    for name in names:
        mask = candidate_masks[name]
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"candidate {name!r} must be a tensor.")
        if tuple(mask.shape) != (batch_size, sequence_length):
            raise ValueError(f"candidate {name!r} must have shape [B, T].")
        if mask.dtype != torch.bool:
            raise TypeError(f"candidate {name!r} must be boolean.")
        if mask.device != eligible_mask.device:
            raise ValueError("all candidate masks must share the eligible device.")
        masks.append(mask)

    candidate_count = len(masks)
    if masks:
        stacked_masks = torch.stack(masks, dim=0)
    else:
        stacked_masks = torch.empty(
            (0, batch_size, sequence_length),
            device=eligible_mask.device,
            dtype=torch.bool,
        )
    if torch.any(stacked_masks & ~eligible_mask.unsqueeze(0)):
        raise ValueError("legacy candidates must stay inside eligible_mask.")

    action_sizes = stacked_masks.sum(dim=-1, dtype=torch.long)
    candidate_valid = action_sizes > 0
    requested_k = torch.zeros(
        batch_size,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    for batch_index in range(batch_size):
        valid_sizes = action_sizes[:, batch_index][candidate_valid[:, batch_index]]
        if valid_sizes.numel():
            if torch.any(valid_sizes != valid_sizes[0]):
                raise ValueError(
                    "all valid legacy candidates must use the same k per batch row."
                )
            requested_k[batch_index] = valid_sizes[0]
    eligible_counts = eligible_mask.sum(dim=-1, dtype=torch.long)
    clipped_k = torch.minimum(requested_k, eligible_counts)
    storage_width = int(clipped_k.max().item()) if batch_size else 0
    selected_positions = torch.full(
        (candidate_count, batch_size, storage_width),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    seed_anchors = torch.full(
        (candidate_count, batch_size),
        -1,
        device=eligible_mask.device,
        dtype=torch.long,
    )
    for candidate_index in range(candidate_count):
        for batch_index in range(batch_size):
            if not bool(candidate_valid[candidate_index, batch_index]):
                continue
            positions = torch.where(stacked_masks[candidate_index, batch_index])[0]
            selected_positions[
                candidate_index,
                batch_index,
                : positions.numel(),
            ] = positions
            seed_anchors[candidate_index, batch_index] = positions[0]

    proposal_scores = torch.where(
        candidate_valid,
        torch.zeros(
            (candidate_count, batch_size),
            device=eligible_mask.device,
            dtype=torch.float32,
        ),
        torch.full(
            (candidate_count, batch_size),
            -torch.inf,
            device=eligible_mask.device,
            dtype=torch.float32,
        ),
    )
    return CandidateBatch(
        candidate_masks=stacked_masks,
        names=names,
        proposal_scores=proposal_scores,
        seed_anchors=seed_anchors,
        selected_positions=selected_positions,
        mean_within_set_dependency=torch.zeros_like(proposal_scores),
        candidate_valid=candidate_valid,
        eligible_mask=eligible_mask,
        requested_k=requested_k,
        clipped_k=clipped_k,
        metadata=tuple(
            {"source": "legacy_mask_mapping", "original_name": name}
            for name in names
        ),
        generation_seed=None,
        configuration={
            "adapter": "legacy_mask_mapping",
            "candidate_count": candidate_count,
        },
    )


def _extract_logits(model_output: object) -> torch.Tensor:
    """Extract logits from a raw tensor or Hugging Face-style model output."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    logits = getattr(model_output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("The lookahead model must return a tensor or expose .logits.")
    return logits


def _validate_base_metric_map(
    base_metric_map: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> None:
    """Validate the already-computed entropy or risk map."""
    if not isinstance(base_metric_map, torch.Tensor):
        raise TypeError("base_metric_map must be a tensor.")
    if tuple(base_metric_map.shape) != shape:
        raise ValueError(f"base_metric_map must have shape {shape}.")
    if base_metric_map.device != device:
        raise ValueError("base_metric_map must share the input device.")
    if not base_metric_map.is_floating_point():
        raise TypeError("base_metric_map must be floating point.")
    if not torch.isfinite(base_metric_map).all():
        raise ValueError("base_metric_map must contain only finite values.")


def _metric_per_token(
    logits: torch.Tensor,
    metric: LookaheadMetric,
) -> torch.Tensor:
    """Dispatch the metric-specific computation after shared model execution."""
    if metric == "entropy_drop":
        return entropy_per_token(logits)
    if metric == "risk_reduction":
        return decoding_risk_per_token(logits)
    raise ValueError(f"Unsupported lookahead metric: {metric!r}.")


def _gather_selection(
    candidates: CandidateBatch,
    best_index: torch.Tensor,
) -> tuple[
    torch.Tensor,
    tuple[str | None, ...],
    tuple[Mapping[str, object] | None, ...],
]:
    """Gather masks, names, and metadata with ``-1`` as the no-action index."""
    batch_size, sequence_length = candidates.eligible_mask.shape
    best_mask = torch.zeros(
        (batch_size, sequence_length),
        device=candidates.candidate_masks.device,
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
            candidate_index,
            batch_index,
        ]
        best_names.append(candidates.names[candidate_index])
        best_metadata.append(candidates.metadata[candidate_index])
    return best_mask, tuple(best_names), tuple(best_metadata)


@torch.no_grad()
def evaluate_batched_lookahead(
    model: nn.Module,
    input_ids: torch.Tensor,
    predicted_token_ids: torch.Tensor,
    candidates: CandidateBatch,
    *,
    base_metric_map: torch.Tensor,
    metric: LookaheadMetric,
    attention_mask: torch.Tensor | None = None,
    masked_active_mask: torch.Tensor | None = None,
    candidate_chunk_size: int | None = None,
) -> BatchedLookaheadOutput:
    """Score candidate actions in candidate chunks and select stable maxima.

    Chunking is over N: a chunk containing C candidates calls the model once on
    ``C * B`` rows. Scores are accumulated in global candidate order, and the
    online selector updates only on a strict improvement. Therefore equal scores
    choose the earliest candidate exactly like the old sequential loop.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if metric not in SUPPORTED_LOOKAHEAD_METRICS:
        raise ValueError(f"Unsupported lookahead metric: {metric!r}.")
    if candidate_chunk_size is not None and (
        isinstance(candidate_chunk_size, bool)
        or not isinstance(candidate_chunk_size, int)
        or candidate_chunk_size <= 0
    ):
        raise ValueError("candidate_chunk_size must be a positive integer or None.")

    expansion = expand_candidate_states(
        input_ids,
        predicted_token_ids,
        candidates,
        attention_mask=attention_mask,
        masked_active_mask=masked_active_mask,
    )
    batch_size, sequence_length = input_ids.shape
    candidate_count = candidates.shape[0]
    _validate_base_metric_map(
        base_metric_map,
        shape=(batch_size, sequence_length),
        device=input_ids.device,
    )

    scores = torch.full(
        (candidate_count, batch_size),
        -torch.inf,
        device=input_ids.device,
        dtype=torch.float32,
    )
    base_sums = torch.zeros_like(scores)
    lookahead_sums = torch.zeros_like(scores)
    heldout_counts = expansion.heldout_masks.sum(dim=-1, dtype=torch.long)
    best_score = torch.full(
        (batch_size,),
        -torch.inf,
        device=input_ids.device,
        dtype=torch.float32,
    )
    best_index = torch.full(
        (batch_size,),
        -1,
        device=input_ids.device,
        dtype=torch.long,
    )

    chunk_size = candidate_chunk_size or candidate_count
    model_calls = 0
    if candidate_count:
        for start in range(0, candidate_count, chunk_size):
            end = min(start + chunk_size, candidate_count)
            flat_start = start * batch_size
            flat_end = end * batch_size
            model_kwargs: dict[str, torch.Tensor] = {
                "input_ids": expansion.flat_states[flat_start:flat_end],
            }
            if expansion.flat_attention_mask is not None:
                model_kwargs["attention_mask"] = expansion.flat_attention_mask[
                    flat_start:flat_end
                ]
            lookahead_logits = _extract_logits(model(**model_kwargs))
            expected_shape = (
                (end - start) * batch_size,
                sequence_length,
            )
            if not lookahead_logits.is_floating_point():
                raise TypeError("lookahead logits must be floating point.")
            if lookahead_logits.ndim != 3 or tuple(lookahead_logits.shape[:2]) != (
                expected_shape
            ):
                raise ValueError(
                    "lookahead logits must have shape [candidate_chunk * B, T, V]."
                )
            if lookahead_logits.device != input_ids.device:
                raise ValueError("lookahead logits must share the input device.")

            chunk_metric = _metric_per_token(lookahead_logits, metric).reshape(
                end - start,
                batch_size,
                sequence_length,
            )
            chunk_heldout = expansion.heldout_masks[start:end]
            chunk_base = torch.where(
                chunk_heldout,
                base_metric_map.float().unsqueeze(0),
                torch.zeros_like(chunk_metric),
            ).sum(dim=-1)
            chunk_lookahead = torch.where(
                chunk_heldout,
                chunk_metric,
                torch.zeros_like(chunk_metric),
            ).sum(dim=-1)
            chunk_scores = chunk_base - chunk_lookahead
            chunk_valid = candidates.candidate_valid[start:end]
            chunk_scores = torch.where(
                chunk_valid,
                chunk_scores,
                torch.full_like(chunk_scores, -torch.inf),
            )
            base_sums[start:end] = chunk_base
            lookahead_sums[start:end] = chunk_lookahead
            scores[start:end] = chunk_scores
            model_calls += 1

            for local_index in range(end - start):
                global_index = start + local_index
                improved = (
                    chunk_valid[local_index]
                    & (chunk_scores[local_index] > best_score)
                )
                best_score = torch.where(
                    improved,
                    chunk_scores[local_index],
                    best_score,
                )
                best_index = torch.where(
                    improved,
                    torch.full_like(best_index, global_index),
                    best_index,
                )

            del lookahead_logits, chunk_metric

    best_mask, best_names, best_metadata = _gather_selection(
        candidates,
        best_index,
    )
    return BatchedLookaheadOutput(
        metric=metric,
        scores=scores,
        base_metric_sums=base_sums,
        lookahead_metric_sums=lookahead_sums,
        heldout_counts=heldout_counts,
        best_index=best_index,
        best_score=best_score,
        best_mask=best_mask,
        best_names=best_names,
        best_metadata=best_metadata,
        model_calls=model_calls,
        candidate_chunk_size=candidate_chunk_size,
    )
