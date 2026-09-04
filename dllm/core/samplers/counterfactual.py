"""
Evaluate one-position counterfactual entropy and risk reduction.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_counterfactual_oracle.py" -v
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OnePositionCounterfactualOutput:
    """Scores and mappings for a flattened batch of one-position reveals."""

    batch_indices: torch.Tensor
    positions: torch.Tensor
    revealed_token_ids: torch.Tensor
    heldout_mask: torch.Tensor
    heldout_counts: torch.Tensor
    base_entropy_sum: torch.Tensor
    lookahead_entropy_sum: torch.Tensor
    entropy_drop_sum: torch.Tensor
    entropy_drop_per_heldout: torch.Tensor
    base_risk_sum: torch.Tensor
    lookahead_risk_sum: torch.Tensor
    risk_reduction_sum: torch.Tensor
    risk_reduction_per_heldout: torch.Tensor

    @property
    def candidate_count(self) -> int:
        """Return the number of flattened candidate reveals."""
        return self.positions.numel()


def _validate_logits(
    logits: torch.Tensor,
    *,
    name: str,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    vocabulary_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Validate logits used by the counterfactual oracle."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if logits.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, V].")
    if not torch.is_floating_point(logits):
        raise TypeError(f"{name} must have a floating-point dtype.")
    if batch_size is not None and logits.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}.")
    if sequence_length is not None and logits.shape[1] != sequence_length:
        raise ValueError(f"{name} sequence length must be {sequence_length}.")
    if vocabulary_size is not None and logits.shape[2] != vocabulary_size:
        raise ValueError(f"{name} vocabulary size must be {vocabulary_size}.")
    if logits.shape[-1] <= 0:
        raise ValueError(f"{name} must have a nonempty vocabulary axis.")
    if device is not None and logits.device != device:
        raise ValueError(f"{name} must be on {device}, got {logits.device}.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError(f"{name} must not contain NaN or positive infinity.")
    if (~torch.isneginf(logits)).sum(dim=-1).eq(0).any():
        raise ValueError(f"{name} has a position with no finite logits.")


def entropy_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Return float32 categorical entropy for logits shaped ``[..., V]``."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise ValueError("logits must have shape [..., V] with nonempty V.")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits must not contain NaN or positive infinity.")
    if (~torch.isneginf(logits)).sum(dim=-1).eq(0).any():
        raise ValueError("Each position must contain at least one finite logit.")

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    terms = torch.where(
        probabilities > 0,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=-1)


def decoding_risk_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Return float32 decoding risk ``1 - max softmax probability``."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise ValueError("logits must have shape [..., V] with nonempty V.")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits must not contain NaN or positive infinity.")
    if (~torch.isneginf(logits)).sum(dim=-1).eq(0).any():
        raise ValueError("Each position must contain at least one finite logit.")

    maximum_probability = F.softmax(logits.float(), dim=-1).amax(dim=-1)
    return 1.0 - maximum_probability


def _normalize_mask(
    mask: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Validate and convert one batch-by-sequence mask to bool."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}.")
    return mask.detach().to(device=device) != 0


def _extract_logits(model_output: object) -> torch.Tensor:
    """Extract logits from a Hugging Face-style output or raw tensor."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    logits = getattr(model_output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("The lookahead model must return a tensor or expose .logits.")
    return logits


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum values over an explicit mask without including other positions."""
    return torch.where(mask, values, torch.zeros_like(values)).sum(dim=-1)


def _normalize_sums(
    values: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Normalize sums by held-out count, defining empty-region scores as zero."""
    safe_counts = counts.clamp_min(1).to(dtype=values.dtype)
    normalized = values / safe_counts
    return torch.where(counts > 0, normalized, torch.zeros_like(normalized))


def _empty_output(
    *,
    sequence_length: int,
    device: torch.device,
) -> OnePositionCounterfactualOutput:
    """Build a defined result when there are no positions to evaluate."""
    empty_long = torch.empty(0, device=device, dtype=torch.long)
    empty_float = torch.empty(0, device=device, dtype=torch.float32)
    return OnePositionCounterfactualOutput(
        batch_indices=empty_long,
        positions=empty_long.clone(),
        revealed_token_ids=empty_long.clone(),
        heldout_mask=torch.empty(
            (0, sequence_length),
            device=device,
            dtype=torch.bool,
        ),
        heldout_counts=empty_long.clone(),
        base_entropy_sum=empty_float,
        lookahead_entropy_sum=empty_float.clone(),
        entropy_drop_sum=empty_float.clone(),
        entropy_drop_per_heldout=empty_float.clone(),
        base_risk_sum=empty_float.clone(),
        lookahead_risk_sum=empty_float.clone(),
        risk_reduction_sum=empty_float.clone(),
        risk_reduction_per_heldout=empty_float.clone(),
    )


@torch.no_grad()
def evaluate_one_position_counterfactuals(
    model: nn.Module,
    input_ids: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    masked_active_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    evaluation_mask: torch.Tensor | None = None,
    candidate_chunk_size: int | None = None,
) -> OnePositionCounterfactualOutput:
    """Score requested one-position reveals in ordered lookahead batches.

    ``masked_active_mask`` defines the current region M. ``evaluation_mask`` may
    select a subset of M as candidate anchors, but every candidate is evaluated
    against the full held-out region M without that candidate. A positive
    ``candidate_chunk_size`` limits peak lookahead-logit memory without changing
    candidate order or scores.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T].")
    if input_ids.dtype == torch.bool or torch.is_floating_point(input_ids):
        raise TypeError("input_ids must contain integer token IDs.")

    batch_size, sequence_length = input_ids.shape
    device = input_ids.device
    _validate_logits(
        base_logits,
        name="base_logits",
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )

    shape = (batch_size, sequence_length)
    active = _normalize_mask(
        masked_active_mask,
        name="masked_active_mask",
        shape=shape,
        device=device,
    )
    if attention_mask is None:
        valid = torch.ones(shape, device=device, dtype=torch.bool)
        normalized_attention_mask = None
    else:
        valid = _normalize_mask(
            attention_mask,
            name="attention_mask",
            shape=shape,
            device=device,
        )
        normalized_attention_mask = attention_mask.detach().to(device=device)
    if torch.any(active & ~valid):
        raise ValueError("masked_active_mask must be a subset of attention_mask.")

    if evaluation_mask is None:
        evaluate = active
    else:
        evaluate = _normalize_mask(
            evaluation_mask,
            name="evaluation_mask",
            shape=shape,
            device=device,
        )
        if torch.any(evaluate & ~active):
            raise ValueError("evaluation_mask must be a subset of masked_active_mask.")

    if candidate_chunk_size is not None and (
        isinstance(candidate_chunk_size, bool)
        or not isinstance(candidate_chunk_size, int)
        or candidate_chunk_size <= 0
    ):
        raise ValueError("candidate_chunk_size must be a positive integer or None.")

    batch_indices, positions = torch.nonzero(evaluate, as_tuple=True)
    if positions.numel() == 0:
        return _empty_output(sequence_length=sequence_length, device=device)

    candidate_count = positions.numel()
    candidate_rows = torch.arange(candidate_count, device=device)
    predicted_tokens = base_logits.argmax(dim=-1)
    revealed_token_ids = predicted_tokens[batch_indices, positions]

    heldout_mask = active.index_select(0, batch_indices).clone()
    heldout_mask[candidate_rows, positions] = False
    heldout_counts = heldout_mask.sum(dim=-1)

    base_entropy = entropy_per_token(base_logits).index_select(0, batch_indices)
    base_risk = decoding_risk_per_token(base_logits).index_select(0, batch_indices)
    chunk_size = candidate_chunk_size or candidate_count
    lookahead_entropy_chunks = []
    lookahead_risk_chunks = []
    for chunk_start in range(0, candidate_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, candidate_count)
        chunk_batch_indices = batch_indices[chunk_start:chunk_end]
        chunk_positions = positions[chunk_start:chunk_end]
        chunk_rows = torch.arange(
            chunk_end - chunk_start,
            device=device,
        )
        candidate_states = input_ids.index_select(0, chunk_batch_indices).clone()
        candidate_states[chunk_rows, chunk_positions] = revealed_token_ids[
            chunk_start:chunk_end
        ].to(dtype=input_ids.dtype)
        model_kwargs = {"input_ids": candidate_states}
        if normalized_attention_mask is not None:
            model_kwargs["attention_mask"] = (
                normalized_attention_mask.index_select(0, chunk_batch_indices)
            )
        lookahead_logits = _extract_logits(model(**model_kwargs))
        _validate_logits(
            lookahead_logits,
            name="lookahead_logits",
            batch_size=chunk_end - chunk_start,
            sequence_length=sequence_length,
            vocabulary_size=base_logits.shape[-1],
            device=device,
        )
        lookahead_entropy_chunks.append(entropy_per_token(lookahead_logits))
        lookahead_risk_chunks.append(decoding_risk_per_token(lookahead_logits))
        del lookahead_logits

    lookahead_entropy = torch.cat(lookahead_entropy_chunks, dim=0)
    lookahead_risk = torch.cat(lookahead_risk_chunks, dim=0)
    base_entropy_sum = _masked_sum(base_entropy, heldout_mask)
    lookahead_entropy_sum = _masked_sum(lookahead_entropy, heldout_mask)
    entropy_drop_sum = base_entropy_sum - lookahead_entropy_sum

    base_risk_sum = _masked_sum(base_risk, heldout_mask)
    lookahead_risk_sum = _masked_sum(lookahead_risk, heldout_mask)
    risk_reduction_sum = base_risk_sum - lookahead_risk_sum

    return OnePositionCounterfactualOutput(
        batch_indices=batch_indices,
        positions=positions,
        revealed_token_ids=revealed_token_ids,
        heldout_mask=heldout_mask,
        heldout_counts=heldout_counts,
        base_entropy_sum=base_entropy_sum,
        lookahead_entropy_sum=lookahead_entropy_sum,
        entropy_drop_sum=entropy_drop_sum,
        entropy_drop_per_heldout=_normalize_sums(
            entropy_drop_sum,
            heldout_counts,
        ),
        base_risk_sum=base_risk_sum,
        lookahead_risk_sum=lookahead_risk_sum,
        risk_reduction_sum=risk_reduction_sum,
        risk_reduction_per_heldout=_normalize_sums(
            risk_reduction_sum,
            heldout_counts,
        ),
    )
