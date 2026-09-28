"""Native-token canvas views and sparse spatial alignment for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass
from typing import Sequence

import torch


Span = tuple[float, float]


@dataclass(frozen=True)
class ModelCanvasView:
    """One model's native input and its generated-token spatial metadata."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    generation_start: int
    offsets: tuple[Span, ...]
    overlap_matrix: torch.Tensor

    @property
    def generation_length(self) -> int:
        return len(self.offsets)

    @property
    def generation_slice(self) -> slice:
        return slice(self.generation_start, self.generation_start + self.generation_length)


def _validate_offsets(offsets: Sequence[Span], name: str) -> None:
    for index, span in enumerate(offsets):
        if len(span) != 2:
            raise ValueError(f"{name}[{index}] must contain a start and end")
        start, end = float(span[0]), float(span[1])
        if not torch.isfinite(torch.tensor([start, end])).all():
            raise ValueError(f"{name}[{index}] must be finite")
        if start < 0 or end < start:
            raise ValueError(f"{name}[{index}] is not a valid non-negative span")


def build_canvas_overlap_matrix(
    offsets_aux: Sequence[Span],
    offsets_master: Sequence[Span],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build sparse auxiliary-to-master overlap fractions.

    Each non-empty auxiliary row sums to one when its span is fully covered by
    the master spans. Zero-length special-token spans have empty rows.
    """
    _validate_offsets(offsets_aux, "offsets_aux")
    _validate_offsets(offsets_master, "offsets_master")

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for aux_index, (aux_start, aux_end) in enumerate(offsets_aux):
        aux_start = float(aux_start)
        aux_end = float(aux_end)
        aux_length = aux_end - aux_start
        if aux_length == 0:
            continue
        for master_index, (master_start, master_end) in enumerate(offsets_master):
            overlap = max(
                0.0,
                min(aux_end, float(master_end))
                - max(aux_start, float(master_start)),
            )
            if overlap > 0:
                row_indices.append(aux_index)
                column_indices.append(master_index)
                values.append(overlap / aux_length)

    indices = torch.tensor(
        [row_indices, column_indices], dtype=torch.long, device=device
    )
    entries = torch.tensor(values, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        entries,
        size=(len(offsets_aux), len(offsets_master)),
        device=device,
        dtype=dtype,
    ).coalesce()


def spatial_warp_probabilities(
    probabilities_aux: torch.Tensor,
    overlap_matrix: torch.Tensor,
) -> torch.Tensor:
    """Warp auxiliary probabilities onto master positions with sparse Omega."""
    if probabilities_aux.ndim != 2:
        raise ValueError("probabilities_aux must have shape [N_aux, V_aux]")
    if overlap_matrix.ndim != 2 or overlap_matrix.shape[0] != probabilities_aux.shape[0]:
        raise ValueError("overlap_matrix must have shape [N_aux, N_master]")
    if not overlap_matrix.is_sparse:
        raise ValueError("overlap_matrix must be a sparse COO tensor")
    if (probabilities_aux < 0).any() or not torch.isfinite(probabilities_aux).all():
        raise ValueError("probabilities_aux must be finite and non-negative")

    overlap = overlap_matrix.to(
        device=probabilities_aux.device, dtype=probabilities_aux.dtype
    )
    return torch.sparse.mm(overlap.transpose(0, 1), probabilities_aux)


def _encode_without_special_tokens(tokenizer, text: str) -> tuple[list[int], list[Span]]:
    if not text:
        return [], []
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    return [int(token_id) for token_id in input_ids], [tuple(map(float, x)) for x in offsets]


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _prefix_boundaries(tokenizer, token_ids: Sequence[int]) -> list[int]:
    boundaries = [0]
    for end in range(1, len(token_ids) + 1):
        boundaries.append(len(_decode(tokenizer, token_ids[:end])))
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1])
    return boundaries


def _character_to_logical(
    value: float,
    boundaries: Sequence[int],
    logical_start: int,
) -> float:
    if value <= boundaries[0]:
        return float(logical_start)
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if value <= end or index == len(boundaries) - 2:
            if end == start:
                return float(logical_start + index + 1)
            fraction = min(1.0, max(0.0, (value - start) / (end - start)))
            return logical_start + index + fraction
    return float(logical_start + len(boundaries) - 1)


def build_model_canvas_view(
    native_prompt_ids: torch.Tensor | Sequence[int],
    master_generated_ids: torch.Tensor | Sequence[int],
    *,
    master_tokenizer,
    model_tokenizer,
    master_mask_token_id: int,
    model_mask_token_id: int,
) -> ModelCanvasView:
    """Derive a native model view without rendering masks as placeholder text.

    Committed master runs are decoded and retokenized by the target tokenizer.
    Every unresolved logical slot is represented by exactly one native mask ID.
    """
    prompt_ids = torch.as_tensor(native_prompt_ids, dtype=torch.long).flatten().tolist()
    generated_ids = (
        torch.as_tensor(master_generated_ids, dtype=torch.long).flatten().tolist()
    )
    if model_mask_token_id is None:
        raise ValueError("model_tokenizer must define mask_token_id")

    native_generated_ids: list[int] = []
    native_offsets: list[Span] = []
    index = 0
    while index < len(generated_ids):
        if generated_ids[index] == master_mask_token_id:
            native_generated_ids.append(int(model_mask_token_id))
            native_offsets.append((float(index), float(index + 1)))
            index += 1
            continue

        run_start = index
        while index < len(generated_ids) and generated_ids[index] != master_mask_token_id:
            index += 1
        run_ids = generated_ids[run_start:index]
        run_text = _decode(master_tokenizer, run_ids)
        encoded_ids, character_offsets = _encode_without_special_tokens(
            model_tokenizer, run_text
        )
        if not encoded_ids:
            fallback_id = getattr(model_tokenizer, "unk_token_id", None)
            if fallback_id is None:
                raise ValueError("committed text produced no auxiliary token IDs")
            encoded_ids = [int(fallback_id)]
            character_offsets = [(0.0, float(max(1, len(run_text))))]

        boundaries = _prefix_boundaries(master_tokenizer, run_ids)
        native_generated_ids.extend(encoded_ids)
        for start, end in character_offsets:
            native_offsets.append(
                (
                    _character_to_logical(start, boundaries, run_start),
                    _character_to_logical(end, boundaries, run_start),
                )
            )

    all_ids = torch.tensor(prompt_ids + native_generated_ids, dtype=torch.long)
    attention_mask = torch.ones_like(all_ids)
    master_offsets = tuple((float(i), float(i + 1)) for i in range(len(generated_ids)))
    overlap = build_canvas_overlap_matrix(native_offsets, master_offsets)
    return ModelCanvasView(
        input_ids=all_ids,
        attention_mask=attention_mask,
        generation_start=len(prompt_ids),
        offsets=tuple(native_offsets),
        overlap_matrix=overlap,
    )
