"""Run a CTCA sandbox with real LLaDA and Dream tokenizers.

From the repository root:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/sandbox/ctca_real_tokenizers_sandbox.py

This script intentionally avoids imports from the project. It uses real
Hugging Face tokenizers for canvas construction, then uses deterministic fake
embedding vectors and fake distributions to inspect CTCA's spatial alignment,
sparse vocab projection, and probability fusion without loading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


Span = tuple[float, float]


DEFAULT_LLADA = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_DREAM = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_LLADA_MASK_ID = 126336
DEFAULT_QUESTION = (
    "Question: Mia has 9 apples and gives away 3. How many remain?"
)


@dataclass(frozen=True)
class CanvasView:
    """A model-native view of the master generation canvas."""

    input_ids: torch.Tensor
    generation_start: int
    offsets: tuple[Span, ...]
    overlap_matrix: torch.Tensor

    @property
    def generation_slice(self) -> slice:
        return slice(self.generation_start, self.generation_start + len(self.offsets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llada-tokenizer", default=DEFAULT_LLADA)
    parser.add_argument("--dream-tokenizer", default=DEFAULT_DREAM)
    parser.add_argument(
        "--master",
        choices=("llada", "dream"),
        default="dream",
        help="Tokenizer that owns the master canvas.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--projection-temperature", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--llada-mask-id",
        type=int,
        default=DEFAULT_LLADA_MASK_ID,
        help="Fallback mask ID because the LLaDA tokenizer does not expose mask_token_id.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_tokenizer(model_id: str, *, local_files_only: bool):
    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def mask_token_id(tokenizer, label: str, *, fallback: int | None = None) -> int:
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    candidates = ("[MASK]", "<mask>", "<|mask|>", "<MASK>", "<|mdm_mask|>")
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    if fallback is not None and 0 <= fallback < len(tokenizer):
        return int(fallback)
    raise ValueError(f"{label} tokenizer does not expose a usable mask token")


def decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(map(int, token_ids)),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def token_label(tokenizer, token_id: int, mask_id: int) -> str:
    token_id = int(token_id)
    if token_id == mask_id:
        return "[MASK]"
    token = tokenizer.convert_ids_to_tokens(token_id)
    text = decode(tokenizer, [token_id])
    if text and text != token:
        return f"{token!r}/{text!r}"
    return repr(token)


def encode(tokenizer, text: str) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    ]


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[Span]]:
    if not text:
        return [], []
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(text, add_special_tokens=False)
        offsets = None

    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        if offsets is not None:
            offsets = offsets[0]
    token_ids = [int(token_id) for token_id in input_ids]
    if offsets is not None:
        return token_ids, [tuple(map(float, span)) for span in offsets]
    return token_ids, slow_tokenizer_offsets(tokenizer, token_ids, text)


def slow_tokenizer_offsets(tokenizer, token_ids: Sequence[int], text: str) -> list[Span]:
    boundaries = [0]
    for end in range(1, len(token_ids) + 1):
        decoded_prefix = decode(tokenizer, token_ids[:end])
        common_length = 0
        for actual, candidate in zip(text, decoded_prefix):
            if actual != candidate:
                break
            common_length += 1
        boundaries.append(max(boundaries[-1], common_length))
    if decode(tokenizer, token_ids) != text:
        raise ValueError("slow tokenizer must round-trip text to derive offsets")
    boundaries[-1] = len(text)
    return [(float(start), float(end)) for start, end in zip(boundaries, boundaries[1:])]


def prefix_boundaries(tokenizer, token_ids: Sequence[int]) -> list[int]:
    boundaries = [0]
    for end in range(1, len(token_ids) + 1):
        boundaries.append(len(decode(tokenizer, token_ids[:end])))
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1])
    return boundaries


def character_to_logical(
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
            return float(logical_start + index + fraction)
    return float(logical_start + len(boundaries) - 1)


def build_overlap_matrix(
    offsets_aux: Sequence[Span],
    offsets_master: Sequence[Span],
) -> torch.Tensor:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for aux_index, (aux_start, aux_end) in enumerate(offsets_aux):
        aux_length = aux_end - aux_start
        if aux_length == 0:
            continue
        for master_index, (master_start, master_end) in enumerate(offsets_master):
            overlap = max(0.0, min(aux_end, master_end) - max(aux_start, master_start))
            if overlap > 0:
                row_indices.append(aux_index)
                column_indices.append(master_index)
                values.append(overlap / aux_length)
    indices = torch.tensor([row_indices, column_indices], dtype=torch.long)
    entries = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        entries,
        size=(len(offsets_aux), len(offsets_master)),
    ).coalesce()


def build_model_canvas_view(
    native_prompt_ids: Sequence[int],
    master_generated_ids: Sequence[int],
    *,
    master_tokenizer,
    model_tokenizer,
    master_mask_id: int,
    model_mask_id: int,
) -> CanvasView:
    native_generated_ids: list[int] = []
    native_offsets: list[Span] = []
    index = 0
    while index < len(master_generated_ids):
        if master_generated_ids[index] == master_mask_id:
            native_generated_ids.append(model_mask_id)
            native_offsets.append((float(index), float(index + 1)))
            index += 1
            continue

        run_start = index
        while index < len(master_generated_ids) and master_generated_ids[index] != master_mask_id:
            index += 1
        run_ids = list(map(int, master_generated_ids[run_start:index]))
        run_text = decode(master_tokenizer, run_ids)
        encoded_ids, character_offsets = encode_with_offsets(model_tokenizer, run_text)
        boundaries = prefix_boundaries(master_tokenizer, run_ids)
        native_generated_ids.extend(encoded_ids)
        for start, end in character_offsets:
            native_offsets.append(
                (
                    character_to_logical(start, boundaries, run_start),
                    character_to_logical(end, boundaries, run_start),
                )
            )

    master_offsets = tuple((float(i), float(i + 1)) for i in range(len(master_generated_ids)))
    return CanvasView(
        input_ids=torch.tensor(list(native_prompt_ids) + native_generated_ids),
        generation_start=len(native_prompt_ids),
        offsets=tuple(native_offsets),
        overlap_matrix=build_overlap_matrix(native_offsets, master_offsets),
    )


def spatial_warp_probabilities(probabilities_aux: torch.Tensor, overlap_matrix: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(overlap_matrix.transpose(0, 1), probabilities_aux)


def stable_vector(token_text: str, dim: int = 16) -> torch.Tensor:
    digest = hashlib.sha256(token_text.encode("utf-8")).digest()
    values = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(dim)]
    return torch.tensor(values, dtype=torch.float32)


def semantic_vector(decoded_text: str, dim: int = 16) -> torch.Tensor:
    canonical = decoded_text.strip().lower()
    groups = {
        "answer": 0,
        "ans": 0,
        "wer": 0,
        "is": 1,
        "5": 2,
        "five": 2,
        "6": 3,
        "six": 3,
        "7": 4,
        "seven": 4,
        "apple": 5,
        "apples": 5,
        "s": 5,
        ".": 6,
    }
    vector = 0.15 * stable_vector(decoded_text, dim=dim)
    if canonical in groups:
        vector[groups[canonical]] += 2.0
    return vector


def embeddings_for_token_ids(tokenizer, token_ids: Sequence[int]) -> torch.Tensor:
    rows = []
    for token_id in token_ids:
        rows.append(semantic_vector(decode(tokenizer, [int(token_id)])))
    return torch.stack(rows)


def build_sparse_vocab_projection_for_active_ids(
    auxiliary_tokenizer,
    master_tokenizer,
    active_aux_ids: Sequence[int],
    candidate_master_ids: Sequence[int],
    *,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aux_embeddings = F.normalize(
        embeddings_for_token_ids(auxiliary_tokenizer, active_aux_ids),
        dim=-1,
    )
    master_embeddings = F.normalize(
        embeddings_for_token_ids(master_tokenizer, candidate_master_ids),
        dim=-1,
    )
    similarities = aux_embeddings @ master_embeddings.transpose(0, 1)
    k = min(top_k, len(candidate_master_ids))
    top_values, local_indices = torch.topk(similarities, k=k, dim=-1)
    top_weights = torch.softmax(top_values / temperature, dim=-1)
    candidate_ids = torch.tensor(list(map(int, candidate_master_ids)), dtype=torch.long)
    top_master_ids = candidate_ids[local_indices]
    return similarities, top_master_ids, top_weights


def make_sparse_distribution(vocab_size: int, weighted_ids: dict[int, float]) -> torch.Tensor:
    probabilities = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id, value in weighted_ids.items():
        probabilities[int(token_id)] = float(value)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("distribution must contain positive mass")
    return probabilities / total


def project_sparse_vocab(
    spatial_probabilities: torch.Tensor,
    active_aux_ids: Sequence[int],
    top_master_ids: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    master_vocab_size: int,
) -> torch.Tensor:
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        master_vocab_size,
        dtype=torch.float32,
    )
    for active_index, aux_id in enumerate(active_aux_ids):
        mass = spatial_probabilities[:, int(aux_id)].unsqueeze(-1)
        projected.scatter_add_(
            1,
            top_master_ids[active_index].unsqueeze(0).expand(spatial_probabilities.shape[0], -1),
            mass * top_weights[active_index].unsqueeze(0),
        )
    row_mass = projected.sum(dim=-1, keepdim=True)
    return torch.where(row_mass > 0, projected / row_mass.clamp_min(1e-9), projected)


def top_tokens(tokenizer, probabilities: torch.Tensor, *, mask_id: int, k: int = 5) -> list[tuple[str, float]]:
    values, token_ids = torch.topk(probabilities.float(), k=min(k, probabilities.shape[-1]))
    return [
        (token_label(tokenizer, int(token_id), mask_id), round(float(value), 4))
        for value, token_id in zip(values, token_ids)
        if float(value) > 0
    ]


def print_token_sequence(title: str, tokenizer, token_ids: Sequence[int], mask_id: int) -> None:
    print(title)
    for index, token_id in enumerate(token_ids):
        print(f"  {index:02d}: id={int(token_id):6d} {token_label(tokenizer, int(token_id), mask_id)}")


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=4, sci_mode=False)

    llada = load_tokenizer(args.llada_tokenizer, local_files_only=args.local_files_only)
    dream = load_tokenizer(args.dream_tokenizer, local_files_only=args.local_files_only)
    llada_mask_id = mask_token_id(llada, "LLaDA", fallback=args.llada_mask_id)
    dream_mask_id = mask_token_id(dream, "Dream")

    tokenizers = {"llada": llada, "dream": dream}
    mask_ids = {"llada": llada_mask_id, "dream": dream_mask_id}
    master_name = args.master
    auxiliary_name = "llada" if master_name == "dream" else "dream"
    master = tokenizers[master_name]
    auxiliary = tokenizers[auxiliary_name]
    master_mask_id = mask_ids[master_name]
    auxiliary_mask_id = mask_ids[auxiliary_name]

    committed_prefix = " answer is "
    committed_suffix = " apples."
    master_prompt_ids = encode(master, args.question)
    auxiliary_prompt_ids = encode(auxiliary, args.question)
    master_generated_ids = (
        encode(master, committed_prefix)
        + [master_mask_id]
        + encode(master, committed_suffix)
    )

    master_view = build_model_canvas_view(
        master_prompt_ids,
        master_generated_ids,
        master_tokenizer=master,
        model_tokenizer=master,
        master_mask_id=master_mask_id,
        model_mask_id=master_mask_id,
    )
    auxiliary_view = build_model_canvas_view(
        auxiliary_prompt_ids,
        master_generated_ids,
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_id=master_mask_id,
        model_mask_id=auxiliary_mask_id,
    )

    answer_texts = ["5", "6", "7"]
    master_answer_ids = [encode(master, text)[0] for text in answer_texts if encode(master, text)]
    auxiliary_answer_ids = [encode(auxiliary, text)[0] for text in answer_texts if encode(auxiliary, text)]
    auxiliary_answer_weights = dict(zip(auxiliary_answer_ids, [0.62, 0.30, 0.08]))
    master_answer_weights = dict(zip(master_answer_ids, [0.30, 0.48, 0.14]))

    active_aux_ids = sorted(
        set(auxiliary_view.input_ids[auxiliary_view.generation_slice].tolist())
        | set(auxiliary_answer_ids)
    )
    candidate_master_ids = sorted(
        set(master_view.input_ids[master_view.generation_slice].tolist())
        | set(master_answer_ids)
    )

    auxiliary_probabilities = torch.zeros(
        len(auxiliary_view.offsets),
        len(auxiliary),
        dtype=torch.float32,
    )
    for row, token_id in enumerate(auxiliary_view.input_ids[auxiliary_view.generation_slice].tolist()):
        if token_id == auxiliary_mask_id:
            auxiliary_probabilities[row] = make_sparse_distribution(
                len(auxiliary),
                auxiliary_answer_weights,
            )
        else:
            auxiliary_probabilities[row] = make_sparse_distribution(
                len(auxiliary),
                {int(token_id): 1.0},
            )

    master_probabilities = torch.zeros(
        len(master_view.offsets),
        len(master),
        dtype=torch.float32,
    )
    for row, token_id in enumerate(master_view.input_ids[master_view.generation_slice].tolist()):
        if token_id == master_mask_id:
            master_probabilities[row] = make_sparse_distribution(
                len(master),
                master_answer_weights,
            )
        else:
            master_probabilities[row] = make_sparse_distribution(
                len(master),
                {int(token_id): 1.0},
            )

    similarities, top_master_ids, top_weights = build_sparse_vocab_projection_for_active_ids(
        auxiliary,
        master,
        active_aux_ids,
        candidate_master_ids,
        temperature=args.projection_temperature,
        top_k=args.top_k,
    )
    spatial_aux = spatial_warp_probabilities(
        auxiliary_probabilities,
        auxiliary_view.overlap_matrix,
    )
    projected_aux = project_sparse_vocab(
        spatial_aux,
        active_aux_ids,
        top_master_ids,
        top_weights,
        master_vocab_size=len(master),
    )
    fused = args.alpha * master_probabilities + (1.0 - args.alpha) * projected_aux

    print("Question:")
    print(f"  {args.question}")
    print(f"\nMaster tokenizer: {master_name} ({args.dream_tokenizer if master_name == 'dream' else args.llada_tokenizer})")
    print(f"Auxiliary tokenizer: {auxiliary_name} ({args.llada_tokenizer if auxiliary_name == 'llada' else args.dream_tokenizer})")
    print(f"Master vocab size: {len(master)}")
    print(f"Auxiliary vocab size: {len(auxiliary)}")
    print(f"Conceptual dense Pi shape: ({len(auxiliary)}, {len(master)})")
    print(f"Realized sparse Pi table shape: active_aux_ids={len(active_aux_ids)}, top_k={top_master_ids.shape[1]}")

    print_token_sequence(
        "\nMaster native generated ids:",
        master,
        master_view.input_ids[master_view.generation_slice].tolist(),
        master_mask_id,
    )
    print(f"Master logical offsets: {master_view.offsets}")
    print_token_sequence(
        "\nAuxiliary native generated ids:",
        auxiliary,
        auxiliary_view.input_ids[auxiliary_view.generation_slice].tolist(),
        auxiliary_mask_id,
    )
    print(f"Auxiliary logical offsets: {auxiliary_view.offsets}")

    print(f"\nSpatial matrix Omega shape={tuple(auxiliary_view.overlap_matrix.shape)}")
    print(auxiliary_view.overlap_matrix.to_dense())

    print("\nSparse vocab projection rows, active auxiliary token -> top master tokens:")
    for row, aux_id in enumerate(active_aux_ids):
        mapped = [
            (
                token_label(master, int(master_id), master_mask_id),
                round(float(weight), 4),
                round(float(similarity), 4),
            )
            for master_id, weight, similarity in zip(
                top_master_ids[row],
                top_weights[row],
                similarities[row, torch.tensor([
                    candidate_master_ids.index(int(master_id))
                    for master_id in top_master_ids[row].tolist()
                ])],
            )
        ]
        print(f"  {token_label(auxiliary, aux_id, auxiliary_mask_id)} -> {mapped}")

    print(f"\nSpatially warped auxiliary probabilities shape={tuple(spatial_aux.shape)}")
    print(f"Auxiliary probabilities projected into master vocab shape={tuple(projected_aux.shape)}")
    print(f"Master probabilities shape={tuple(master_probabilities.shape)}")
    print(f"Fused probabilities shape={tuple(fused.shape)}")

    print("\nTop tokens by master slot:")
    for slot in range(fused.shape[0]):
        print(f"  slot {slot}:")
        print(f"    master     {top_tokens(master, master_probabilities[slot], mask_id=master_mask_id)}")
        print(f"    aux->master {top_tokens(master, projected_aux[slot], mask_id=master_mask_id)}")
        print(f"    fused      {top_tokens(master, fused[slot], mask_id=master_mask_id)}")


if __name__ == "__main__":
    main()
