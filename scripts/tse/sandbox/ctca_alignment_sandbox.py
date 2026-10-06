"""Run a standalone CTCA tokenizer-alignment and probability-fusion sandbox.

From the repository root:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/sandbox/ctca_alignment_sandbox.py

This script intentionally avoids imports from the project. It duplicates only
the small CTCA slice needed to inspect:
    1. native tokenizer canvas construction,
    2. auxiliary-to-master spatial overlap,
    3. auxiliary-to-master vocabulary projection,
    4. probability fusion on the master canvas.
"""

from dataclasses import dataclass
from typing import Sequence

import torch


Span = tuple[float, float]


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


class ToyTokenizer:
    """A tiny greedy tokenizer with offsets and decode support."""

    def __init__(self, tokens: Sequence[str], *, mask_token: str, pad_token: str) -> None:
        self.tokens = list(tokens)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.mask_token_id = self.vocab[mask_token]
        self.pad_token_id = self.vocab[pad_token]
        self.all_special_ids = {self.mask_token_id, self.pad_token_id}

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            self.tokens[int(token_id)]
            for token_id in token_ids
            if int(token_id) not in self.all_special_ids
        )

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[Span]]:
        ids: list[int] = []
        offsets: list[Span] = []
        ordinary_tokens = sorted(
            (
                token
                for token, token_id in self.vocab.items()
                if token_id not in self.all_special_ids
            ),
            key=len,
            reverse=True,
        )
        position = 0
        while position < len(text):
            token = next(
                (
                    candidate
                    for candidate in ordinary_tokens
                    if text.startswith(candidate, position)
                ),
                None,
            )
            if token is None:
                raise ValueError(
                    f"{self.__class__.__name__} cannot encode text at "
                    f"offset {position}: {text!r}"
                )
            ids.append(self.vocab[token])
            offsets.append((float(position), float(position + len(token))))
            position += len(token)
        return ids, offsets

    def token_label(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id == self.mask_token_id:
            return "[MASK]"
        return repr(self.tokens[token_id])


def normalize_rows(matrix: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    row_norms = matrix.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    return matrix / row_norms


def softmax(matrix: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = matrix - matrix.max(dim=dim, keepdim=True).values
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def _prefix_boundaries(tokenizer: ToyTokenizer, token_ids: Sequence[int]) -> list[int]:
    boundaries = [0]
    for end in range(1, len(token_ids) + 1):
        boundaries.append(len(tokenizer.decode(token_ids[:end])))
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
            return float(logical_start + index + fraction)
    return float(logical_start + len(boundaries) - 1)


def build_overlap_matrix(
    offsets_aux: Sequence[Span],
    offsets_master: Sequence[Span],
) -> torch.Tensor:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for aux_index, (aux_start, aux_end) in enumerate(offsets_aux):
        aux_length = aux_end - aux_start
        if aux_length == 0:
            continue
        for master_index, (master_start, master_end) in enumerate(offsets_master):
            overlap = max(0.0, min(aux_end, master_end) - max(aux_start, master_start))
            if overlap > 0:
                rows.append(aux_index)
                columns.append(master_index)
                values.append(overlap / aux_length)

    indices = torch.tensor([rows, columns], dtype=torch.long)
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
    master_tokenizer: ToyTokenizer,
    model_tokenizer: ToyTokenizer,
) -> CanvasView:
    native_generated_ids: list[int] = []
    native_offsets: list[Span] = []

    index = 0
    while index < len(master_generated_ids):
        if master_generated_ids[index] == master_tokenizer.mask_token_id:
            native_generated_ids.append(model_tokenizer.mask_token_id)
            native_offsets.append((float(index), float(index + 1)))
            index += 1
            continue

        run_start = index
        while (
            index < len(master_generated_ids)
            and master_generated_ids[index] != master_tokenizer.mask_token_id
        ):
            index += 1
        run_ids = list(master_generated_ids[run_start:index])
        run_text = master_tokenizer.decode(run_ids)
        encoded_ids, character_offsets = model_tokenizer.encode_with_offsets(run_text)
        boundaries = _prefix_boundaries(master_tokenizer, run_ids)

        native_generated_ids.extend(encoded_ids)
        for start, end in character_offsets:
            native_offsets.append(
                (
                    _character_to_logical(start, boundaries, run_start),
                    _character_to_logical(end, boundaries, run_start),
                )
            )

    master_offsets = tuple((float(i), float(i + 1)) for i in range(len(master_generated_ids)))
    overlap = build_overlap_matrix(native_offsets, master_offsets)
    return CanvasView(
        input_ids=torch.tensor(list(native_prompt_ids) + native_generated_ids),
        generation_start=len(native_prompt_ids),
        offsets=tuple(native_offsets),
        overlap_matrix=overlap,
    )


def spatial_warp_probabilities(
    probabilities_aux: torch.Tensor,
    overlap_matrix: torch.Tensor,
) -> torch.Tensor:
    return torch.sparse.mm(overlap_matrix.transpose(0, 1), probabilities_aux)


def build_vocab_projection_matrix(
    auxiliary_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    auxiliary = normalize_rows(auxiliary_embeddings.float())
    master = normalize_rows(master_embeddings.float())
    similarities = auxiliary @ master.transpose(0, 1)
    return softmax(similarities / temperature, dim=-1)


def project_vocab(
    spatial_probabilities: torch.Tensor,
    vocab_projection: torch.Tensor,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    projected = spatial_probabilities.float() @ vocab_projection.float()
    row_mass = projected.sum(dim=-1, keepdim=True)
    return torch.where(row_mass > epsilon, projected / row_mass.clamp_min(epsilon), projected)


def fuse_probabilities(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * probabilities_a.float() + (1.0 - alpha) * probabilities_b.float()


def distribution(vocab_size: int, weighted_tokens: dict[int, float]) -> torch.Tensor:
    probabilities = torch.full((vocab_size,), 0.01, dtype=torch.float32)
    for token_id, probability in weighted_tokens.items():
        probabilities[int(token_id)] = float(probability)
    return probabilities / probabilities.sum()


def fake_embeddings(tokenizer: ToyTokenizer, semantic_coordinates: dict[str, Sequence[float]]) -> torch.Tensor:
    rows = []
    for token in tokenizer.tokens:
        coordinates = semantic_coordinates.get(token, (0.0, 0.0, 0.0, 0.0))
        rows.append(list(coordinates))
    return torch.tensor(rows, dtype=torch.float32)


def top_tokens(
    tokenizer: ToyTokenizer,
    probabilities: torch.Tensor,
    *,
    k: int = 4,
) -> list[tuple[str, float]]:
    values, token_ids = torch.topk(probabilities.float(), k=min(k, probabilities.shape[-1]))
    return [
        (tokenizer.token_label(int(token_id)), round(float(value), 4))
        for value, token_id in zip(values, token_ids)
    ]


def print_matrix(name: str, matrix: torch.Tensor) -> None:
    print(f"\n{name} shape={tuple(matrix.shape)}")
    print(torch.round(matrix.float() * 10000) / 10000)


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    question = "Question: Mia has 9 apples and gives away 3. How many remain?"
    master = ToyTokenizer(
        [
            "<unk>",
            " answer",
            " is",
            " 5",
            " 6",
            " 7",
            " apples",
            ".",
            "[PAD]",
            "[MASK]",
        ],
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    auxiliary = ToyTokenizer(
        [
            "<unk>",
            " ans",
            "wer",
            " is",
            " five",
            " six",
            " seven",
            " apple",
            "s",
            ".",
            "<pad>",
            "<mask>",
        ],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    prompt_master_ids = [master.vocab["<unk>"]]
    prompt_aux_ids = [auxiliary.vocab["<unk>"]]
    master_generated_ids = [
        master.vocab[" answer"],
        master.vocab[" is"],
        master.mask_token_id,
        master.vocab[" apples"],
        master.vocab["."],
    ]

    master_view = build_model_canvas_view(
        prompt_master_ids,
        master_generated_ids,
        master_tokenizer=master,
        model_tokenizer=master,
    )
    auxiliary_view = build_model_canvas_view(
        prompt_aux_ids,
        master_generated_ids,
        master_tokenizer=master,
        model_tokenizer=auxiliary,
    )

    master_semantics = {
        " answer": (1.0, 0.0, 0.0, 0.0),
        " is": (0.0, 1.0, 0.0, 0.0),
        " 5": (0.0, 0.0, 0.8, 0.2),
        " 6": (0.0, 0.0, 1.0, 0.0),
        " 7": (0.0, 0.0, 0.8, -0.2),
        " apples": (0.0, 0.0, 0.0, 1.0),
        ".": (0.3, 0.3, 0.0, 0.0),
    }
    auxiliary_semantics = {
        " ans": (1.0, 0.0, 0.0, 0.0),
        "wer": (1.0, 0.0, 0.0, 0.0),
        " is": (0.0, 1.0, 0.0, 0.0),
        " five": (0.0, 0.0, 0.8, 0.2),
        " six": (0.0, 0.0, 1.0, 0.0),
        " seven": (0.0, 0.0, 0.8, -0.2),
        " apple": (0.0, 0.0, 0.0, 1.0),
        "s": (0.0, 0.0, 0.0, 1.0),
        ".": (0.3, 0.3, 0.0, 0.0),
    }
    master_embeddings = fake_embeddings(master, master_semantics)
    auxiliary_embeddings = fake_embeddings(auxiliary, auxiliary_semantics)
    vocab_matrix = build_vocab_projection_matrix(
        auxiliary_embeddings,
        master_embeddings,
        temperature=0.20,
    )

    master_probabilities = torch.stack(
        [
            distribution(len(master.tokens), {master.vocab[" answer"]: 0.85, master.vocab[" is"]: 0.08}),
            distribution(len(master.tokens), {master.vocab[" is"]: 0.82, master.vocab[" answer"]: 0.10}),
            distribution(len(master.tokens), {master.vocab[" 6"]: 0.48, master.vocab[" 5"]: 0.30, master.vocab[" 7"]: 0.14}),
            distribution(len(master.tokens), {master.vocab[" apples"]: 0.76, master.vocab["."]: 0.12}),
            distribution(len(master.tokens), {master.vocab["."]: 0.90, master.vocab[" apples"]: 0.04}),
        ]
    )
    auxiliary_probabilities = torch.stack(
        [
            distribution(len(auxiliary.tokens), {auxiliary.vocab[" ans"]: 0.72, auxiliary.vocab["wer"]: 0.18}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab["wer"]: 0.70, auxiliary.vocab[" ans"]: 0.16}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab[" is"]: 0.86, auxiliary.vocab[" ans"]: 0.04}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab[" six"]: 0.62, auxiliary.vocab[" five"]: 0.20, auxiliary.vocab[" seven"]: 0.08}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab[" apple"]: 0.78, auxiliary.vocab["s"]: 0.10}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab["s"]: 0.80, auxiliary.vocab[" apple"]: 0.08}),
            distribution(len(auxiliary.tokens), {auxiliary.vocab["."]: 0.88, auxiliary.vocab["s"]: 0.05}),
        ]
    )

    spatial_aux_probabilities = spatial_warp_probabilities(
        auxiliary_probabilities,
        auxiliary_view.overlap_matrix,
    )
    projected_aux_probabilities = project_vocab(spatial_aux_probabilities, vocab_matrix)
    fused_probabilities = fuse_probabilities(
        master_probabilities,
        projected_aux_probabilities,
        alpha=0.50,
    )

    print("Question:")
    print(f"  {question}")
    print("\nMaster generation canvas:")
    print("  " + " | ".join(master.token_label(token_id) for token_id in master_generated_ids))
    print("\nMaster native generated ids:")
    print("  " + " | ".join(master.token_label(token_id) for token_id in master_view.input_ids[master_view.generation_slice]))
    print("Master logical offsets:")
    print(f"  {master_view.offsets}")
    print("\nAuxiliary native generated ids:")
    print("  " + " | ".join(auxiliary.token_label(token_id) for token_id in auxiliary_view.input_ids[auxiliary_view.generation_slice]))
    print("Auxiliary logical offsets:")
    print(f"  {auxiliary_view.offsets}")

    print_matrix("Spatial matrix Omega, auxiliary tokens -> master slots", auxiliary_view.overlap_matrix.to_dense())
    print_matrix("Vocab matrix Pi, auxiliary vocab -> master vocab", vocab_matrix)
    print_matrix("Spatially warped auxiliary probabilities", spatial_aux_probabilities)
    print_matrix("Auxiliary probabilities projected into master vocab", projected_aux_probabilities)
    print_matrix("Master probabilities", master_probabilities)
    print_matrix("Fused probabilities", fused_probabilities)

    print("\nTop tokens by master slot:")
    for slot in range(fused_probabilities.shape[0]):
        print(f"  slot {slot}:")
        print(f"    master     {top_tokens(master, master_probabilities[slot])}")
        print(f"    aux->master {top_tokens(master, projected_aux_probabilities[slot])}")
        print(f"    fused      {top_tokens(master, fused_probabilities[slot])}")


if __name__ == "__main__":
    main()
