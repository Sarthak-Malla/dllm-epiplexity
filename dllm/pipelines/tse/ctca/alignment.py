"""Embedding-space alignment for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AlignmentResult:
    """Orthogonal map and token IDs used to estimate it."""

    rotation: torch.Tensor
    auxiliary_anchor_ids: torch.Tensor
    master_anchor_ids: torch.Tensor


def normalize_token_text(token: str) -> str:
    """Normalize common tokenizer boundary markers for anchor matching."""
    normalized = token.replace("Ġ", "").replace("▁", "").replace("##", "")
    return normalized.strip().casefold()


def collect_shared_anchors(
    auxiliary_tokenizer,
    master_tokenizer,
    *,
    auxiliary_vocab_size: int,
    master_vocab_size: int,
    num_anchors: int,
) -> tuple[list[int], list[int]]:
    """Collect deterministic one-to-one shared string anchors."""
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")

    def normalized_vocab(tokenizer, vocab_size: int) -> dict[str, int]:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        candidates: dict[str, int] = {}
        for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            token_id = int(token_id)
            normalized = normalize_token_text(token)
            if (
                not normalized
                or token_id < 0
                or token_id >= vocab_size
                or token_id in special_ids
            ):
                continue
            candidates.setdefault(normalized, token_id)
        return candidates

    auxiliary = normalized_vocab(auxiliary_tokenizer, auxiliary_vocab_size)
    master = normalized_vocab(master_tokenizer, master_vocab_size)
    shared = sorted(set(auxiliary).intersection(master))[:num_anchors]
    return [auxiliary[token] for token in shared], [master[token] for token in shared]


def align_embeddings_procrustes(
    auxiliary_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    auxiliary_tokenizer,
    master_tokenizer,
    *,
    num_anchors: int = 3000,
    min_anchors: int = 128,
) -> AlignmentResult:
    """Estimate an auxiliary-to-master orthogonal Procrustes map."""
    if auxiliary_embeddings.ndim != 2 or master_embeddings.ndim != 2:
        raise ValueError("embedding matrices must have shape [vocab_size, hidden_size]")
    if min_anchors < 1 or min_anchors > num_anchors:
        raise ValueError("min_anchors must be between 1 and num_anchors")

    auxiliary_ids, master_ids = collect_shared_anchors(
        auxiliary_tokenizer,
        master_tokenizer,
        auxiliary_vocab_size=auxiliary_embeddings.shape[0],
        master_vocab_size=master_embeddings.shape[0],
        num_anchors=num_anchors,
    )
    if len(auxiliary_ids) < min_anchors:
        raise ValueError(
            "CTCA embedding alignment found too few shared anchors: "
            f"{len(auxiliary_ids)} < {min_anchors}"
        )

    compute_device = auxiliary_embeddings.device
    auxiliary_index = torch.tensor(auxiliary_ids, dtype=torch.long, device=compute_device)
    master_index = torch.tensor(
        master_ids, dtype=torch.long, device=master_embeddings.device
    )
    auxiliary_anchor = F.normalize(
        auxiliary_embeddings.index_select(0, auxiliary_index).float(), dim=-1
    )
    master_anchor = F.normalize(
        master_embeddings.index_select(0, master_index).to(compute_device).float(), dim=-1
    )
    cross_covariance = auxiliary_anchor.transpose(0, 1) @ master_anchor
    left, _, right = torch.svd(cross_covariance)
    rotation = left @ right.transpose(0, 1)
    return AlignmentResult(
        rotation=rotation,
        auxiliary_anchor_ids=auxiliary_index.cpu(),
        master_anchor_ids=master_index.cpu(),
    )
