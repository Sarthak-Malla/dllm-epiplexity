"""Token matching and chunked CTCA vocabulary projection.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass

import torch

from dllm.pipelines.tse.utils import timer


@dataclass(frozen=True)
class TokenMatches:
    """Direct matches and conditional first pieces for the fallback columns."""

    auxiliary_ids: torch.Tensor
    master_ids: torch.Tensor
    unmatched_ids: torch.Tensor
    prefix_columns: torch.Tensor
    prefix_master_ids: torch.Tensor
    required_right_masks: torch.Tensor


def projection_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """Identify assigned model IDs that can carry generation probability."""
    mask_id = getattr(tokenizer, "mask_token_id", None)
    token_ids = {
        token_id
        for token_id in tokenizer.get_vocab().values()
        if 0 <= token_id < vocab_size and token_id != mask_id
    }
    if not token_ids:
        raise ValueError("CTCA requires an assigned vocabulary ID other than the mask")
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    allowed[list(token_ids)] = True
    return allowed


def _decoded_vocabulary(tokenizer, allowed: torch.Tensor) -> dict[int, str]:
    token_ids = allowed.nonzero(as_tuple=True)[0].tolist()
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    decoded = {}
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        try:
            text = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except UnicodeDecodeError:
            continue
        if text and "\ufffd" not in text:
            decoded[token_id] = text
    return decoded


def match_decoded_tokens(
    auxiliary_tokenizer,
    master_tokenizer,
    *,
    auxiliary_vocab_size: int,
    master_vocab_size: int,
) -> TokenMatches:
    """Find direct matches and lossless multi-token encodings in the master."""
    auxiliary_allowed = projection_token_mask(auxiliary_tokenizer, auxiliary_vocab_size)
    auxiliary_texts = _decoded_vocabulary(auxiliary_tokenizer, auxiliary_allowed)
    master_allowed = projection_token_mask(master_tokenizer, master_vocab_size)
    master_texts = _decoded_vocabulary(master_tokenizer, master_allowed)
    master_by_text: dict[str, int | None] = {}
    for master_id, text in master_texts.items():
        master_by_text[text] = None if text in master_by_text else master_id

    master_vocab = master_tokenizer.get_vocab()
    matches: dict[int, int] = {}
    for token, auxiliary_id in auxiliary_tokenizer.get_vocab().items():
        decoded = auxiliary_texts.get(auxiliary_id)
        if decoded is None:
            continue
        same_spelling = master_vocab.get(token)
        master_id = (
            same_spelling
            if same_spelling is not None and master_texts.get(same_spelling) == decoded
            else master_by_text.get(decoded)
        )
        if master_id is not None:
            matches[auxiliary_id] = master_id

    matched_ids = sorted(matches)
    unmatched_ids = [
        idx
        for idx in auxiliary_allowed.nonzero(as_tuple=True)[0].tolist()
        if idx not in matches
    ]
    master_piece_ids = set(master_allowed.nonzero(as_tuple=True)[0].tolist())
    master_piece_ids.difference_update(getattr(master_tokenizer, "all_special_ids", ()))
    prefix_columns = []
    prefix_master_ids = []
    required_right_masks = []
    for column, auxiliary_id in enumerate(unmatched_ids):
        decoded = auxiliary_texts.get(auxiliary_id)
        if decoded is None or decoded in master_by_text:
            continue
        try:
            pieces = master_tokenizer(decoded, add_special_tokens=False)["input_ids"]
            if len(pieces) < 2 or not master_piece_ids.issuperset(pieces):
                continue
            reconstructed = master_tokenizer.decode(
                pieces, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except UnicodeError:
            continue
        if reconstructed == decoded:
            prefix_columns.append(column)
            prefix_master_ids.append(pieces[0])
            required_right_masks.append(len(pieces) - 1)

    return TokenMatches(
        auxiliary_ids=torch.tensor(matched_ids, dtype=torch.long),
        master_ids=torch.tensor([matches[idx] for idx in matched_ids], dtype=torch.long),
        unmatched_ids=torch.tensor(unmatched_ids, dtype=torch.long),
        prefix_columns=torch.tensor(prefix_columns, dtype=torch.long),
        prefix_master_ids=torch.tensor(prefix_master_ids, dtype=torch.long),
        required_right_masks=torch.tensor(required_right_masks, dtype=torch.long),
    )


def normalize_projection_rows(
    probabilities: torch.Tensor, epsilon: float = 1e-9
) -> torch.Tensor:
    """Normalize projected rows while leaving zero-mass rows empty."""
    row_mass = probabilities.sum(dim=-1, keepdim=True)
    return torch.where(
        row_mass > epsilon,
        probabilities / row_mass.clamp_min(epsilon),
        probabilities,
    )


def project_vocab_sparse_topk(
    spatial_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    master_vocab_size: int,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
    normalize: bool = True,
) -> torch.Tensor:
    """Project probabilities through a sparse top-k vocabulary table."""
    if spatial_probabilities.ndim != 2:
        raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
    if top_indices.ndim != 2 or top_weights.ndim != 2:
        raise ValueError("top-k tables must be two-dimensional")
    if top_indices.shape != top_weights.shape:
        raise ValueError("top-k indices and weights must have identical shapes")
    if spatial_probabilities.shape[1] != top_indices.shape[0]:
        raise ValueError("auxiliary probability vocabulary does not match top-k table")
    if master_vocab_size < 1 or chunk_size < 1 or epsilon <= 0:
        raise ValueError("master_vocab_size, chunk_size, and epsilon must be positive")
    if (spatial_probabilities < 0).any() or not torch.isfinite(
        spatial_probabilities
    ).all():
        raise ValueError("spatial_probabilities must be finite and non-negative")

    device = spatial_probabilities.device
    indices = top_indices.to(device=device, dtype=torch.long)
    weights = top_weights.to(device=device, dtype=torch.float32)
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        master_vocab_size,
        dtype=torch.float32,
        device=device,
    )
    probabilities = spatial_probabilities.float()
    with timer("ctca.projection.sparse_topk_scatter"):
        for start in range(0, probabilities.shape[1], chunk_size):
            end = min(start + chunk_size, probabilities.shape[1])
            chunk_indices = indices[start:end]
            chunk_weights = weights[start:end]
            contributions = (
                probabilities[:, start:end].unsqueeze(-1)
                * chunk_weights.unsqueeze(0)
            )
            expanded_indices = chunk_indices.flatten().unsqueeze(0).expand(
                probabilities.shape[0], -1
            )
            projected.scatter_add_(1, expanded_indices, contributions.flatten(1))

    if normalize:
        with timer("ctca.projection.normalize"):
            return normalize_projection_rows(projected, epsilon)
    return projected
