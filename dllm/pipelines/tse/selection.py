"""Schedule-aware position selection and canvas commitment for TSE."""

import torch


def select_positions(
    scores: torch.Tensor,
    predicted_tokens: torch.Tensor,
    active_positions: torch.Tensor,
    transfer_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exactly each batch item's scheduled number of positions."""
    if scores.ndim != 1 or predicted_tokens.ndim != 1:
        raise ValueError("scores and predicted_tokens must be one-dimensional")
    if scores.shape != predicted_tokens.shape:
        raise ValueError("scores and predicted_tokens must have identical shapes")
    if active_positions.ndim != 2 or active_positions.shape[1] != 2:
        raise ValueError("active_positions must have shape [N, 2]")
    if active_positions.shape[0] != scores.shape[0]:
        raise ValueError("active positions and scores must have the same length")
    if transfer_counts.ndim != 1 or (transfer_counts < 0).any():
        raise ValueError("transfer_counts must be a non-negative vector")

    selected = []
    for batch_index, count in enumerate(transfer_counts.tolist()):
        batch_indices = torch.nonzero(
            active_positions[:, 0] == batch_index, as_tuple=False
        ).flatten()
        if count > batch_indices.numel():
            raise ValueError("transfer count exceeds active positions for a batch item")
        if count == 0:
            continue
        order = torch.argsort(scores[batch_indices], descending=True, stable=True)
        selected.append(batch_indices[order[:count]])

    if not selected:
        empty_positions = active_positions[:0]
        empty_tokens = predicted_tokens[:0]
        return empty_positions, empty_tokens

    selected_indices = torch.cat(selected)
    return active_positions[selected_indices], predicted_tokens[selected_indices]


def commit_tokens(
    canvas: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Commit selected token IDs into a cloned canvas and return it."""
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [N, 2]")
    if token_ids.ndim != 1 or token_ids.shape[0] != positions.shape[0]:
        raise ValueError("token_ids must align with positions")

    updated = canvas.clone()
    if positions.numel():
        updated[positions[:, 0], positions[:, 1]] = token_ids.to(updated.device)
    return updated