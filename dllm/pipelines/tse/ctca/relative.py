"""Relative-anchor vocabulary alignment for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.pipelines.tse.utils import timer

from .projection import normalize_projection_rows


RELATIVE_ANCHOR_VERSION = 2


@dataclass(frozen=True)
class RelativeAnchorSelection:
    """Shared anchors selected for relative-anchor projection."""

    auxiliary_anchor_ids: torch.Tensor
    master_anchor_ids: torch.Tensor
    total_anchors: int

    @property
    def selected_anchors(self) -> int:
        return int(self.auxiliary_anchor_ids.numel())


def collect_relative_anchor_ids(
    auxiliary_tokenizer,
    master_tokenizer,
    *,
    auxiliary_vocab_size: int,
    master_vocab_size: int,
    num_anchors: int | str = "auto",
    min_anchors: int = 128,
) -> RelativeAnchorSelection:
    """Pair exact non-special vocabulary strings in deterministic order."""
    if min_anchors < 1:
        raise ValueError("min_anchors must be positive")

    def usable_vocab(tokenizer, vocab_size: int) -> dict[str, int]:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        special_ids.add(getattr(tokenizer, "mask_token_id", None))
        return {
            token: token_id
            for token, token_id in tokenizer.get_vocab().items()
            if token and 0 <= token_id < vocab_size and token_id not in special_ids
        }

    auxiliary = usable_vocab(auxiliary_tokenizer, auxiliary_vocab_size)
    master = usable_vocab(master_tokenizer, master_vocab_size)
    shared = sorted(auxiliary.keys() & master.keys())
    selected_count = _selected_anchor_count(num_anchors, len(shared))
    if selected_count < min_anchors:
        raise ValueError(
            "CTCA relative-anchor alignment found too few shared anchors: "
            f"{selected_count} < {min_anchors}"
        )
    selected = shared[:selected_count]
    return RelativeAnchorSelection(
        auxiliary_anchor_ids=torch.tensor(
            [auxiliary[token] for token in selected], dtype=torch.long
        ),
        master_anchor_ids=torch.tensor(
            [master[token] for token in selected], dtype=torch.long
        ),
        total_anchors=len(shared),
    )


def _selected_anchor_count(num_anchors: int | str, total_anchors: int) -> int:
    if isinstance(num_anchors, str):
        if num_anchors.casefold() == "auto":
            return total_anchors
        try:
            parsed = int(num_anchors)
        except ValueError as error:
            raise ValueError("num_anchors must be an integer or 'auto'") from error
    else:
        parsed = int(num_anchors)
    if parsed < 1:
        raise ValueError("num_anchors must be positive or 'auto'")
    return min(parsed, total_anchors)


def relative_anchor_profiles(
    embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Represent tokens by normalized softmax similarities to anchors."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or anchor_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, hidden_size]")
    if embeddings.shape[1] != anchor_embeddings.shape[1]:
        raise ValueError("tokens and anchors must share a hidden size")
    tokens = F.normalize(embeddings.float(), dim=-1)
    anchors = F.normalize(anchor_embeddings.float(), dim=-1)
    scores = tokens @ anchors.transpose(0, 1)
    profiles = torch.softmax(scores / temperature, dim=-1)
    return F.normalize(profiles, dim=-1)


def _master_similarities(
    auxiliary_profiles: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchors: torch.Tensor,
    allowed: torch.Tensor | None,
    anchor_temperature: float,
) -> torch.Tensor:
    master_profiles = relative_anchor_profiles(
        master_embeddings, master_anchors, temperature=anchor_temperature
    )
    similarities = auxiliary_profiles @ master_profiles.transpose(0, 1)
    if allowed is not None:
        similarities.masked_fill_(~allowed, -torch.inf)
    return similarities


def _prepare_master_mask(
    allowed: torch.Tensor | None, vocab_size: int, device: torch.device
) -> torch.Tensor | None:
    if allowed is None:
        return None
    if allowed.shape != (vocab_size,) or allowed.dtype != torch.bool:
        raise ValueError("master_token_mask must be a boolean master vocabulary vector")
    if not allowed.any():
        raise ValueError("master_token_mask must allow at least one destination")
    return allowed.to(device)


def build_sparse_topk_relative_anchor_projection(
    auxiliary_embeddings: torch.Tensor,
    auxiliary_anchor_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchor_embeddings: torch.Tensor,
    *,
    top_k: int,
    anchor_temperature: float = 0.01,
    projection_temperature: float = 0.05,
    chunk_size: int = 2500,
    master_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse top-k aux-to-master projection in relative-anchor space."""
    _validate_relative_inputs(
        auxiliary_embeddings,
        auxiliary_anchor_embeddings,
        master_embeddings,
        master_anchor_embeddings,
        anchor_temperature=anchor_temperature,
        projection_temperature=projection_temperature,
        chunk_size=chunk_size,
    )
    if top_k < 1:
        raise ValueError("top_k must be positive")
    device = auxiliary_embeddings.device
    master_token_mask = _prepare_master_mask(
        master_token_mask, master_embeddings.shape[0], device
    )
    candidate_count = (
        master_embeddings.shape[0]
        if master_token_mask is None
        else int(master_token_mask.sum())
    )
    top_k = min(top_k, candidate_count)
    auxiliary_anchor_embeddings = auxiliary_anchor_embeddings.to(device)
    master_anchor_embeddings = master_anchor_embeddings.to(device)
    top_indices_by_chunk = []
    top_weights_by_chunk = []

    with timer("ctca.relative.sparse_topk_build"):
        for auxiliary_start in range(0, auxiliary_embeddings.shape[0], chunk_size):
            auxiliary_end = min(auxiliary_start + chunk_size, auxiliary_embeddings.shape[0])
            with timer("ctca.relative.build_query_profiles"):
                auxiliary_profiles = relative_anchor_profiles(
                    auxiliary_embeddings[auxiliary_start:auxiliary_end],
                    auxiliary_anchor_embeddings,
                    temperature=anchor_temperature,
                )
            best_values = torch.full(
                (auxiliary_profiles.shape[0], top_k),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            )
            best_indices = torch.zeros(
                (auxiliary_profiles.shape[0], top_k),
                dtype=torch.long,
                device=device,
            )
            for master_start in range(0, master_embeddings.shape[0], chunk_size):
                master_end = min(master_start + chunk_size, master_embeddings.shape[0])
                allowed = (
                    None
                    if master_token_mask is None
                    else master_token_mask[master_start:master_end]
                )
                similarities = _master_similarities(
                    auxiliary_profiles,
                    master_embeddings[master_start:master_end].to(device),
                    master_anchor_embeddings,
                    allowed,
                    anchor_temperature,
                )
                local_k = min(top_k, similarities.shape[1])
                local_values, local_indices = torch.topk(similarities, k=local_k, dim=-1)
                local_indices = local_indices + master_start
                merged_values = torch.cat([best_values, local_values], dim=-1)
                merged_indices = torch.cat([best_indices, local_indices], dim=-1)
                best_values, order = torch.topk(merged_values, k=top_k, dim=-1)
                best_indices = torch.gather(merged_indices, 1, order)

            top_indices_by_chunk.append(best_indices.cpu())
            top_weights_by_chunk.append(
                torch.softmax(best_values / projection_temperature, dim=-1).cpu()
            )

    return torch.cat(top_indices_by_chunk, dim=0), torch.cat(top_weights_by_chunk, dim=0)


def project_vocab_relative_exact(
    spatial_probabilities: torch.Tensor,
    auxiliary_embeddings: torch.Tensor,
    auxiliary_anchor_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchor_embeddings: torch.Tensor,
    *,
    anchor_temperature: float = 0.01,
    projection_temperature: float = 0.05,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
    normalize: bool = True,
    master_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project probabilities through full relative-anchor similarities.

    This path is intentionally slow and intended for tests/debugging.
    """
    if spatial_probabilities.ndim != 2:
        raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
    if spatial_probabilities.shape[1] != auxiliary_embeddings.shape[0]:
        raise ValueError("auxiliary probability vocabulary does not match embeddings")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (spatial_probabilities < 0).any() or not torch.isfinite(
        spatial_probabilities
    ).all():
        raise ValueError("spatial_probabilities must be finite and non-negative")
    _validate_relative_inputs(
        auxiliary_embeddings,
        auxiliary_anchor_embeddings,
        master_embeddings,
        master_anchor_embeddings,
        anchor_temperature=anchor_temperature,
        projection_temperature=projection_temperature,
        chunk_size=chunk_size,
    )

    device = spatial_probabilities.device
    master_token_mask = _prepare_master_mask(
        master_token_mask, master_embeddings.shape[0], device
    )
    probabilities = spatial_probabilities.float()
    auxiliary_embeddings = auxiliary_embeddings.to(device)
    auxiliary_anchor_embeddings = auxiliary_anchor_embeddings.to(device)
    master_embeddings = master_embeddings.to(device)
    master_anchor_embeddings = master_anchor_embeddings.to(device)
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        master_embeddings.shape[0],
        dtype=torch.float32,
        device=device,
    )

    with timer("ctca.relative.exact_project"):
        for auxiliary_start in range(0, auxiliary_embeddings.shape[0], chunk_size):
            auxiliary_end = min(auxiliary_start + chunk_size, auxiliary_embeddings.shape[0])
            with timer("ctca.relative.build_query_profiles"):
                auxiliary_profiles = relative_anchor_profiles(
                    auxiliary_embeddings[auxiliary_start:auxiliary_end],
                    auxiliary_anchor_embeddings,
                    temperature=anchor_temperature,
                )

            def projection_logits(start: int, end: int) -> torch.Tensor:
                allowed = (
                    None if master_token_mask is None else master_token_mask[start:end]
                )
                return _master_similarities(
                    auxiliary_profiles,
                    master_embeddings[start:end],
                    master_anchor_embeddings,
                    allowed,
                    anchor_temperature,
                ) / projection_temperature

            log_denominator = torch.full(
                (auxiliary_profiles.shape[0],),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            )
            with timer("ctca.relative.exact_logsumexp"):
                for master_start in range(0, master_embeddings.shape[0], chunk_size):
                    master_end = min(master_start + chunk_size, master_embeddings.shape[0])
                    logits = projection_logits(master_start, master_end)
                    log_denominator = torch.logaddexp(
                        log_denominator, torch.logsumexp(logits, dim=-1)
                    )

            active_probabilities = probabilities[:, auxiliary_start:auxiliary_end]
            for master_start in range(0, master_embeddings.shape[0], chunk_size):
                master_end = min(master_start + chunk_size, master_embeddings.shape[0])
                logits = projection_logits(master_start, master_end)
                weights = torch.exp(logits - log_denominator.unsqueeze(-1))
                projected[:, master_start:master_end].add_(
                    active_probabilities @ weights
                )

    if normalize:
        with timer("ctca.projection.normalize"):
            return normalize_projection_rows(projected, epsilon)
    return projected


def _validate_relative_inputs(
    auxiliary_embeddings: torch.Tensor,
    auxiliary_anchor_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchor_embeddings: torch.Tensor,
    *,
    anchor_temperature: float,
    projection_temperature: float,
    chunk_size: int,
) -> None:
    if anchor_temperature <= 0 or projection_temperature <= 0 or chunk_size < 1:
        raise ValueError("temperatures and chunk_size must be positive")
    for tensor in (
        auxiliary_embeddings,
        auxiliary_anchor_embeddings,
        master_embeddings,
        master_anchor_embeddings,
    ):
        if tensor.ndim != 2:
            raise ValueError("embedding matrices must be two-dimensional")
    if auxiliary_embeddings.shape[1] != auxiliary_anchor_embeddings.shape[1]:
        raise ValueError("auxiliary tokens and anchors must share a hidden size")
    if master_embeddings.shape[1] != master_anchor_embeddings.shape[1]:
        raise ValueError("master tokens and anchors must share a hidden size")
    if auxiliary_anchor_embeddings.shape[0] != master_anchor_embeddings.shape[0]:
        raise ValueError("auxiliary and master anchor counts must match")
