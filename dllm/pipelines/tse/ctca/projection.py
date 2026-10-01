"""Memory-bounded vocabulary projection for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

import torch
import torch.nn.functional as F

from dllm.pipelines.tse.utils import timer


def project_vocab_fused(
    spatial_probabilities: torch.Tensor,
    aligned_auxiliary_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    *,
    temperature: float = 0.05,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Project auxiliary probabilities without materializing the full map."""
    if spatial_probabilities.ndim != 2:
        raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
    if aligned_auxiliary_embeddings.ndim != 2 or master_embeddings.ndim != 2:
        raise ValueError("embedding matrices must be two-dimensional")
    if spatial_probabilities.shape[1] != aligned_auxiliary_embeddings.shape[0]:
        raise ValueError("auxiliary probability vocabulary does not match embeddings")
    if aligned_auxiliary_embeddings.shape[1] != master_embeddings.shape[1]:
        raise ValueError("aligned auxiliary and master embedding dimensions must match")
    if temperature <= 0 or chunk_size < 1 or epsilon <= 0:
        raise ValueError("temperature, chunk_size, and epsilon must be positive")
    if (spatial_probabilities < 0).any() or not torch.isfinite(
        spatial_probabilities
    ).all():
        raise ValueError("spatial_probabilities must be finite and non-negative")

    device = spatial_probabilities.device
    master = F.normalize(master_embeddings.to(device).float(), dim=-1)
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        master.shape[0],
        dtype=torch.float32,
        device=device,
    )
    probabilities = spatial_probabilities.float()
    for start in range(0, aligned_auxiliary_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, aligned_auxiliary_embeddings.shape[0])
        auxiliary_chunk = F.normalize(
            aligned_auxiliary_embeddings[start:end].to(device).float(), dim=-1
        )
        similarities = auxiliary_chunk @ master.transpose(0, 1)
        vocabulary_map = torch.softmax(similarities / temperature, dim=-1)
        projected.add_(probabilities[:, start:end] @ vocabulary_map)

    row_mass = projected.sum(dim=-1, keepdim=True)
    projected = torch.where(
        row_mass > epsilon,
        projected / row_mass.clamp_min(epsilon),
        projected,
    )
    return projected


def project_vocab_from_normalized_fused(
    spatial_probabilities: torch.Tensor,
    normalized_aligned_auxiliary_embeddings: torch.Tensor,
    normalized_master_embeddings: torch.Tensor,
    *,
    temperature: float = 0.05,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Project probabilities with pre-normalized, device-local embeddings."""
    if spatial_probabilities.ndim != 2:
        raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
    if (
        normalized_aligned_auxiliary_embeddings.ndim != 2
        or normalized_master_embeddings.ndim != 2
    ):
        raise ValueError("embedding matrices must be two-dimensional")
    if spatial_probabilities.shape[1] != normalized_aligned_auxiliary_embeddings.shape[0]:
        raise ValueError("auxiliary probability vocabulary does not match embeddings")
    if (
        normalized_aligned_auxiliary_embeddings.shape[1]
        != normalized_master_embeddings.shape[1]
    ):
        raise ValueError("aligned auxiliary and master dimensions must match")
    if temperature <= 0 or chunk_size < 1 or epsilon <= 0:
        raise ValueError("temperature, chunk_size, and epsilon must be positive")
    if (spatial_probabilities < 0).any() or not torch.isfinite(
        spatial_probabilities
    ).all():
        raise ValueError("spatial_probabilities must be finite and non-negative")

    device = spatial_probabilities.device
    if normalized_aligned_auxiliary_embeddings.device != device:
        raise ValueError("normalized auxiliary embeddings must already be on device")
    if normalized_master_embeddings.device != device:
        raise ValueError("normalized master embeddings must already be on device")

    probabilities = spatial_probabilities.float()
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        normalized_master_embeddings.shape[0],
        dtype=torch.float32,
        device=device,
    )
    with timer("ctca.projection.chunk_loop"):
        for start in range(
            0, normalized_aligned_auxiliary_embeddings.shape[0], chunk_size
        ):
            end = min(start + chunk_size, normalized_aligned_auxiliary_embeddings.shape[0])
            similarities = (
                normalized_aligned_auxiliary_embeddings[start:end]
                @ normalized_master_embeddings.transpose(0, 1)
            )
            vocabulary_map = torch.softmax(similarities / temperature, dim=-1)
            projected.add_(probabilities[:, start:end] @ vocabulary_map)

    with timer("ctca.projection.normalize"):
        row_mass = projected.sum(dim=-1, keepdim=True)
        return torch.where(
            row_mass > epsilon,
            projected / row_mass.clamp_min(epsilon),
            projected,
        )


def build_sparse_topk_vocab_projection(
    normalized_aligned_auxiliary_embeddings: torch.Tensor,
    normalized_master_embeddings: torch.Tensor,
    *,
    top_k: int,
    temperature: float = 0.05,
    chunk_size: int = 2500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse auxiliary-to-master top-k projection table."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("temperature and chunk_size must be positive")
    if (
        normalized_aligned_auxiliary_embeddings.ndim != 2
        or normalized_master_embeddings.ndim != 2
    ):
        raise ValueError("embedding matrices must be two-dimensional")
    if (
        normalized_aligned_auxiliary_embeddings.shape[1]
        != normalized_master_embeddings.shape[1]
    ):
        raise ValueError("aligned auxiliary and master dimensions must match")
    if top_k > normalized_master_embeddings.shape[0]:
        raise ValueError("top_k cannot exceed the master vocabulary size")
    if (
        normalized_aligned_auxiliary_embeddings.device
        != normalized_master_embeddings.device
    ):
        raise ValueError("embedding matrices must be on the same device")

    indices_by_chunk = []
    weights_by_chunk = []
    with timer("ctca.projection.sparse_topk_build"):
        for start in range(
            0, normalized_aligned_auxiliary_embeddings.shape[0], chunk_size
        ):
            end = min(start + chunk_size, normalized_aligned_auxiliary_embeddings.shape[0])
            similarities = (
                normalized_aligned_auxiliary_embeddings[start:end]
                @ normalized_master_embeddings.transpose(0, 1)
            )
            top_values, top_indices = torch.topk(similarities, k=top_k, dim=-1)
            top_weights = torch.softmax(top_values / temperature, dim=-1)
            indices_by_chunk.append(top_indices)
            weights_by_chunk.append(top_weights)

    return torch.cat(indices_by_chunk, dim=0), torch.cat(weights_by_chunk, dim=0)


def project_vocab_sparse_topk(
    spatial_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    master_vocab_size: int,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
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

    with timer("ctca.projection.normalize"):
        row_mass = projected.sum(dim=-1, keepdim=True)
        return torch.where(
            row_mass > epsilon,
            projected / row_mass.clamp_min(epsilon),
            projected,
        )


def project_vocab_with_rotation_fused(
    spatial_probabilities: torch.Tensor,
    auxiliary_embeddings: torch.Tensor,
    rotation: torch.Tensor,
    master_embeddings: torch.Tensor,
    *,
    temperature: float = 0.05,
    chunk_size: int = 2500,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Align and project auxiliary embedding chunks without storing E_aux @ R."""
    if spatial_probabilities.ndim != 2:
        raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
    if auxiliary_embeddings.ndim != 2 or master_embeddings.ndim != 2:
        raise ValueError("embedding matrices must be two-dimensional")
    if rotation.ndim != 2:
        raise ValueError("rotation must be a two-dimensional matrix")
    if auxiliary_embeddings.shape[1] != rotation.shape[0]:
        raise ValueError("rotation input dimension does not match auxiliary embeddings")
    if rotation.shape[1] != master_embeddings.shape[1]:
        raise ValueError("rotation output dimension does not match master embeddings")
    if spatial_probabilities.shape[1] != auxiliary_embeddings.shape[0]:
        raise ValueError("auxiliary probability vocabulary does not match embeddings")
    if temperature <= 0 or chunk_size < 1 or epsilon <= 0:
        raise ValueError("temperature, chunk_size, and epsilon must be positive")
    if (spatial_probabilities < 0).any() or not torch.isfinite(
        spatial_probabilities
    ).all():
        raise ValueError("spatial_probabilities must be finite and non-negative")

    device = spatial_probabilities.device
    with timer("ctca.projection.prepare"):
        normalized_master = F.normalize(master_embeddings.to(device).float(), dim=-1)
        rotation_on_device = rotation.to(device).float()
        projected = torch.zeros(
            spatial_probabilities.shape[0],
            master_embeddings.shape[0],
            dtype=torch.float32,
            device=device,
        )
        probabilities = spatial_probabilities.float()
    with timer("ctca.projection.chunk_loop"):
        for start in range(0, auxiliary_embeddings.shape[0], chunk_size):
            end = min(start + chunk_size, auxiliary_embeddings.shape[0])
            aligned_chunk = auxiliary_embeddings[start:end].to(device).float()
            aligned_chunk = F.normalize(aligned_chunk @ rotation_on_device, dim=-1)
            similarities = aligned_chunk @ normalized_master.transpose(0, 1)
            vocabulary_map = torch.softmax(similarities / temperature, dim=-1)
            projected.add_(probabilities[:, start:end] @ vocabulary_map)

    with timer("ctca.projection.normalize"):
        row_mass = projected.sum(dim=-1, keepdim=True)
        return torch.where(
            row_mass > epsilon,
            projected / row_mass.clamp_min(epsilon),
            projected,
        )
