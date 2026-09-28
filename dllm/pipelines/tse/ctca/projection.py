"""Memory-bounded vocabulary projection for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

import torch
import torch.nn.functional as F


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
    normalized_master = F.normalize(master_embeddings.to(device).float(), dim=-1)
    rotation_on_device = rotation.to(device).float()
    projected = torch.zeros(
        spatial_probabilities.shape[0],
        master_embeddings.shape[0],
        dtype=torch.float32,
        device=device,
    )
    probabilities = spatial_probabilities.float()
    for start in range(0, auxiliary_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, auxiliary_embeddings.shape[0])
        aligned_chunk = auxiliary_embeddings[start:end].to(device).float()
        aligned_chunk = F.normalize(aligned_chunk @ rotation_on_device, dim=-1)
        similarities = aligned_chunk @ normalized_master.transpose(0, 1)
        vocabulary_map = torch.softmax(similarities / temperature, dim=-1)
        projected.add_(probabilities[:, start:end] @ vocabulary_map)

    row_mass = projected.sum(dim=-1, keepdim=True)
    return torch.where(
        row_mass > epsilon,
        projected / row_mass.clamp_min(epsilon),
        projected,
    )
