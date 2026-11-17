"""Distribution-level relative-space projection for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


RELATIVE_DISTRIBUTION_VERSION = 2


def build_relative_distribution_profiles(
    embeddings: torch.Tensor,
    anchor_ids: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Build row-stochastic token profiles in shared anchor coordinates."""
    if embeddings.ndim != 2 or anchor_ids.ndim != 1:
        raise ValueError("embeddings must be 2-D and anchor_ids must be 1-D")
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("temperature and chunk_size must be positive")
    anchors = F.normalize(
        embeddings.index_select(0, anchor_ids.to(embeddings.device)).float(),
        dim=-1,
    )
    profiles = []
    for start in range(0, embeddings.shape[0], chunk_size):
        tokens = F.normalize(embeddings[start : start + chunk_size].float(), dim=-1)
        profiles.append(torch.softmax(tokens @ anchors.T / temperature, dim=-1).half().cpu())
    return torch.cat(profiles, dim=0)


def build_relative_distribution_profile_file(
    embeddings: torch.Tensor,
    anchor_ids: torch.Tensor,
    path: Path,
    *,
    temperature: float,
    chunk_size: int,
) -> None:
    """Write full-anchor profiles in bounded vocabulary chunks to a NumPy file."""
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("temperature and chunk_size must be positive")
    anchors = F.normalize(
        embeddings.index_select(0, anchor_ids.to(embeddings.device)).float(), dim=-1
    )
    table = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16,
        shape=(embeddings.shape[0], anchor_ids.numel()),
    )
    for start in range(0, embeddings.shape[0], chunk_size):
        tokens = F.normalize(embeddings[start : start + chunk_size].float(), dim=-1)
        table[start : start + chunk_size] = torch.softmax(
            tokens @ anchors.T / temperature, dim=-1
        ).half().cpu().numpy()
    table.flush()
    del table


def project_relative_distribution(
    probabilities: torch.Tensor,
    auxiliary_profiles: torch.Tensor,
    master_profiles: torch.Tensor,
    *,
    auxiliary_special_ids: torch.Tensor,
    master_special_ids: torch.Tensor,
    shared_auxiliary_ids: torch.Tensor,
    shared_master_ids: torch.Tensor,
    exact_shared: bool,
    projection_temperature: float,
    chunk_size: int,
    iterations: int = 20,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match a probability-weighted anchor profile with a master-token mixture."""
    if probabilities.ndim != 2 or auxiliary_profiles.ndim != 2 or master_profiles.ndim != 2:
        raise ValueError("probabilities and profiles must be 2-D")
    if probabilities.shape[1] != auxiliary_profiles.shape[0]:
        raise ValueError("auxiliary vocabulary size does not match its profiles")
    if auxiliary_profiles.shape[1] != master_profiles.shape[1]:
        raise ValueError("anchor dimensions must match")
    if projection_temperature <= 0 or chunk_size < 1 or iterations < 1:
        raise ValueError("projection_temperature, chunk_size, and iterations must be positive")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")

    device = probabilities.device
    batch_size, _ = probabilities.shape
    master_vocab_size = master_profiles.shape[0]
    if master_vocab_size < 1:
        raise ValueError("master vocabulary must not be empty")
    source = probabilities.float().clone()
    output = torch.zeros(batch_size, master_vocab_size, device=device, dtype=torch.float32)
    auxiliary_special_ids = auxiliary_special_ids.to(device=device, dtype=torch.long)
    master_special_ids = master_special_ids.to(device=device, dtype=torch.long)
    auxiliary_special_ids = auxiliary_special_ids[
        (auxiliary_special_ids >= 0) & (auxiliary_special_ids < source.shape[1])
    ]
    master_special_ids = master_special_ids[
        (master_special_ids >= 0) & (master_special_ids < master_vocab_size)
    ]
    shared_auxiliary_ids = shared_auxiliary_ids.to(device=device, dtype=torch.long)
    shared_master_ids = shared_master_ids.to(device=device, dtype=torch.long)

    filtered_mass = source.index_select(1, auxiliary_special_ids).sum(dim=1)
    source[:, auxiliary_special_ids] = 0
    ordinary_mass = source.sum(dim=1, keepdim=True)
    source = torch.where(ordinary_mass > 0, source / ordinary_mass.clamp_min(1e-12), source)
    if exact_shared and shared_auxiliary_ids.numel():
        output.scatter_add_(
            1,
            shared_master_ids.unsqueeze(0).expand(batch_size, -1),
            source.index_select(1, shared_auxiliary_ids),
        )
        source[:, shared_auxiliary_ids] = 0
    residual_mass = source.sum(dim=1)
    # A tiny softmax tail cannot change the chosen token meaningfully, but
    # would otherwise trigger an expensive inverse solve at every position.
    active = residual_mass > 1e-4
    fit = torch.zeros(batch_size, device=device, dtype=torch.float32)
    candidate_count = torch.zeros(batch_size, device=device, dtype=torch.int32)
    top_source_coverage = torch.zeros(batch_size, device=device, dtype=torch.float32)

    if active.any():
        active_indices = active.nonzero(as_tuple=True)[0]
        residual = source.index_select(0, active_indices)
        target = torch.zeros(residual.shape[0], auxiliary_profiles.shape[1], device=device)
        for start in range(0, residual.shape[1], chunk_size):
            end = min(start + chunk_size, residual.shape[1])
            profiles = auxiliary_profiles[start:end].to(device=device, dtype=torch.float32)
            target += residual[:, start:end] @ profiles
        target /= residual_mass.index_select(0, active_indices).unsqueeze(1)

        # Global mixture matches and token-specific neighbors keep multimodal
        # distributions from losing one of their high-probability components.
        top_values, top_source = torch.topk(residual, k=min(8, residual.shape[1]), dim=1)
        meaningful_source = top_values / residual_mass.index_select(0, active_indices)[:, None] >= 0.05
        top_source_coverage[active_indices] = (
            (top_values * meaningful_source).sum(1)
            / residual_mass.index_select(0, active_indices)
        )
        unique_source = torch.unique(top_source[meaningful_source])
        source_profiles = auxiliary_profiles.index_select(
            0, unique_source.to(auxiliary_profiles.device)
        ).to(device=device, dtype=torch.float32)
        queries = torch.cat([target, source_profiles])
        query_norms = queries.norm(dim=1).clamp_min(1e-12)
        global_count = min(256, master_vocab_size)
        best_values = torch.full((queries.shape[0], global_count), -torch.inf, device=device)
        best_indices = torch.zeros((queries.shape[0], global_count), dtype=torch.long, device=device)
        for start in range(0, master_vocab_size, chunk_size):
            end = min(start + chunk_size, master_vocab_size)
            profiles = master_profiles[start:end].to(device=device, dtype=torch.float32)
            norms = profiles.norm(dim=1).clamp_min(1e-12)
            scores = (queries @ profiles.T) / (
                query_norms[:, None] * norms[None, :]
            )
            local_special = master_special_ids[
                (master_special_ids >= start) & (master_special_ids < end)
            ] - start
            scores[:, local_special] = -torch.inf
            count = min(global_count, end - start)
            values, indices = torch.topk(scores, k=count, dim=1)
            values = torch.cat([best_values, values], dim=1)
            indices = torch.cat([best_indices, indices + start], dim=1)
            best_values, order = torch.topk(values, k=global_count, dim=1)
            best_indices = indices.gather(1, order)

        candidates = []
        source_lookup = {int(token): row for row, token in enumerate(unique_source.tolist())}
        for row in range(residual.shape[0]):
            ids = best_indices[row, torch.isfinite(best_values[row])].tolist()
            for token, selected in zip(top_source[row].tolist(), meaningful_source[row].tolist()):
                if not selected:
                    continue
                neighbor_row = residual.shape[0] + source_lookup[token]
                valid = torch.isfinite(best_values[neighbor_row, :32])
                ids.extend(best_indices[neighbor_row, :32][valid].tolist())
            seen = set()
            distinct = [i for i in ids if not (i in seen or seen.add(i))]
            candidates.append(distinct[:512])
        if not all(candidates):
            raise ValueError("no ordinary master tokens are available for projection")
        width = max(map(len, candidates))
        candidate_ids = torch.zeros(len(candidates), width, device=device, dtype=torch.long)
        candidate_mask = torch.zeros(len(candidates), width, device=device, dtype=torch.bool)
        for row, ids in enumerate(candidates):
            candidate_ids[row, : len(ids)] = torch.tensor(ids, device=device)
            candidate_mask[row, : len(ids)] = True
        # Fit one position at a time: [batch, candidates, anchors] would grow
        # with the full anchor count even though each position is independent.
        for row in range(len(candidates)):
            ids = candidate_ids[row, candidate_mask[row]]
            profiles = master_profiles.index_select(
                0, ids.to(master_profiles.device)
            ).to(device=device, dtype=torch.float32)
            similarities = F.cosine_similarity(target[row][None, :], profiles, dim=-1)
            weights = torch.softmax(similarities / projection_temperature, dim=0)
            initial_weights = weights.clone()
            for _ in range(iterations):
                mixture = weights @ profiles
                ratio = target[row] / mixture.clamp_min(1e-12)
                update = profiles @ ratio
                updated = weights * update
                updated_mass = updated.sum()
                weights = torch.where(
                    updated_mass > 1e-12,
                    updated / updated_mass.clamp_min(1e-12),
                    initial_weights,
                )
            mixture = weights @ profiles
            fit[active_indices[row]] = (
                target[row] * (target[row].clamp_min(1e-12).log()
                - mixture.clamp_min(1e-12).log())
            ).sum()
            output[active_indices[row]].scatter_add_(
                0, ids, weights * residual_mass[active_indices[row]]
            )
        candidate_count[active_indices] = candidate_mask.sum(1).int()

    row_mass = output.sum(dim=1, keepdim=True)
    output = torch.where(row_mass > 0, output / row_mass.clamp_min(1e-12), output)
    diagnostics = {
        "fit_kl": fit.detach().cpu(),
        "candidate_count": candidate_count.detach().cpu(),
        "filtered_special_mass": filtered_mass.detach().cpu(),
        "residual_mass": residual_mass.detach().cpu(),
        "top_source_coverage": top_source_coverage.detach().cpu(),
    }
    return (output, diagnostics) if return_diagnostics else output
