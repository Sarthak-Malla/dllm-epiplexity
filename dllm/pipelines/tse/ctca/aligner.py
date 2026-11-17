"""High-level star-topology CTCA orchestration.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

import torch

from .cache import CTCACacheManager, tokenizer_vocab_signature
from .canvas import build_canvas_overlap_matrix, spatial_warp_probabilities
from .projection import (
    project_vocab_sparse_topk,
)
from .distribution_projection import (
    RELATIVE_DISTRIBUTION_VERSION,
    build_relative_distribution_profile_file,
    build_relative_distribution_profiles,
    project_relative_distribution,
)
from .relative import (
    RELATIVE_ANCHOR_VERSION,
    RelativeAnchorSelection,
    build_sparse_topk_relative_anchor_projection,
    collect_relative_anchor_ids,
    preserve_shared_token_rows,
    project_vocab_relative_exact,
)
from dllm.pipelines.tse.utils import timer


DISTRIBUTION_DENSE_ANCHOR_LIMIT = 2048


@dataclass
class AuxiliaryRegistration:
    """An auxiliary model projected into the master's vocabulary."""

    tokenizer: object
    embeddings: torch.Tensor
    anchors: RelativeAnchorSelection
    anchor_temperature: float
    temperature: float
    chunk_size: int
    projection_mode: str
    projection_top_k: int
    sparse_topk_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]]
    sparse_topk_rebuilt_by_device: dict[str, bool]
    distribution_profiles_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]]
    distribution_profiles_rebuilt_by_device: dict[str, bool]
    shared_ordinary_ids: tuple[torch.Tensor, torch.Tensor] | None


class CrossTokenizerAligner:
    """Project one or more auxiliary distributions onto a master canvas."""

    def __init__(
        self,
        master_tokenizer,
        master_embeddings: torch.Tensor,
        *,
        master_id: str,
        cache_dir: str | None = ".cache/ctca",
        force_rebuild: bool = False,
        shared_token_policy: str = "exact_shared",
    ) -> None:
        if master_embeddings.ndim != 2:
            raise ValueError("master_embeddings must have shape [vocab_size, hidden_size]")
        self.master_tokenizer = master_tokenizer
        self.master_embeddings = master_embeddings.detach()
        self.master_id = master_id
        if shared_token_policy not in {"relative", "exact_shared"}:
            raise ValueError("shared_token_policy must be 'relative' or 'exact_shared'")
        self.shared_token_policy = shared_token_policy
        self.cache = CTCACacheManager(cache_dir, force_rebuild=force_rebuild)
        self._temporary_profile_dir = None
        self.auxiliary_models: dict[str, AuxiliaryRegistration] = {}

    def register_auxiliary_model(
        self,
        auxiliary_id: str,
        auxiliary_tokenizer,
        auxiliary_embeddings: torch.Tensor,
        *,
        anchor_temperature: float = 0.01,
        temperature: float = 0.05,
        chunk_size: int = 2500,
        num_anchors: int | str = "auto",
        min_anchors: int = 128,
        projection_mode: str = "sparse_topk",
        projection_top_k: int = 64,
    ) -> None:
        """Register an auxiliary model for relative-anchor projection."""
        if auxiliary_id in self.auxiliary_models:
            raise ValueError(f"auxiliary model is already registered: {auxiliary_id}")
        if anchor_temperature <= 0 or temperature <= 0 or chunk_size < 1:
            raise ValueError("temperatures and chunk_size must be positive")
        if projection_mode not in {"exact", "sparse_topk", "relative_distribution"}:
            raise ValueError("projection_mode must be 'exact', 'sparse_topk', or 'relative_distribution'")
        if projection_top_k < 1:
            raise ValueError("projection_top_k must be positive")
        with timer("ctca.relative.collect_anchors"):
            anchors = collect_relative_anchor_ids(
                auxiliary_tokenizer,
                self.master_tokenizer,
                auxiliary_vocab_size=auxiliary_embeddings.shape[0],
                master_vocab_size=self.master_embeddings.shape[0],
                num_anchors=num_anchors,
                min_anchors=min_anchors,
            )
        self.auxiliary_models[auxiliary_id] = AuxiliaryRegistration(
            tokenizer=auxiliary_tokenizer,
            embeddings=auxiliary_embeddings.detach(),
            anchors=anchors,
            anchor_temperature=anchor_temperature,
            temperature=temperature,
            chunk_size=chunk_size,
            projection_mode=projection_mode,
            projection_top_k=projection_top_k,
            sparse_topk_by_device={},
            sparse_topk_rebuilt_by_device={},
            distribution_profiles_by_device={},
            distribution_profiles_rebuilt_by_device={},
            shared_ordinary_ids=None,
        )

    def _distribution_profiles(
        self, auxiliary_id: str, registration: AuxiliaryRegistration, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = registration.distribution_profiles_by_device.get(key)
        if cached is None:
            metadata = {
                **self._relative_cache_metadata(auxiliary_id, registration),
                "algorithm": "relative_distribution",
                "version": RELATIVE_DISTRIBUTION_VERSION,
                "selected_auxiliary_anchor_ids": registration.anchors.auxiliary_anchor_ids.tolist(),
                "selected_master_anchor_ids": registration.anchors.master_anchor_ids.tolist(),
            }

            if registration.anchors.selected_anchors > DISTRIBUTION_DENSE_ANCHOR_LIMIT:
                cached = self._chunked_distribution_profiles(metadata, registration)
                registration.distribution_profiles_by_device[key] = cached
                return cached

            def build_tables() -> dict[str, torch.Tensor]:
                return {
                    "auxiliary_profiles": build_relative_distribution_profiles(
                        registration.embeddings,
                        registration.anchors.auxiliary_anchor_ids,
                        temperature=registration.anchor_temperature,
                        chunk_size=registration.chunk_size,
                    ),
                    "master_profiles": build_relative_distribution_profiles(
                        self.master_embeddings,
                        registration.anchors.master_anchor_ids,
                        temperature=registration.anchor_temperature,
                        chunk_size=registration.chunk_size,
                    ),
                }

            with timer("ctca.relative.cache_distribution_profiles"):
                tensors, rebuilt = self.cache.get_or_create_tensors(
                    metadata,
                    artifact="relative_distribution_profiles",
                    builder=build_tables,
                    required_keys=("auxiliary_profiles", "master_profiles"),
                )
            cached = (
                tensors["auxiliary_profiles"].to(device),
                tensors["master_profiles"].to(device),
            )
            registration.distribution_profiles_by_device[key] = cached
            registration.distribution_profiles_rebuilt_by_device[key] = rebuilt
        return cached

    def _chunked_distribution_profiles(
        self, metadata: dict, registration: AuxiliaryRegistration
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory-map full profiles; never load both full tables onto a GPU."""
        if self.cache.cache_dir is None:
            if self._temporary_profile_dir is None:
                self._temporary_profile_dir = tempfile.TemporaryDirectory(prefix="ctca-profiles-")
            cache_dir = Path(self._temporary_profile_dir.name)
        else:
            cache_dir = self.cache.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        signature = self.cache.signature(metadata)
        paths = [
            cache_dir / f"relative_distribution_aux_{signature}.npy",
            cache_dir / f"relative_distribution_master_{signature}.npy",
        ]
        sources = [
            (registration.embeddings, registration.anchors.auxiliary_anchor_ids),
            (self.master_embeddings, registration.anchors.master_anchor_ids),
        ]
        rebuilt = False
        profiles = []
        for path, (embeddings, anchor_ids) in zip(paths, sources):
            expected = (embeddings.shape[0], anchor_ids.numel())
            if path.exists() and not self.cache.force_rebuild:
                table = np.load(path, mmap_mode="r+")
                if table.shape != expected or table.dtype != np.float16:
                    raise ValueError(f"Malformed CTCA profile cache: {path}")
            else:
                rebuilt = True
                temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
                try:
                    build_relative_distribution_profile_file(
                        embeddings, anchor_ids, temporary,
                        temperature=registration.anchor_temperature,
                        chunk_size=min(registration.chunk_size, 512),
                    )
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                table = np.load(path, mmap_mode="r+")
            profiles.append(torch.from_numpy(table))
        registration.distribution_profiles_rebuilt_by_device["chunked"] = rebuilt
        return profiles[0], profiles[1]

    def _shared_ordinary_ids(self, registration: AuxiliaryRegistration) -> tuple[torch.Tensor, torch.Tensor]:
        if registration.shared_ordinary_ids is not None:
            return registration.shared_ordinary_ids
        auxiliary_special = set(getattr(registration.tokenizer, "all_special_ids", []))
        master_special = set(getattr(self.master_tokenizer, "all_special_ids", []))
        master_vocab = self.master_tokenizer.get_vocab()
        pairs = [
            (int(auxiliary_id), int(master_vocab[token]))
            for token, auxiliary_id in registration.tokenizer.get_vocab().items()
            if token in master_vocab
            and 0 <= int(auxiliary_id) < registration.embeddings.shape[0]
            and 0 <= int(master_vocab[token]) < self.master_embeddings.shape[0]
            and int(auxiliary_id) not in auxiliary_special
            and int(master_vocab[token]) not in master_special
        ]
        pairs.sort()
        registration.shared_ordinary_ids = (
            torch.tensor([p[0] for p in pairs], dtype=torch.long),
            torch.tensor([p[1] for p in pairs], dtype=torch.long),
        )
        return registration.shared_ordinary_ids

    def _sparse_topk_projection(
        self,
        auxiliary_id: str,
        registration: AuxiliaryRegistration,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = registration.sparse_topk_by_device.get(key)
        if cached is None:
            metadata = self._relative_cache_metadata(auxiliary_id, registration)

            def build_tables() -> dict[str, torch.Tensor]:
                auxiliary_anchor_ids = registration.anchors.auxiliary_anchor_ids.to(
                    registration.embeddings.device
                )
                master_anchor_ids = registration.anchors.master_anchor_ids.to(
                    self.master_embeddings.device
                )
                top_indices, top_weights = build_sparse_topk_relative_anchor_projection(
                    registration.embeddings.to(device),
                    registration.embeddings.index_select(0, auxiliary_anchor_ids).to(device),
                    self.master_embeddings.to(device),
                    self.master_embeddings.index_select(0, master_anchor_ids).to(device),
                    top_k=registration.projection_top_k,
                    anchor_temperature=registration.anchor_temperature,
                    projection_temperature=registration.temperature,
                    chunk_size=registration.chunk_size,
                )
                return {"top_indices": top_indices, "top_weights": top_weights}

            with timer("ctca.relative.cache_sparse_topk"):
                tensors, rebuilt = self.cache.get_or_create_tensors(
                    metadata,
                    artifact="relative_sparse_topk",
                    builder=build_tables,
                    required_keys=("top_indices", "top_weights"),
                )
            cached = (tensors["top_indices"], tensors["top_weights"])
            if self.shared_token_policy == "exact_shared":
                cached = preserve_shared_token_rows(
                    *cached,
                    registration.tokenizer,
                    self.master_tokenizer,
                    master_vocab_size=self.master_embeddings.shape[0],
                )
            registration.sparse_topk_by_device[key] = cached
            registration.sparse_topk_rebuilt_by_device[key] = rebuilt
        return cached

    def _relative_cache_metadata(
        self,
        auxiliary_id: str,
        registration: AuxiliaryRegistration,
    ) -> dict:
        return {
            "algorithm": "relative_anchor_sparse_topk",
            "version": RELATIVE_ANCHOR_VERSION,
            "auxiliary_id": auxiliary_id,
            "master_id": self.master_id,
            "auxiliary_vocab": tokenizer_vocab_signature(registration.tokenizer),
            "master_vocab": tokenizer_vocab_signature(self.master_tokenizer),
            "auxiliary_shape": list(registration.embeddings.shape),
            "master_shape": list(self.master_embeddings.shape),
            "total_anchors": registration.anchors.total_anchors,
            "selected_anchors": registration.anchors.selected_anchors,
            "anchor_temperature": registration.anchor_temperature,
            "projection_temperature": registration.temperature,
            "projection_top_k": registration.projection_top_k,
        }

    def projection_diagnostics(self, auxiliary_id: str) -> dict:
        """Return lightweight CTCA projection metadata for logging."""
        registration = self.auxiliary_models[auxiliary_id]
        return {
            "total_anchors": registration.anchors.total_anchors,
            "selected_anchors": registration.anchors.selected_anchors,
            "projection_mode": registration.projection_mode,
            "projection_top_k": registration.projection_top_k,
            "anchor_temperature": registration.anchor_temperature,
            "projection_temperature": registration.temperature,
            "sparse_topk_cache_rebuilt": dict(registration.sparse_topk_rebuilt_by_device),
            "relative_distribution_cache_rebuilt": dict(registration.distribution_profiles_rebuilt_by_device),
            "shared_token_policy": (
                self.shared_token_policy
                if registration.projection_mode in {"sparse_topk", "relative_distribution"}
                else "relative"
            ),
        }

    def warm_projection_cache(self, auxiliary_id: str, device: torch.device) -> None:
        """Build or load cached projection state before decoding."""
        registration = self.auxiliary_models[auxiliary_id]
        if registration.projection_mode == "sparse_topk":
            self._sparse_topk_projection(auxiliary_id, registration, device)
        elif registration.projection_mode == "relative_distribution":
            self._distribution_profiles(auxiliary_id, registration, device)

    def project_model_probabilities(
        self,
        auxiliary_id: str,
        probabilities_aux: torch.Tensor,
        offsets_aux,
        offsets_master,
        *,
        overlap_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse spatial warping and chunked vocabulary projection."""
        spatial = self.spatial_warp_model_probabilities(
            auxiliary_id,
            probabilities_aux,
            offsets_aux,
            offsets_master,
            overlap_matrix=overlap_matrix,
        )
        return self.project_spatial_probabilities(auxiliary_id, spatial)

    def spatial_warp_model_probabilities(
        self,
        auxiliary_id: str,
        probabilities_aux: torch.Tensor,
        offsets_aux,
        offsets_master,
        *,
        overlap_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Warp auxiliary-token probabilities onto master canvas positions."""
        try:
            self.auxiliary_models[auxiliary_id]
        except KeyError as error:
            raise KeyError(f"unknown CTCA auxiliary model: {auxiliary_id}") from error
        if overlap_matrix is None:
            with timer("ctca.aligner.build_overlap"):
                overlap = build_canvas_overlap_matrix(
                    offsets_aux,
                    offsets_master,
                    device=probabilities_aux.device,
                    dtype=probabilities_aux.dtype,
                )
        else:
            overlap = overlap_matrix.to(
                device=probabilities_aux.device, dtype=probabilities_aux.dtype
            )
        with timer("ctca.aligner.spatial_warp"):
            return spatial_warp_probabilities(probabilities_aux, overlap)

    def project_spatial_probabilities(
        self,
        auxiliary_id: str,
        spatial_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Project already warped auxiliary probabilities into master vocabulary."""
        try:
            registration = self.auxiliary_models[auxiliary_id]
        except KeyError as error:
            raise KeyError(f"unknown CTCA auxiliary model: {auxiliary_id}") from error
        if registration.projection_mode == "relative_distribution":
            with timer("ctca.projection.prepare"):
                auxiliary_profiles, master_profiles = self._distribution_profiles(
                    auxiliary_id, registration, spatial_probabilities.device
                )
                shared_auxiliary_ids, shared_master_ids = self._shared_ordinary_ids(registration)
            with timer("ctca.aligner.vocab_projection"):
                projected, diagnostics = project_relative_distribution(
                    spatial_probabilities,
                    auxiliary_profiles,
                    master_profiles,
                    auxiliary_special_ids=torch.tensor(
                        getattr(registration.tokenizer, "all_special_ids", []), dtype=torch.long
                    ),
                    master_special_ids=torch.tensor(
                        getattr(self.master_tokenizer, "all_special_ids", []), dtype=torch.long
                    ),
                    shared_auxiliary_ids=shared_auxiliary_ids,
                    shared_master_ids=shared_master_ids,
                    exact_shared=self.shared_token_policy == "exact_shared",
                    projection_temperature=registration.temperature,
                    chunk_size=registration.chunk_size,
                    return_diagnostics=True,
                )
            self.last_distribution_diagnostics = diagnostics
            return projected
        if registration.projection_mode == "sparse_topk":
            with timer("ctca.projection.prepare"):
                top_indices, top_weights = self._sparse_topk_projection(
                    auxiliary_id, registration, spatial_probabilities.device
                )
            with timer("ctca.aligner.vocab_projection"):
                return project_vocab_sparse_topk(
                    spatial_probabilities,
                    top_indices,
                    top_weights,
                    master_vocab_size=self.master_embeddings.shape[0],
                    chunk_size=registration.chunk_size,
                )

        with timer("ctca.projection.prepare"):
            auxiliary_anchor_ids = registration.anchors.auxiliary_anchor_ids.to(
                registration.embeddings.device
            )
            master_anchor_ids = registration.anchors.master_anchor_ids.to(
                self.master_embeddings.device
            )
        with timer("ctca.aligner.vocab_projection"):
            return project_vocab_relative_exact(
                spatial_probabilities,
                registration.embeddings,
                registration.embeddings.index_select(0, auxiliary_anchor_ids),
                self.master_embeddings,
                self.master_embeddings.index_select(0, master_anchor_ids),
                anchor_temperature=registration.anchor_temperature,
                projection_temperature=registration.temperature,
                chunk_size=registration.chunk_size,
            )
