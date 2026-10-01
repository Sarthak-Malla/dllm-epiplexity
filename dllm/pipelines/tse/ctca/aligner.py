"""High-level star-topology CTCA orchestration.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .alignment import align_embeddings_procrustes
from .cache import CTCACacheManager, tokenizer_vocab_signature
from .canvas import build_canvas_overlap_matrix, spatial_warp_probabilities
from .projection import (
    build_sparse_topk_vocab_projection,
    project_vocab_from_normalized_fused,
    project_vocab_sparse_topk,
)
from dllm.pipelines.tse.utils import timer


@dataclass
class AuxiliaryRegistration:
    """An auxiliary model projected into the master's embedding space."""

    tokenizer: object
    embeddings: torch.Tensor
    rotation: torch.Tensor
    temperature: float
    chunk_size: int
    projection_mode: str
    projection_top_k: int
    normalized_aligned_embeddings_by_device: dict[str, torch.Tensor]
    sparse_topk_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]]


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
    ) -> None:
        if master_embeddings.ndim != 2:
            raise ValueError("master_embeddings must have shape [vocab_size, hidden_size]")
        self.master_tokenizer = master_tokenizer
        self.master_embeddings = master_embeddings.detach()
        self.master_id = master_id
        self.cache = CTCACacheManager(cache_dir, force_rebuild=force_rebuild)
        self.auxiliary_models: dict[str, AuxiliaryRegistration] = {}
        self._normalized_master_by_device: dict[str, torch.Tensor] = {}

    def register_auxiliary_model(
        self,
        auxiliary_id: str,
        auxiliary_tokenizer,
        auxiliary_embeddings: torch.Tensor,
        *,
        temperature: float = 0.05,
        chunk_size: int = 2500,
        num_anchors: int = 3000,
        min_anchors: int = 128,
        projection_mode: str = "exact",
        projection_top_k: int = 64,
    ) -> None:
        """Register an auxiliary model and cache its Procrustes rotation."""
        if auxiliary_id in self.auxiliary_models:
            raise ValueError(f"auxiliary model is already registered: {auxiliary_id}")
        if temperature <= 0 or chunk_size < 1:
            raise ValueError("temperature and chunk_size must be positive")
        if projection_mode not in {"exact", "sparse_topk"}:
            raise ValueError("projection_mode must be 'exact' or 'sparse_topk'")
        if projection_top_k < 1:
            raise ValueError("projection_top_k must be positive")
        metadata = {
            "auxiliary_id": auxiliary_id,
            "master_id": self.master_id,
            "auxiliary_vocab": tokenizer_vocab_signature(auxiliary_tokenizer),
            "master_vocab": tokenizer_vocab_signature(self.master_tokenizer),
            "auxiliary_shape": list(auxiliary_embeddings.shape),
            "master_shape": list(self.master_embeddings.shape),
            "num_anchors": num_anchors,
            "min_anchors": min_anchors,
        }

        def build_rotation() -> torch.Tensor:
            return align_embeddings_procrustes(
                auxiliary_embeddings,
                self.master_embeddings,
                auxiliary_tokenizer,
                self.master_tokenizer,
                num_anchors=num_anchors,
                min_anchors=min_anchors,
            ).rotation

        with timer("ctca.aligner.get_or_create_rotation"):
            rotation = self.cache.get_or_create(metadata, build_rotation)
        self.auxiliary_models[auxiliary_id] = AuxiliaryRegistration(
            tokenizer=auxiliary_tokenizer,
            embeddings=auxiliary_embeddings.detach(),
            rotation=rotation,
            temperature=temperature,
            chunk_size=chunk_size,
            projection_mode=projection_mode,
            projection_top_k=projection_top_k,
            normalized_aligned_embeddings_by_device={},
            sparse_topk_by_device={},
        )

    def _normalized_master_embeddings(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._normalized_master_by_device.get(key)
        if cached is None:
            with timer("ctca.aligner.cache_master_embeddings"):
                cached = F.normalize(self.master_embeddings.to(device).float(), dim=-1)
            self._normalized_master_by_device[key] = cached
        return cached

    def _sparse_topk_projection(
        self,
        registration: AuxiliaryRegistration,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = registration.sparse_topk_by_device.get(key)
        if cached is None:
            normalized_auxiliary = self._normalized_aligned_auxiliary_embeddings(
                registration, device
            )
            normalized_master = self._normalized_master_embeddings(device)
            cached = build_sparse_topk_vocab_projection(
                normalized_auxiliary,
                normalized_master,
                top_k=registration.projection_top_k,
                temperature=registration.temperature,
                chunk_size=registration.chunk_size,
            )
            registration.sparse_topk_by_device[key] = cached
        return cached

    def _normalized_aligned_auxiliary_embeddings(
        self,
        registration: AuxiliaryRegistration,
        device: torch.device,
    ) -> torch.Tensor:
        key = str(device)
        cached = registration.normalized_aligned_embeddings_by_device.get(key)
        if cached is None:
            with timer("ctca.aligner.cache_aligned_auxiliary_embeddings"):
                rotation = registration.rotation.to(device).float()
                aligned = registration.embeddings.to(device).float() @ rotation
                cached = F.normalize(aligned, dim=-1)
            registration.normalized_aligned_embeddings_by_device[key] = cached
        return cached

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
        try:
            registration = self.auxiliary_models[auxiliary_id]
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
            spatial = spatial_warp_probabilities(probabilities_aux, overlap)
        if registration.projection_mode == "sparse_topk":
            with timer("ctca.projection.prepare"):
                top_indices, top_weights = self._sparse_topk_projection(
                    registration, probabilities_aux.device
                )
            with timer("ctca.aligner.vocab_projection"):
                return project_vocab_sparse_topk(
                    spatial,
                    top_indices,
                    top_weights,
                    master_vocab_size=self.master_embeddings.shape[0],
                    chunk_size=registration.chunk_size,
                )

        with timer("ctca.projection.prepare"):
            normalized_auxiliary = self._normalized_aligned_auxiliary_embeddings(
                registration, probabilities_aux.device
            )
            normalized_master = self._normalized_master_embeddings(probabilities_aux.device)
        with timer("ctca.aligner.vocab_projection"):
            return project_vocab_from_normalized_fused(
                spatial,
                normalized_auxiliary,
                normalized_master,
                temperature=registration.temperature,
                chunk_size=registration.chunk_size,
            )
