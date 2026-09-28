"""High-level star-topology CTCA orchestration.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass

import torch

from .alignment import align_embeddings_procrustes
from .cache import CTCACacheManager, tokenizer_vocab_signature
from .canvas import build_canvas_overlap_matrix, spatial_warp_probabilities
from .projection import project_vocab_with_rotation_fused


@dataclass
class AuxiliaryRegistration:
    """An auxiliary model projected into the master's embedding space."""

    tokenizer: object
    embeddings: torch.Tensor
    rotation: torch.Tensor
    temperature: float
    chunk_size: int


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
    ) -> None:
        """Register an auxiliary model and cache its Procrustes rotation."""
        if auxiliary_id in self.auxiliary_models:
            raise ValueError(f"auxiliary model is already registered: {auxiliary_id}")
        if temperature <= 0 or chunk_size < 1:
            raise ValueError("temperature and chunk_size must be positive")
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

        rotation = self.cache.get_or_create(metadata, build_rotation)
        self.auxiliary_models[auxiliary_id] = AuxiliaryRegistration(
            tokenizer=auxiliary_tokenizer,
            embeddings=auxiliary_embeddings.detach(),
            rotation=rotation,
            temperature=temperature,
            chunk_size=chunk_size,
        )

    def project_model_probabilities(
        self,
        auxiliary_id: str,
        probabilities_aux: torch.Tensor,
        offsets_aux,
        offsets_master,
    ) -> torch.Tensor:
        """Apply sparse spatial warping and chunked vocabulary projection."""
        try:
            registration = self.auxiliary_models[auxiliary_id]
        except KeyError as error:
            raise KeyError(f"unknown CTCA auxiliary model: {auxiliary_id}") from error
        overlap = build_canvas_overlap_matrix(
            offsets_aux,
            offsets_master,
            device=probabilities_aux.device,
            dtype=probabilities_aux.dtype,
        )
        spatial = spatial_warp_probabilities(probabilities_aux, overlap)
        return project_vocab_with_rotation_fused(
            spatial,
            registration.embeddings,
            registration.rotation,
            self.master_embeddings,
            temperature=registration.temperature,
            chunk_size=registration.chunk_size,
        )
