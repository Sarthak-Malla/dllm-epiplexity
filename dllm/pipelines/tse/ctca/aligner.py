"""High-level star-topology CTCA orchestration.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from dataclasses import dataclass

import torch

from .cache import CTCACacheManager, tokenizer_vocab_signature
from .canvas import build_canvas_overlap_matrix, spatial_warp_probabilities
from .projection import (
    TokenMatches,
    match_decoded_tokens,
    normalize_projection_rows,
    projection_token_mask,
    project_vocab_sparse_topk,
)
from .relative import (
    RELATIVE_ANCHOR_VERSION,
    RelativeAnchorSelection,
    build_sparse_topk_relative_anchor_projection,
    collect_relative_anchor_ids,
    project_vocab_relative_exact,
)
from dllm.pipelines.tse.utils import timer


PROJECTION_VERSION = 42


@dataclass
class AuxiliaryRegistration:
    """An auxiliary model projected into the master's vocabulary."""

    tokenizer: object
    embeddings: torch.Tensor
    anchors: RelativeAnchorSelection
    matches: TokenMatches
    anchor_temperature: float
    temperature: float
    chunk_size: int
    projection_mode: str
    projection_top_k: int
    sparse_topk_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]]
    sparse_topk_rebuilt_by_device: dict[str, bool]


class CrossTokenizerAligner:
    """Project one or more auxiliary distributions onto a master canvas."""

    def __init__(
        self,
        master_tokenizer,
        master_embeddings: torch.Tensor,
        *,
        master_id: str,
        cache_dir: str | None = ".cache/ctca",
        force_rebuild: bool = True,
    ) -> None:
        if master_embeddings.ndim != 2:
            raise ValueError("master_embeddings must have shape [vocab_size, hidden_size]")
        self.master_tokenizer = master_tokenizer
        self.master_embeddings = master_embeddings.detach()
        self.master_token_mask = projection_token_mask(
            master_tokenizer, master_embeddings.shape[0]
        )
        self.master_id = master_id
        self.cache = CTCACacheManager(cache_dir, force_rebuild=force_rebuild)
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
        """Register direct matches and the anchors needed for fallback."""
        if auxiliary_id in self.auxiliary_models:
            raise ValueError(f"auxiliary model is already registered: {auxiliary_id}")
        if anchor_temperature <= 0 or temperature <= 0 or chunk_size < 1:
            raise ValueError("temperatures and chunk_size must be positive")
        if projection_mode not in {"exact", "sparse_topk"}:
            raise ValueError("projection_mode must be 'exact' or 'sparse_topk'")
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
        with timer("ctca.projection.match_decoded_tokens"):
            matches = match_decoded_tokens(
                auxiliary_tokenizer,
                self.master_tokenizer,
                auxiliary_vocab_size=auxiliary_embeddings.shape[0],
                master_vocab_size=self.master_embeddings.shape[0],
            )
        self.auxiliary_models[auxiliary_id] = AuxiliaryRegistration(
            tokenizer=auxiliary_tokenizer,
            embeddings=auxiliary_embeddings.detach(),
            anchors=anchors,
            matches=matches,
            anchor_temperature=anchor_temperature,
            temperature=temperature,
            chunk_size=chunk_size,
            projection_mode=projection_mode,
            projection_top_k=projection_top_k,
            sparse_topk_by_device={},
            sparse_topk_rebuilt_by_device={},
        )

    def _sparse_topk_projection(
        self,
        auxiliary_id: str,
        registration: AuxiliaryRegistration,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = registration.sparse_topk_by_device.get(key)
        if cached is None:
            metadata = self._projection_cache_metadata(auxiliary_id, registration)
            with timer("ctca.projection.cache_sparse_topk"):
                tensors, rebuilt = self.cache.get_or_create_tensors(
                    metadata,
                    artifact=f"decoded_relative_sparse_topk_v{PROJECTION_VERSION}",
                    builder=lambda: self._build_sparse_tables(registration, device),
                    required_keys=("top_indices", "top_weights"),
                )
            cached = (
                tensors["top_indices"].to(device),
                tensors["top_weights"].to(device),
            )
            registration.sparse_topk_by_device[key] = cached
            registration.sparse_topk_rebuilt_by_device[key] = rebuilt
        return cached

    def _build_sparse_tables(
        self, registration: AuxiliaryRegistration, device: torch.device
    ) -> dict[str, torch.Tensor]:
        unmatched = self._unmatched_embeddings(registration)
        if unmatched.shape[0]:
            auxiliary_anchors, master_anchors = self._anchor_embeddings(registration)
            indices, weights = build_sparse_topk_relative_anchor_projection(
                unmatched.to(device),
                auxiliary_anchors.to(device),
                self.master_embeddings.to(device),
                master_anchors.to(device),
                top_k=registration.projection_top_k,
                anchor_temperature=registration.anchor_temperature,
                projection_temperature=registration.temperature,
                chunk_size=registration.chunk_size,
                master_token_mask=self.master_token_mask,
            )
        else:
            indices = torch.empty((0, 0), dtype=torch.long)
            weights = torch.empty((0, 0), dtype=torch.float32)
        return {"top_indices": indices, "top_weights": weights}

    @staticmethod
    def _unmatched_embeddings(registration: AuxiliaryRegistration) -> torch.Tensor:
        return registration.embeddings.index_select(
            0, registration.matches.unmatched_ids.to(registration.embeddings.device)
        )

    def _anchor_embeddings(
        self, registration: AuxiliaryRegistration
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = registration.anchors
        return (
            registration.embeddings.index_select(
                0, anchors.auxiliary_anchor_ids.to(registration.embeddings.device)
            ),
            self.master_embeddings.index_select(
                0, anchors.master_anchor_ids.to(self.master_embeddings.device)
            ),
        )

    def _projection_cache_metadata(
        self,
        auxiliary_id: str,
        registration: AuxiliaryRegistration,
    ) -> dict:
        return {
            "projection_version": PROJECTION_VERSION,
            "relative_anchor_version": RELATIVE_ANCHOR_VERSION,
            "auxiliary_id": auxiliary_id,
            "master_id": self.master_id,
            "auxiliary_vocab": tokenizer_vocab_signature(registration.tokenizer),
            "master_vocab": tokenizer_vocab_signature(self.master_tokenizer),
            "auxiliary_shape": list(registration.embeddings.shape),
            "master_shape": list(self.master_embeddings.shape),
            "auxiliary_dtype": str(registration.embeddings.dtype),
            "master_dtype": str(self.master_embeddings.dtype),
            "direct_matches": self.cache.signature(
                {
                    "auxiliary_ids": registration.matches.auxiliary_ids.tolist(),
                    "master_ids": registration.matches.master_ids.tolist(),
                }
            ),
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
            "projection_version": PROJECTION_VERSION,
            "total_anchors": registration.anchors.total_anchors,
            "selected_anchors": registration.anchors.selected_anchors,
            "direct_token_matches": registration.matches.auxiliary_ids.numel(),
            "first_piece_matches": registration.matches.prefix_master_ids.numel(),
            "relative_projection_rows": registration.matches.unmatched_ids.numel(),
            "projection_mode": registration.projection_mode,
            "projection_top_k": registration.projection_top_k,
            "anchor_temperature": registration.anchor_temperature,
            "projection_temperature": registration.temperature,
            "sparse_topk_cache_rebuilt": dict(registration.sparse_topk_rebuilt_by_device),
        }

    def warm_projection_cache(self, auxiliary_id: str, device: torch.device) -> None:
        """Build or load cached projection state before decoding."""
        registration = self.auxiliary_models[auxiliary_id]
        if registration.projection_mode == "sparse_topk":
            self._sparse_topk_projection(auxiliary_id, registration, device)

    def project_model_probabilities(
        self,
        auxiliary_id: str,
        probabilities_aux: torch.Tensor,
        offsets_aux,
        offsets_master,
        *,
        overlap_matrix: torch.Tensor | None = None,
        masked_slots_to_right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse spatial warping and chunked vocabulary projection."""
        spatial = self.spatial_warp_model_probabilities(
            auxiliary_id,
            probabilities_aux,
            offsets_aux,
            offsets_master,
            overlap_matrix=overlap_matrix,
        )
        return self.project_spatial_probabilities(
            auxiliary_id, spatial, masked_slots_to_right=masked_slots_to_right
        )

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
        *,
        masked_slots_to_right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project aligned rows, using first pieces when consecutive right slots fit."""
        try:
            registration = self.auxiliary_models[auxiliary_id]
        except KeyError as error:
            raise KeyError(f"unknown CTCA auxiliary model: {auxiliary_id}") from error
        probabilities = spatial_probabilities
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != registration.embeddings.shape[0]
        ):
            raise ValueError("spatial_probabilities must have shape [N_master, V_aux]")
        if (probabilities < 0).any() or not torch.isfinite(probabilities).all():
            raise ValueError("spatial_probabilities must be finite and non-negative")
        matches = registration.matches
        device = probabilities.device
        fallback = probabilities.index_select(1, matches.unmatched_ids.to(device))
        prefix_mass = None
        if masked_slots_to_right is not None:
            if (
                masked_slots_to_right.shape != (probabilities.shape[0],)
                or masked_slots_to_right.dtype not in (torch.int32, torch.int64)
                or (masked_slots_to_right < 0).any()
            ):
                raise ValueError(
                    "masked_slots_to_right must give a non-negative integer for each row"
                )
            if matches.prefix_columns.numel():
                columns = matches.prefix_columns.to(device)
                prefix_mass = fallback.index_select(1, columns)
                prefix_mass.masked_fill_(
                    masked_slots_to_right.to(device).unsqueeze(1)
                    < matches.required_right_masks.to(device),
                    0,
                )
                fallback.index_add_(1, columns, prefix_mass, alpha=-1)

        with timer("ctca.aligner.vocab_projection"):
            if registration.projection_mode == "sparse_topk":
                top_indices, top_weights = self._sparse_topk_projection(
                    auxiliary_id, registration, device
                )
                projected = project_vocab_sparse_topk(
                    fallback,
                    top_indices,
                    top_weights,
                    master_vocab_size=self.master_embeddings.shape[0],
                    chunk_size=registration.chunk_size,
                    normalize=False,
                )
            else:
                auxiliary_anchors, master_anchors = self._anchor_embeddings(registration)
                projected = project_vocab_relative_exact(
                    fallback,
                    self._unmatched_embeddings(registration),
                    auxiliary_anchors,
                    self.master_embeddings,
                    master_anchors,
                    anchor_temperature=registration.anchor_temperature,
                    projection_temperature=registration.temperature,
                    chunk_size=registration.chunk_size,
                    normalize=False,
                    master_token_mask=self.master_token_mask,
                )
            projected.index_add_(
                1,
                matches.master_ids.to(device),
                probabilities.index_select(1, matches.auxiliary_ids.to(device)).float(),
            )
            if prefix_mass is not None:
                projected.index_add_(
                    1, matches.prefix_master_ids.to(device), prefix_mass.float()
                )
            return normalize_projection_rows(projected)
