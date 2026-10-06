"""Cross-tokenizer canvas alignment utilities.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from .aligner import CrossTokenizerAligner
from .alignment import AlignmentResult, align_embeddings_procrustes
from .cache import CTCACacheManager
from .canvas import (
    CanvasRunCache,
    CachedCanvasRun,
    ModelCanvasView,
    build_canvas_overlap_matrix,
    build_model_canvas_view,
    spatial_warp_probabilities,
)
from .projection import (
    build_sparse_topk_vocab_projection,
    project_vocab_fused,
    project_vocab_sparse_topk,
    project_vocab_with_rotation_fused,
)

__all__ = [
    "AlignmentResult",
    "CachedCanvasRun",
    "CanvasRunCache",
    "CTCACacheManager",
    "CrossTokenizerAligner",
    "ModelCanvasView",
    "align_embeddings_procrustes",
    "build_canvas_overlap_matrix",
    "build_model_canvas_view",
    "build_sparse_topk_vocab_projection",
    "project_vocab_fused",
    "project_vocab_sparse_topk",
    "project_vocab_with_rotation_fused",
    "spatial_warp_probabilities",
]
