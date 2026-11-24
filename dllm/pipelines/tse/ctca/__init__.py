"""Cross-tokenizer canvas alignment utilities.

Canvas views align generation positions across tokenizers. Vocabulary projection
sends matching decoded tokens directly to master IDs. A lossless multi-token
encoding can contribute to its first piece when enough adjacent slots are masked.
Relative anchors handle the remaining probability mass.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

from .aligner import CrossTokenizerAligner
from .cache import CTCACacheManager
from .canvas import (
    CanvasRunCache,
    CachedCanvasRun,
    ModelCanvasView,
    build_canvas_overlap_matrix,
    build_model_canvas_view,
    spatial_warp_probabilities,
)
from .projection import project_vocab_sparse_topk
from .relative import (
    RelativeAnchorSelection,
    build_sparse_topk_relative_anchor_projection,
    collect_relative_anchor_ids,
    project_vocab_relative_exact,
    relative_anchor_profiles,
)

__all__ = [
    "CachedCanvasRun",
    "CanvasRunCache",
    "CTCACacheManager",
    "CrossTokenizerAligner",
    "ModelCanvasView",
    "RelativeAnchorSelection",
    "build_canvas_overlap_matrix",
    "build_model_canvas_view",
    "build_sparse_topk_relative_anchor_projection",
    "collect_relative_anchor_ids",
    "project_vocab_relative_exact",
    "project_vocab_sparse_topk",
    "relative_anchor_profiles",
    "spatial_warp_probabilities",
]
