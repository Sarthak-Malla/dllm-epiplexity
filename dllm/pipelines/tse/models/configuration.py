"""Configuration for synchronized two-model TSE inference.

Run an evaluation with:
    python -m dllm.pipelines.tse.eval --model tse_llada --model_args ...
"""

from dataclasses import dataclass


@dataclass
class TSEConfig:
    """Runtime configuration for synchronized TSE inference."""

    model_a_path: str
    model_b_path: str
    model_a_device: str = "cuda:0"
    model_b_device: str = "cuda:1"
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    capture_logits: bool = False
    alpha: float = 0.5
    temperature_a: float = 1.0
    temperature_b: float = 1.0
    epsilon: float = 1e-9
    fusion_device: str = "cuda:0"
    weighting_mode: str = "static"
    weight_temperature: float = 1.0
    normalize_entropy: bool = True
    model_a_suppress_tokens: tuple[int, ...] = ()
    model_a_begin_suppress_tokens: tuple[int, ...] = ()
    model_b_suppress_tokens: tuple[int, ...] = ()
    model_b_begin_suppress_tokens: tuple[int, ...] = ()
    ctca_enabled: bool = False
    master_model: str = "a"
    ctca_cache_dir: str | None = ".cache/ctca"
    ctca_force_rebuild: bool = False
    ctca_anchor_temperature: float = 0.01
    ctca_projection_temperature: float = 0.05
    ctca_chunk_size: int = 2500
    ctca_projection_mode: str = "sparse_topk"
    ctca_projection_top_k: int = 64
    ctca_num_anchors: int | str = "auto"
    ctca_min_anchors: int = 128

    def suppression_tokens_for(self, model_id: str) -> tuple[int, ...]:
        """Return always-on suppression IDs owned by model ``a`` or ``b``."""
        if model_id not in {"a", "b"}:
            raise ValueError("model_id must be 'a' or 'b'")
        return getattr(self, f"model_{model_id}_suppress_tokens")

    def begin_suppression_tokens_for(self, model_id: str) -> tuple[int, ...]:
        """Return begin-only suppression IDs owned by model ``a`` or ``b``."""
        if model_id not in {"a", "b"}:
            raise ValueError("model_id must be 'a' or 'b'")
        return getattr(self, f"model_{model_id}_begin_suppress_tokens")

    @property
    def effective_suppress_tokens(self) -> tuple[int, ...]:
        """Resolve always-on suppression from the selected master model."""
        return self.suppression_tokens_for(self.master_model)

    @property
    def effective_begin_suppress_tokens(self) -> tuple[int, ...]:
        """Resolve begin-only suppression from the selected master model."""
        return self.begin_suppression_tokens_for(self.master_model)
