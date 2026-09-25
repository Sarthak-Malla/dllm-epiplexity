"""Configuration for synchronized two-model TSE inference."""

from dataclasses import dataclass


@dataclass
class TSEConfig:
    """Runtime configuration for the Phase 2 TSE sampler."""

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