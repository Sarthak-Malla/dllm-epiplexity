"""Configuration for synchronized two-model TSE inference."""

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