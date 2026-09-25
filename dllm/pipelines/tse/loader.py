"""Model and tokenizer loading for synchronized TSE inference."""

from dataclasses import dataclass

import transformers

from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_model, get_tokenizer

from .models import TSEConfig


@dataclass
class TSEModels:
    model_a: transformers.PreTrainedModel
    model_b: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer


def _device_map(device: str):
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}
    if device == "cuda":
        return {"": 0}
    return None


def load_tse_models(config: TSEConfig) -> TSEModels:
    """Load both models with explicit placement and one canonical tokenizer."""
    model_a_args = ModelArguments(
        model_name_or_path=config.model_a_path,
        dtype=config.dtype,
    )
    model_b_args = ModelArguments(
        model_name_or_path=config.model_b_path,
        dtype=config.dtype,
    )

    model_a = get_model(
        model_a_args,
        device_map=_device_map(config.model_a_device),
    )
    model_b = get_model(
        model_b_args,
        device_map=_device_map(config.model_b_device),
    )

    tokenizer = get_tokenizer(model_b_args)
    _validate_compatibility(model_a, model_b, tokenizer)
    return TSEModels(model_a=model_a, model_b=model_b, tokenizer=tokenizer)


def _validate_compatibility(model_a, model_b, tokenizer) -> None:
    vocab_a = int(model_a.config.vocab_size)
    vocab_b = int(model_b.config.vocab_size)
    if vocab_a != vocab_b:
        raise ValueError(
            "TSE models must have equal config vocab sizes: "
            f"{vocab_a} != {vocab_b}"
        )

    if tokenizer.mask_token_id is None:
        raise ValueError("The shared TSE tokenizer must define mask_token_id")

    if tokenizer.mask_token_id >= vocab_a:
        raise ValueError(
            f"mask_token_id {tokenizer.mask_token_id} exceeds vocab size {vocab_a}"
        )

    if hasattr(model_a, "get_output_embeddings"):
        output_a = model_a.get_output_embeddings()
        if output_a is not None and output_a.out_features != vocab_a:
            raise ValueError("Model A output vocabulary does not match its config")
    if hasattr(model_b, "get_output_embeddings"):
        output_b = model_b.get_output_embeddings()
        if output_b is not None and output_b.out_features != vocab_b:
            raise ValueError("Model B output vocabulary does not match its config")