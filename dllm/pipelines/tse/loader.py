"""Model and tokenizer loading for synchronized TSE inference.

Run the focused tests with:
    pytest scripts/tests/test_tse_loader.py -v
"""

from dataclasses import dataclass

import transformers

from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_model, get_tokenizer

from .ctca.canvas import encode_with_offsets
from .ctca_sampler import CTCATSESampler
from .models import TSEConfig
from .sampler import TSESampler


@dataclass
class TSEModels:
    model_a: transformers.PreTrainedModel
    model_b: transformers.PreTrainedModel
    tokenizer_a: transformers.PreTrainedTokenizerBase
    tokenizer_b: transformers.PreTrainedTokenizerBase

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        """Return the shared tokenizer used by homogeneous TSE."""
        return self.tokenizer_b

    def model_for(self, name: str):
        if name == "a":
            return self.model_a
        if name == "b":
            return self.model_b
        raise ValueError("model name must be 'a' or 'b'")

    def tokenizer_for(self, name: str):
        if name == "a":
            return self.tokenizer_a
        if name == "b":
            return self.tokenizer_b
        raise ValueError("model name must be 'a' or 'b'")


def _device_map(device: str):
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}
    if device == "cuda":
        return {"": 0}
    return None


def load_tse_models(config: TSEConfig) -> TSEModels:
    """Load both models and their native tokenizers with explicit placement."""
    if config.model_b_path is None:
        raise ValueError("Paired TSE requires a Model B checkpoint")
    _validate_config(config)
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

    tokenizer_b = get_tokenizer(model_b_args)
    tokenizer_a = (
        tokenizer_b
        if config.model_a_path == config.model_b_path
        else get_tokenizer(model_a_args)
    )
    _validate_compatibility(
        model_a,
        model_b,
        tokenizer_a,
        tokenizer_b,
        ctca_enabled=config.ctca_enabled,
    )
    return TSEModels(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
    )


def load_tse_baseline(config: TSEConfig) -> TSESampler:
    """Load one checkpoint for the shared greedy decoding loop."""
    args = ModelArguments(model_name_or_path=config.model_a_path, dtype=config.dtype)
    model = get_model(args, device_map=_device_map(config.model_a_device))
    tokenizer = get_tokenizer(args)
    _validate_model(model, tokenizer, "A", require_mask=True, require_ctca=False)
    return TSESampler(model, None, tokenizer, config.model_a_device)


def build_tse_sampler(config: TSEConfig, models: TSEModels) -> TSESampler | CTCATSESampler:
    """Build the sampler shared by evaluation and smoke tests."""
    if not config.ctca_enabled:
        return TSESampler(
            models.model_a,
            models.model_b,
            models.tokenizer,
            config.model_a_device,
            config.model_b_device,
        )

    master = config.master_model
    auxiliary = "b" if master == "a" else "a"
    return CTCATSESampler(
        models.model_for(master),
        models.model_for(auxiliary),
        models.tokenizer_for(master),
        models.tokenizer_for(auxiliary),
        getattr(config, f"model_{master}_device"),
        getattr(config, f"model_{auxiliary}_device"),
        master_id=master,
        auxiliary_id=auxiliary,
        cache_dir=config.ctca_cache_dir,
        force_rebuild=config.ctca_force_rebuild,
        anchor_temperature=config.ctca_anchor_temperature,
        projection_temperature=config.ctca_projection_temperature,
        projection_chunk_size=config.ctca_chunk_size,
        projection_mode=config.ctca_projection_mode,
        projection_top_k=config.ctca_projection_top_k,
        num_anchors=config.ctca_num_anchors,
        min_anchors=config.ctca_min_anchors,
        master_cache_id=getattr(config, f"model_{master}_path"),
        auxiliary_cache_id=getattr(config, f"model_{auxiliary}_path"),
        fusion_device=config.fusion_device,
    )


def _validate_config(config: TSEConfig) -> None:
    if not config.ctca_enabled:
        return
    if config.master_model not in {"a", "b"}:
        raise ValueError("master_model must be 'a' or 'b'")
    if config.ctca_anchor_temperature <= 0:
        raise ValueError("ctca_anchor_temperature must be positive")
    if config.ctca_projection_temperature <= 0:
        raise ValueError("ctca_projection_temperature must be positive")
    if config.ctca_chunk_size < 1:
        raise ValueError("ctca_chunk_size must be positive")
    if config.ctca_projection_mode not in {"exact", "sparse_topk"}:
        raise ValueError("ctca_projection_mode must be 'exact' or 'sparse_topk'")
    if config.ctca_projection_top_k < 1:
        raise ValueError("ctca_projection_top_k must be positive")
    if config.ctca_min_anchors < 1:
        raise ValueError("ctca_min_anchors must be positive")
    if isinstance(config.ctca_num_anchors, str):
        if config.ctca_num_anchors.casefold() != "auto":
            try:
                config.ctca_num_anchors = int(config.ctca_num_anchors)
            except ValueError as error:
                raise ValueError("ctca_num_anchors must be positive or 'auto'") from error
    if not isinstance(config.ctca_num_anchors, str):
        if config.ctca_num_anchors < 1:
            raise ValueError("ctca_num_anchors must be positive or 'auto'")
        if config.ctca_min_anchors > config.ctca_num_anchors:
            raise ValueError("ctca_min_anchors must be <= ctca_num_anchors")


def _validate_model(
    model,
    tokenizer,
    label: str,
    *,
    require_mask: bool,
    require_ctca: bool,
) -> None:
    vocab_size = int(model.config.vocab_size)
    if require_mask and tokenizer.mask_token_id is None:
        raise ValueError(f"Model {label} tokenizer must define mask_token_id")
    if require_mask and (
        tokenizer.mask_token_id < 0 or tokenizer.mask_token_id >= vocab_size
    ):
        raise ValueError(
            f"Model {label} mask_token_id {tokenizer.mask_token_id} exceeds "
            f"vocab size {vocab_size}"
        )
    if require_ctca and (
        not hasattr(model, "get_input_embeddings")
        or model.get_input_embeddings() is None
    ):
        raise ValueError(f"Model {label} must expose input embeddings for CTCA")
    if require_ctca:
        input_embeddings = model.get_input_embeddings()
        input_size = getattr(input_embeddings, "num_embeddings", None)
        if input_size is None and hasattr(input_embeddings, "weight"):
            input_size = input_embeddings.weight.shape[0]
        if input_size != vocab_size:
            raise ValueError(
                f"Model {label} input vocabulary does not match its config"
            )

    if hasattr(model, "get_output_embeddings"):
        output = model.get_output_embeddings()
        output_size = (
            getattr(output, "out_features", None) if output is not None else None
        )
        if output_size is not None and output_size != vocab_size:
            raise ValueError(f"Model {label} output vocabulary does not match its config")

    if require_ctca:
        try:
            _, offsets = encode_with_offsets(tokenizer, "ctca")
        except (NotImplementedError, TypeError, ValueError) as error:
            raise ValueError(
                f"Model {label} tokenizer must support or permit deriving "
                "offset mappings for CTCA"
            ) from error
        if not offsets:
            raise ValueError(
                f"Model {label} tokenizer did not produce offset mappings for CTCA"
            )


def _validate_compatibility(
    model_a,
    model_b,
    tokenizer_a,
    tokenizer_b=None,
    *,
    ctca_enabled: bool = False,
) -> None:
    if tokenizer_b is None:
        tokenizer_b = tokenizer_a
    vocab_a = int(model_a.config.vocab_size)
    vocab_b = int(model_b.config.vocab_size)
    if not ctca_enabled and vocab_a != vocab_b:
        raise ValueError(
            "TSE models must have equal config vocab sizes: "
            f"{vocab_a} != {vocab_b}"
        )
    if not ctca_enabled and tokenizer_a is not tokenizer_b and (
        tokenizer_a.get_vocab() != tokenizer_b.get_vocab()
        or tokenizer_a.mask_token_id != tokenizer_b.mask_token_id
    ):
        raise ValueError("Homogeneous TSE requires matching token IDs and mask IDs")
    _validate_model(
        model_a,
        tokenizer_a,
        "A",
        require_mask=ctca_enabled,
        require_ctca=ctca_enabled,
    )
    _validate_model(
        model_b,
        tokenizer_b,
        "B",
        require_mask=True,
        require_ctca=ctca_enabled,
    )
