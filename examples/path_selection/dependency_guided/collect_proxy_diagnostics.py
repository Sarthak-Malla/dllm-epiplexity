"""
Collect one LLaDA proxy trajectory into resumable diagnostic JSONL.

On an allocated GPU with the dllm environment active:

    REPO=/home/sarthak.malla/dllm-selection-ensemble
    CACHE=/home/sarthak.malla/.cache/huggingface/hub
    MODEL=models--GSAI-ML--LLaDA-8B-Instruct
    REVISION=08b83a6feb34df1a6011b80c3c00c7563e963b07
    SCRIPT_DIR=${REPO}/examples/path_selection/dependency_guided
    OUTPUT=${REPO}/eval_results/path_selection/dependency_guided
    python "${SCRIPT_DIR}/collect_proxy_diagnostics.py" \
        --checkpoint "${CACHE}/${MODEL}/snapshots/${REVISION}" \
        --output-directory "${OUTPUT}/proxy_diagnostics/p2_3_smoke" \
        --dataset-label gsm8k \
        --split test \
        --example-id p2_2_gsm8k_manual_0 \
        --prompt "If 3 boxes each have 12 pencils, how many pencils are there?" \
        --response-length 8 \
        --mask-ratios 1.0 0.75 0.5 0.25 \
        --last-n-layers 4 \
        --top-confidence-pool-size 2 \
        --random-pool-size 2 \
        --sink-filter \
        --sink-quantile 0.99 \
        --renormalize-selected-keys \
        --zero-diagonal \
        --seed 42 \
        --device cuda:0 \
        --dtype bfloat16

The output directory contains separate configuration.json, states.jsonl, and
failures.jsonl files. Reusing it with a different configuration is rejected.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import random

import numpy as np
import torch
import transformers
from transformers import AutoModelForMaskedLM

import dllm.pipelines.llada.models  # noqa: F401
from dllm.core.samplers.diagnostic_io import (
    ProxyDiagnosticStore,
    build_proxy_failure_id,
    build_proxy_state_id,
)
from dllm.core.samplers.proxy_diagnostics import (
    ProxyStateCollectionConfig,
    collect_llada_proxy_states,
    validate_proxy_state_collection_config,
)
from dllm.utils import get_tokenizer


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Parse the explicit single-example P2.3 collection configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument("--last-n-layers", type=int, required=True)
    parser.add_argument("--top-confidence-pool-size", type=int, required=True)
    parser.add_argument("--random-pool-size", type=int, required=True)
    parser.add_argument(
        "--sink-filter",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--sink-quantile", type=float, required=True)
    parser.add_argument(
        "--renormalize-selected-keys",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    return parser.parse_args()


def _run_configuration(
    args: argparse.Namespace,
    *,
    checkpoint: str,
) -> dict[str, object]:
    """Build the semantic configuration protected against mixed resumes."""
    return {
        "checkpoint": checkpoint,
        "dataset": args.dataset_label,
        "split": args.split,
        "response_length": args.response_length,
        "mask_ratios": list(args.mask_ratios),
        "last_n_layers": args.last_n_layers,
        "top_confidence_pool_size": args.top_confidence_pool_size,
        "random_pool_size": args.random_pool_size,
        "sink_filter_enabled": args.sink_filter,
        "sink_quantile": args.sink_quantile,
        "sink_threshold": None,
        "renormalize_selected_keys": args.renormalize_selected_keys,
        "zero_diagonal": args.zero_diagonal,
        "seed": args.seed,
        "dtype": args.dtype,
        "token_value_policy": "base_argmax",
        "trajectory_policy": "stagewise_highest_confidence",
        "evaluation_pool_policy": "top_confidence_plus_seeded_random",
        "chat_template_applied": True,
    }


def _print_summary(**values: object) -> None:
    """Print a compact machine-readable run summary."""
    print(json.dumps(values, indent=2, sort_keys=True))


def main() -> int:
    """Load the pinned model and persist one resumable proxy trajectory."""
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if args.response_length <= 0:
        raise ValueError("response-length must be positive.")
    checkpoint = str(args.checkpoint.resolve())
    config = ProxyStateCollectionConfig(
        mask_ratios=tuple(args.mask_ratios),
        last_n_layers=args.last_n_layers,
        top_confidence_pool_size=args.top_confidence_pool_size,
        random_pool_size=args.random_pool_size,
        seed=args.seed,
        renormalize_selected_keys=args.renormalize_selected_keys,
        zero_diagonal=args.zero_diagonal,
        sink_filter_enabled=args.sink_filter,
        sink_quantile=args.sink_quantile,
    )
    validate_proxy_state_collection_config(
        config,
        response_length=args.response_length,
    )
    configuration = _run_configuration(args, checkpoint=checkpoint)
    requested_device = torch.device(args.device)
    requested_device_index = (
        requested_device.index if requested_device.index is not None else 0
    )
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_device": args.device,
        "gpu_name": (
            torch.cuda.get_device_name(requested_device_index)
            if requested_device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }
    store = ProxyDiagnosticStore(
        args.output_directory,
        configuration=configuration,
        environment=environment,
    )
    expected_state_ids = tuple(
        build_proxy_state_id(
            config_fingerprint=store.configuration_fingerprint,
            example_id=args.example_id,
            prompt=args.prompt,
            state_index=state_index,
            target_mask_ratio=target_ratio,
        )
        for state_index, target_ratio in enumerate(args.mask_ratios)
    )
    completed_before = store.completed_state_ids
    if all(state_id in completed_before for state_id in expected_state_ids):
        _print_summary(
            status="already_completed",
            example_id=args.example_id,
            skipped_state_count=len(expected_state_ids),
            states_path=str(store.states_path.resolve()),
        )
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint collector requires an allocated GPU.")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = get_tokenizer(model_name_or_path=checkpoint)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("The loaded tokenizer does not define mask_token_id.")
    prompt_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": args.prompt}]],
        add_generation_prompt=True,
        tokenize=True,
    )[0]
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
    response_tokens = torch.full(
        (args.response_length,),
        mask_token_id,
        dtype=torch.long,
    )
    input_ids = torch.cat([prompt_tensor, response_tokens]).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, prompt_tensor.numel() :] = True

    source = {
        "dataset": args.dataset_label,
        "split": args.split,
        "example_id": args.example_id,
        "prompt": args.prompt,
        "chat_template_applied": True,
    }
    written_state_count = 0
    skipped_state_count = 0

    def persist_snapshot(snapshot) -> None:
        nonlocal written_state_count, skipped_state_count
        state_id = expected_state_ids[snapshot.state_index]
        state_record = snapshot.to_dict()
        state_record["source"] = source
        if store.append_state(state_id, state_record):
            written_state_count += 1
        else:
            skipped_state_count += 1

    planned_mask_counts = tuple(
        max(
            1,
            int(math.floor(args.response_length * ratio + 0.5)),
        )
        for ratio in args.mask_ratios
    )
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            dtype=DTYPES[args.dtype],
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        model.eval()
        configured_mask_token_id = getattr(model.config, "mask_token_id", None)
        if (
            configured_mask_token_id is not None
            and configured_mask_token_id != mask_token_id
        ):
            raise ValueError(
                "Tokenizer and model mask token IDs differ: "
                f"{mask_token_id} versus {configured_mask_token_id}."
            )
        maximum_length = int(model.config.max_sequence_length)
        if input_ids.shape[1] > maximum_length:
            raise ValueError(
                f"Prompt plus response length {input_ids.shape[1]} exceeds "
                f"model maximum {maximum_length}."
            )
        model_device = next(model.parameters()).device
        collect_llada_proxy_states(
            model,
            input_ids.to(model_device),
            attention_mask=attention_mask.to(model_device),
            response_mask=response_mask.to(model_device),
            mask_token_id=mask_token_id,
            example_id=args.example_id,
            checkpoint=checkpoint,
            config=config,
            snapshot_callback=persist_snapshot,
        )
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        failed_state_index = min(
            written_state_count + skipped_state_count,
            len(args.mask_ratios) - 1,
        )
        failed_pool_size = min(
            planned_mask_counts[failed_state_index],
            args.top_confidence_pool_size + args.random_pool_size,
        )
        failure_id = build_proxy_failure_id(
            config_fingerprint=store.configuration_fingerprint,
            example_id=args.example_id,
            prompt=args.prompt,
            failure_kind="cuda_out_of_memory",
        )
        store.append_failure(
            failure_id,
            {
                "failure_kind": "cuda_out_of_memory",
                "example_id": args.example_id,
                "source": source,
                "error": str(error),
                "sequence_length": int(input_ids.shape[1]),
                "prompt_length": int(prompt_tensor.numel()),
                "response_length": args.response_length,
                "state_index": failed_state_index,
                "target_mask_ratio": args.mask_ratios[failed_state_index],
                "planned_masked_count": planned_mask_counts[failed_state_index],
                "evaluation_pool_size": failed_pool_size,
                "top_confidence_pool_size": args.top_confidence_pool_size,
                "random_pool_size": args.random_pool_size,
                "completed_state_ids": sorted(
                    set(expected_state_ids) & store.completed_state_ids
                ),
            },
        )
        _print_summary(
            status="failed",
            failure_kind="cuda_out_of_memory",
            failure_id=failure_id,
            failures_path=str(store.failures_path.resolve()),
        )
        return 2

    missing_state_ids = set(expected_state_ids) - store.completed_state_ids
    if missing_state_ids:
        raise RuntimeError(
            "Collection returned without persisting expected states: "
            f"{sorted(missing_state_ids)}"
        )
    _print_summary(
        status="completed",
        example_id=args.example_id,
        written_state_count=written_state_count,
        skipped_state_count=skipped_state_count,
        configuration_path=str(store.metadata_path.resolve()),
        states_path=str(store.states_path.resolve()),
        failures_path=str(store.failures_path.resolve()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
