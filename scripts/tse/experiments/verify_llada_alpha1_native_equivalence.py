"""Verify that LLaDA-master CTCA at alpha=1 matches native LLaDA.

Run inside the two-GPU Slurm launcher:

    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments/verify_llada_alpha1_native_equivalence.py \
        --output-json /home/sarthak.malla/dllm-epiplexity/.logs/llada-alpha1-equivalence.json

The check loads LLaDA once, shares that exact model instance between the native
and CTCA samplers, and fails unless every committed-token canvas is identical.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.tse.ctca_sampler import CTCATSESampler
from dllm.pipelines.tse.loader import load_tse_models
from dllm.pipelines.tse.models import TSEConfig
from dllm.utils.sampling import sample_trim


DEFAULT_QUESTION = (
    "Q: A store has 24 apples. It sells 7 in the morning and 14 in the "
    "afternoon. How many apples remain?\n\nA:"
)


def parse_num_anchors(value: str) -> int | str:
    """Accept either a positive anchor count or the CTCA ``auto`` policy."""
    if value.casefold() == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("num anchors must be positive or 'auto'")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the fixed-endpoint equivalence configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--model-b", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--model-a-device", default="cuda:0")
    parser.add_argument("--model-b-device", default="cuda:1")
    parser.add_argument("--fusion-device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=".cache/ctca")
    parser.add_argument("--anchor-temperature", type=float, default=0.01)
    parser.add_argument("--projection-temperature", type=float, default=0.05)
    parser.add_argument("--projection-chunk-size", type=int, default=512)
    parser.add_argument(
        "--projection-mode", choices=("exact", "sparse_topk"), default="sparse_topk"
    )
    parser.add_argument("--projection-top-k", type=int, default=64)
    parser.add_argument("--num-anchors", type=parse_num_anchors, default="auto")
    parser.add_argument("--min-anchors", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def synchronize(device: str) -> None:
    """Synchronize a CUDA device before reading a wall-clock measurement."""
    parsed = torch.device(device)
    if parsed.type == "cuda":
        torch.cuda.synchronize(parsed)


def timed_sample(device: str, operation):
    """Run a sampler operation and return its output plus elapsed seconds."""
    synchronize(device)
    started = time.perf_counter()
    output = operation()
    synchronize(device)
    return output, time.perf_counter() - started


def mismatch_record(
    native_history: list[torch.Tensor], ctca_history: list[torch.Tensor]
) -> dict | None:
    """Describe the first trajectory mismatch, if one exists."""
    if len(native_history) != len(ctca_history):
        return {
            "kind": "history_length",
            "native_steps": len(native_history) - 1,
            "ctca_steps": len(ctca_history) - 1,
        }
    for step, (native_canvas, ctca_canvas) in enumerate(
        zip(native_history, ctca_history)
    ):
        native_cpu = native_canvas.detach().cpu()
        ctca_cpu = ctca_canvas.detach().cpu()
        if torch.equal(native_cpu, ctca_cpu):
            continue
        differing = torch.nonzero(native_cpu != ctca_cpu, as_tuple=False)
        examples = []
        for batch_index, position in differing[:20].tolist():
            examples.append(
                {
                    "batch_index": batch_index,
                    "position": position,
                    "native_token_id": int(native_cpu[batch_index, position]),
                    "ctca_token_id": int(ctca_cpu[batch_index, position]),
                }
            )
        return {
            "kind": "canvas",
            "step": step,
            "difference_count": int(differing.shape[0]),
            "examples": examples,
        }
    return None


def main() -> None:
    """Run native and alpha-one CTCA decoding and write an auditable result."""
    args = parse_args()
    if min(args.max_new_tokens, args.steps, args.block_size) < 1:
        raise ValueError("generation lengths and step counts must be positive")

    config = TSEConfig(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        model_a_device=args.model_a_device,
        model_b_device=args.model_b_device,
        dtype=args.dtype,
        ctca_enabled=True,
        master_model="a",
        fusion_device=args.fusion_device,
        ctca_cache_dir=args.cache_dir,
        ctca_anchor_temperature=args.anchor_temperature,
        ctca_projection_temperature=args.projection_temperature,
        ctca_chunk_size=args.projection_chunk_size,
        ctca_projection_mode=args.projection_mode,
        ctca_projection_top_k=args.projection_top_k,
        ctca_num_anchors=args.num_anchors,
        ctca_min_anchors=args.min_anchors,
    )
    models = load_tse_models(config)
    messages = [{"role": "user", "content": args.question}]
    master_prompt = torch.as_tensor(
        models.tokenizer_a.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0],
        dtype=torch.long,
    )
    auxiliary_prompt = torch.as_tensor(
        models.tokenizer_b.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0],
        dtype=torch.long,
    )

    native_sampler = MDLMSampler(models.model_a, models.tokenizer_a)
    ctca_sampler = CTCATSESampler(
        models.model_a,
        models.model_b,
        models.tokenizer_a,
        models.tokenizer_b,
        args.model_a_device,
        args.model_b_device,
        master_id="a",
        auxiliary_id="b",
        cache_dir=args.cache_dir,
        anchor_temperature=args.anchor_temperature,
        projection_temperature=args.projection_temperature,
        projection_chunk_size=args.projection_chunk_size,
        projection_mode=args.projection_mode,
        projection_top_k=args.projection_top_k,
        num_anchors=args.num_anchors,
        min_anchors=args.min_anchors,
        master_cache_id=args.model_a,
        auxiliary_cache_id=args.model_b,
    )
    native_config = MDLMSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        temperature=0.0,
        probability_temperature=1.0,
        remasking="low_confidence",
        stochastic_transfer=False,
        cfg_scale=0.0,
        suppress_tokens=[],
        begin_suppress_tokens=[126081, 126348],
        return_dict=True,
    )

    transformers.set_seed(args.seed)
    native, native_seconds = timed_sample(
        args.model_a_device,
        lambda: native_sampler.sample([master_prompt], native_config),
    )
    transformers.set_seed(args.seed)
    ctca, ctca_seconds = timed_sample(
        args.model_a_device,
        lambda: ctca_sampler.sample(
            [master_prompt],
            auxiliary_inputs=[auxiliary_prompt],
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
            temperature=0.0,
            remasking="low_confidence",
            stochastic_transfer=False,
            capture_history=True,
            selection_mode="tse",
            weighting_mode="static",
            alpha=1.0,
            temperature_a=1.0,
            temperature_b=1.0,
            fusion_device=args.fusion_device,
            master_suppress_tokens=[],
            master_begin_suppress_tokens=[126081, 126348],
        ),
    )

    native_history = native.histories or []
    trajectory_mismatch = mismatch_record(
        native_history, ctca_sampler.last_canvas_history
    )
    final_tokens_equal = torch.equal(native.sequences.detach().cpu(), ctca.cpu())
    native_text = sample_trim(
        models.tokenizer_a, native.sequences.detach().cpu(), [master_prompt]
    )[0]
    ctca_text = sample_trim(models.tokenizer_a, ctca.cpu(), [master_prompt])[0]
    passed = trajectory_mismatch is None and final_tokens_equal and native_text == ctca_text
    result = {
        "schema_version": 1,
        "protocol_id": "ctca-llada-master-equivalence-v1",
        "experiment_name": "ctca-llada8bi-alpha1-nativeequiv-20261018",
        "passed": passed,
        "endpoint": {
            "master_model": "a",
            "selection_mode": "tse",
            "weighting_mode": "static",
            "alpha": 1.0,
            "temperature_a": 1.0,
            "temperature_b": 1.0,
        },
        "models": {"a": args.model_a, "b": args.model_b},
        "schedule": {
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "block_size": args.block_size,
            "temperature": 0.0,
            "probability_temperature": 1.0,
        },
        "prompt_lengths": {
            "master": int(master_prompt.numel()),
            "auxiliary": int(auxiliary_prompt.numel()),
        },
        "trajectory_steps": {
            "native": max(len(native_history) - 1, 0),
            "ctca": max(len(ctca_sampler.last_canvas_history) - 1, 0),
        },
        "trajectory_mismatch": trajectory_mismatch,
        "final_tokens_equal": final_tokens_equal,
        "trimmed_text_equal": native_text == ctca_text,
        "native_text": native_text,
        "ctca_text": ctca_text,
        "timing_seconds": {"native": native_seconds, "ctca_alpha1": ctca_seconds},
        "ctca_to_native_time_ratio": ctca_seconds / native_seconds,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
