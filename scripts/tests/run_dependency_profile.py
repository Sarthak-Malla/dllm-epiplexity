"""
Profile real-checkpoint LLaDA dependency capture and reconstruction overhead.

On an allocated GPU node:

    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    DEPENDENCY_REPO=/home/sarthak.malla/dllm-selection-ensemble
    DEPENDENCY_CACHE=/home/sarthak.malla/.cache/huggingface/hub
    DEPENDENCY_MODEL=models--GSAI-ML--LLaDA-8B-Instruct
    DEPENDENCY_REVISION=08b83a6feb34df1a6011b80c3c00c7563e963b07
    python "${DEPENDENCY_REPO}/scripts/tests/run_dependency_profile.py" \
        --checkpoint \
        "${DEPENDENCY_CACHE}/${DEPENDENCY_MODEL}/snapshots/${DEPENDENCY_REVISION}" \
        --output-path \
        "${DEPENDENCY_REPO}/eval_results/path_selection/p1_8_capture_profile/result.json"
"""

import argparse
import json
from pathlib import Path
import random
from statistics import mean, median, pstdev
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

import dllm.pipelines.llada.models  # noqa: F401
from dllm.core.samplers.dependency import (
    LLaDAQKCapture,
    build_active_dependency_matrix,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
GIB = 1024**3
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    """Parse the reproducible profiling configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument(
        "--response-lengths",
        type=int,
        nargs="+",
        default=(64, 128, 256, 512),
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        nargs="+",
        default=(1, 2, 4),
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measurement-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize_milliseconds(samples: list[float]) -> dict[str, float]:
    """Summarize synchronized CUDA-event timings."""
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "std_ms": pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure_cuda_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup_runs: int,
    measurement_runs: int,
) -> dict[str, object]:
    """Measure one warmed CUDA operation and its peak allocator state."""
    for _ in range(warmup_runs):
        warmup_result = operation()
        torch.cuda.synchronize(device)
        del warmup_result

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    samples = []
    for _ in range(measurement_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result

    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "latency": summarize_milliseconds(samples),
        "samples_ms": samples,
        "allocated_before_gib": allocated_before / GIB,
        "reserved_before_gib": reserved_before / GIB,
        "peak_allocated_gib": peak_allocated / GIB,
        "peak_reserved_gib": peak_reserved / GIB,
        "incremental_peak_allocated_mib": max(
            0,
            peak_allocated - allocated_before,
        )
        / MIB,
    }


def percentage_overhead(measured_ms: float, baseline_ms: float) -> float:
    """Return latency overhead relative to a positive baseline."""
    if baseline_ms <= 0:
        raise ValueError("baseline_ms must be positive.")
    return 100.0 * (measured_ms / baseline_ms - 1.0)


def save_report(path: Path, report: dict[str, object]) -> None:
    """Atomically replace the JSON report after each completed case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def main() -> int:
    """Load one checkpoint and profile the requested length/layer matrix."""
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if args.prompt_length < 0:
        raise ValueError("prompt-length must be nonnegative.")
    if any(length <= 0 for length in args.response_lengths):
        raise ValueError("response-lengths must contain only positive integers.")
    if any(layer_count <= 0 for layer_count in args.last_n_layers):
        raise ValueError("last-n-layers must contain only positive integers.")
    if args.warmup_runs < 0 or args.measurement_runs <= 0:
        raise ValueError("warmup-runs must be nonnegative and measurement-runs positive.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint profiler requires an allocated CUDA GPU.")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = AutoModelForMaskedLM.from_pretrained(
        args.checkpoint,
        dtype=DTYPES[args.dtype],
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    maximum_length = int(model.config.max_sequence_length)
    maximum_layers = int(model.config.n_layers)
    if args.prompt_length + max(args.response_lengths) > maximum_length:
        raise ValueError("Requested prompt plus response exceeds max_sequence_length.")
    if max(args.last_n_layers) > maximum_layers:
        raise ValueError("Requested last-n-layers exceeds the model layer count.")

    report: dict[str, object] = {
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_class": type(model).__name__,
        "dtype": args.dtype,
        "device": str(model_device),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "batch_size": 1,
        "prompt_length": args.prompt_length,
        "response_lengths": list(args.response_lengths),
        "last_n_layers": list(args.last_n_layers),
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "renormalize_selected_keys": True,
        "zero_diagonal": True,
        "results": [],
    }
    save_report(args.output_path, report)

    vocab_size = int(model.config.vocab_size)
    with torch.inference_mode():
        for response_length in args.response_lengths:
            total_length = args.prompt_length + response_length
            input_ids = torch.arange(
                1,
                total_length + 1,
                device=model_device,
                dtype=torch.long,
            ).remainder(vocab_size)
            input_ids = input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            response_mask[:, args.prompt_length :] = True
            active_mask = response_mask.clone()

            def forward_model() -> torch.Tensor:
                return model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            length_result: dict[str, object] = {
                "status": "running",
                "prompt_length": args.prompt_length,
                "response_length": response_length,
                "total_sequence_length": total_length,
                "aggregated_submatrix_shape": [1, response_length, response_length],
                "aggregated_submatrix_elements": response_length**2,
                "aggregated_submatrix_mib_float32": (
                    response_length**2 * torch.float32.itemsize / MIB
                ),
                "cases": [],
            }
            report["results"].append(length_result)
            try:
                base_profile = measure_cuda_operation(
                    forward_model,
                    device=model_device,
                    warmup_runs=args.warmup_runs,
                    measurement_runs=args.measurement_runs,
                )
                length_result["base_forward"] = base_profile
            except torch.cuda.OutOfMemoryError as error:
                length_result["status"] = "oom"
                length_result["error"] = str(error)
                torch.cuda.empty_cache()
                save_report(args.output_path, report)
                continue

            base_median = base_profile["latency"]["median_ms"]
            base_peak = base_profile["peak_allocated_gib"]
            for last_n_layers in args.last_n_layers:
                case: dict[str, object] = {
                    "status": "running",
                    "last_n_layers": last_n_layers,
                    "per_layer_head_scores_shape": [
                        1,
                        int(model.config.n_heads),
                        response_length,
                        response_length,
                    ],
                    "per_layer_head_scores_elements": (
                        int(model.config.n_heads) * response_length**2
                    ),
                    "per_layer_head_scores_mib_float32": (
                        int(model.config.n_heads)
                        * response_length**2
                        * torch.float32.itemsize
                        / MIB
                    ),
                }
                length_result["cases"].append(case)
                try:
                    with LLaDAQKCapture(
                        model,
                        last_n_layers=last_n_layers,
                    ) as capture:
                        capture_forward = measure_cuda_operation(
                            forward_model,
                            device=model_device,
                            warmup_runs=args.warmup_runs,
                            measurement_runs=args.measurement_runs,
                        )

                        def reconstruct_dependency():
                            if capture.structure is None:
                                raise RuntimeError("Capture structure is unavailable.")
                            return build_active_dependency_matrix(
                                capture.structure,
                                capture.captures,
                                active_mask=active_mask,
                                response_mask=response_mask,
                                attention_mask=attention_mask,
                                zero_diagonal=True,
                                renormalize_selected_keys=True,
                            )

                        reconstruction = measure_cuda_operation(
                            reconstruct_dependency,
                            device=model_device,
                            warmup_runs=args.warmup_runs,
                            measurement_runs=args.measurement_runs,
                        )

                        def capture_and_reconstruct():
                            logits = forward_model()
                            dependency = reconstruct_dependency()
                            return logits, dependency

                        combined = measure_cuda_operation(
                            capture_and_reconstruct,
                            device=model_device,
                            warmup_runs=args.warmup_runs,
                            measurement_runs=args.measurement_runs,
                        )

                        capture_median = capture_forward["latency"]["median_ms"]
                        combined_median = combined["latency"]["median_ms"]
                        case.update(
                            {
                                "status": "completed",
                                "layer_ids": list(capture.structure.layer_ids),
                                "capture_forward": capture_forward,
                                "reconstruction": reconstruction,
                                "capture_plus_reconstruction": combined,
                                "capture_forward_overhead_percent": (
                                    percentage_overhead(capture_median, base_median)
                                ),
                                "total_overhead_percent": percentage_overhead(
                                    combined_median,
                                    base_median,
                                ),
                                "capture_peak_delta_vs_base_mib": (
                                    capture_forward["peak_allocated_gib"] - base_peak
                                )
                                * 1024,
                                "combined_peak_delta_vs_base_mib": (
                                    combined["peak_allocated_gib"] - base_peak
                                )
                                * 1024,
                            }
                        )
                except torch.cuda.OutOfMemoryError as error:
                    case["status"] = "oom"
                    case["error"] = str(error)
                    torch.cuda.empty_cache()
                save_report(args.output_path, report)
            length_result["status"] = "completed"
            save_report(args.output_path, report)

    report["status"] = "completed"
    save_report(args.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved capture profile to {args.output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
