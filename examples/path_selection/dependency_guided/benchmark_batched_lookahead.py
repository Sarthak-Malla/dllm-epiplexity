"""Benchmark sequential, batched, and chunked lookahead on LLaDA.

Run on an allocated GPU with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    REPO=/home/sarthak.malla/dllm-selection-ensemble
    CACHE=/home/sarthak.malla/.cache/huggingface/hub
    MODEL=models--GSAI-ML--LLaDA-8B-Instruct
    REVISION=08b83a6feb34df1a6011b80c3c00c7563e963b07
    SCRIPT=${REPO}/examples/path_selection/dependency_guided
    OUTPUT=${REPO}/eval_results/path_selection/dependency_guided
    python "${SCRIPT}/benchmark_batched_lookahead.py" \
        --checkpoint "${CACHE}/${MODEL}/snapshots/${REVISION}" \
        --output-path "${OUTPUT}/p4_6_batched_lookahead/result.json"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, median, pstdev
import time
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

import dllm.pipelines.llada.models  # noqa: F401
from dllm.core.samplers.batched_lookahead import (
    candidate_batch_from_mask_mapping,
    evaluate_batched_lookahead,
)
from dllm.core.samplers.counterfactual import (
    decoding_risk_per_token,
    entropy_per_token,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
GIB = 1024**3
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    """Parse a reproducible Phase 4 benchmark configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("entropy_drop", "risk_reduction"),
        default=("entropy_drop", "risk_reduction"),
    )
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--response-length", type=int, default=128)
    parser.add_argument(
        "--candidate-budgets",
        type=int,
        nargs="+",
        default=(2, 4, 8),
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=(2, 4),
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measurement-runs", type=int, default=5)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _summary(samples: list[float]) -> dict[str, float]:
    """Summarize synchronized CUDA timings in milliseconds."""
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "std_ms": pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_cuda(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup_runs: int,
    measurement_runs: int,
) -> dict[str, object]:
    """Measure one warmed operation and incremental allocated CUDA memory."""
    for _ in range(warmup_runs):
        output = operation()
        torch.cuda.synchronize(device)
        del output

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    device_samples = []
    wall_samples = []
    for _ in range(measurement_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started_at = time.perf_counter()
        start.record()
        output = operation()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_started_at) * 1000.0)
        device_samples.append(float(start.elapsed_time(end)))
        del output
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "device_time": _summary(device_samples),
        "device_samples_ms": device_samples,
        "wall_time": _summary(wall_samples),
        "wall_samples_ms": wall_samples,
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


def _save_report(path: Path, report: dict[str, object]) -> None:
    """Atomically persist partial results so interrupted runs remain useful."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _fixed_benchmark_candidates(
    active_mask: torch.Tensor,
    candidate_budget: int,
):
    """Create policy-neutral singleton actions solely for verifier timing."""
    batch_size, sequence_length = active_mask.shape
    masks: dict[str, torch.Tensor] = {}
    positions_by_batch = [
        torch.where(active_mask[batch_index])[0]
        for batch_index in range(batch_size)
    ]
    for candidate_index in range(candidate_budget):
        mask = torch.zeros_like(active_mask)
        for batch_index, positions in enumerate(positions_by_batch):
            if positions.numel() < candidate_budget:
                raise ValueError(
                    "Every batch row needs at least candidate_budget active positions."
                )
            if candidate_budget == 1:
                position_rank = 0
            else:
                position_rank = (
                    candidate_index * (positions.numel() - 1)
                ) // (candidate_budget - 1)
            mask[batch_index, positions[position_rank]] = True
        masks[f"benchmark_candidate_{candidate_index}"] = mask
    return candidate_batch_from_mask_mapping(
        masks,
        eligible_mask=active_mask,
    )


def _maximum_score_difference(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    """Return max finite absolute score difference, or zero for empty scores."""
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return 0.0
    return float((actual[finite] - expected[finite]).abs().max().item())


def _maximum_mean_score_difference(
    actual: torch.Tensor,
    expected: torch.Tensor,
    heldout_counts: torch.Tensor,
) -> float:
    """Return the largest score difference per held-out position."""
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return 0.0
    denominators = heldout_counts.clamp_min(1).to(dtype=torch.float32)
    differences = (actual - expected).abs() / denominators
    return float(differences[finite].max().item())


def _json_scores(scores: torch.Tensor) -> list[list[float | None]]:
    """Serialize finite [N,B] scores without non-standard JSON infinities."""
    return [
        [
            float(value) if math.isfinite(float(value)) else None
            for value in candidate_row
        ]
        for candidate_row in scores.detach().float().cpu().tolist()
    ]


def _selection_margins(scores: torch.Tensor) -> list[float | None]:
    """Return top-one minus top-two score margins for each batch row."""
    margins = []
    for batch_index in range(scores.shape[1]):
        finite_scores = scores[:, batch_index][
            torch.isfinite(scores[:, batch_index])
        ]
        if finite_scores.numel() < 2:
            margins.append(None)
            continue
        top_two = torch.topk(finite_scores.float(), k=2).values
        margins.append(float((top_two[0] - top_two[1]).item()))
    return margins


def main() -> int:
    """Run the required N=2/4/8 Phase 4 real-checkpoint comparisons."""
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if args.prompt_length < 0 or args.response_length <= 0:
        raise ValueError(
            "prompt-length must be nonnegative and response-length positive."
        )
    if any(budget <= 0 for budget in args.candidate_budgets):
        raise ValueError("candidate-budgets must contain positive integers.")
    if any(size <= 0 for size in args.chunk_sizes):
        raise ValueError("chunk-sizes must contain positive integers.")
    if max(args.candidate_budgets) > args.response_length:
        raise ValueError("response-length must cover the largest candidate budget.")
    if args.warmup_runs < 0 or args.measurement_runs <= 0:
        raise ValueError(
            "warmup-runs must be nonnegative and measurement-runs positive."
        )
    if not np.isfinite(args.score_atol) or args.score_atol < 0:
        raise ValueError("score-atol must be finite and nonnegative.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint benchmark requires an allocated GPU.")
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
    sequence_length = args.prompt_length + args.response_length
    maximum_length = int(model.config.max_sequence_length)
    if sequence_length > maximum_length:
        raise ValueError("prompt plus response exceeds model max_sequence_length.")
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("The model configuration does not define mask_token_id.")

    vocabulary_size = int(model.config.vocab_size)
    input_ids = torch.arange(
        1,
        sequence_length + 1,
        device=model_device,
        dtype=torch.long,
    ).remainder(vocabulary_size)
    input_ids = input_ids.unsqueeze(0)
    input_ids[:, args.prompt_length :] = int(mask_token_id)
    attention_mask = torch.ones_like(input_ids)
    active_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    active_mask[:, args.prompt_length :] = True

    with torch.inference_mode():
        base_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        predicted_token_ids = base_logits.argmax(dim=-1)
        base_maps = {
            "entropy_drop": entropy_per_token(base_logits),
            "risk_reduction": decoding_risk_per_token(base_logits),
        }

    report: dict[str, object] = {
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_class": type(model).__name__,
        "dtype": args.dtype,
        "device": str(model_device),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "batch_size": input_ids.shape[0],
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "sequence_length": sequence_length,
        "candidate_budgets": list(args.candidate_budgets),
        "chunk_sizes": list(args.chunk_sizes),
        "metrics": list(args.metrics),
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "score_atol": args.score_atol,
        "candidate_policy": (
            "fixed evenly distributed singleton actions for verifier-only timing; "
            "not a decoding proposal method"
        ),
        "compute_accounting": (
            "model_calls counts invocations; expanded_batch_rows reports the "
            "N-fold candidate batch compute"
        ),
        "results": [],
    }
    _save_report(args.output_path, report)

    equivalence_passed = True
    oom_case_count = 0
    for metric in args.metrics:
        for candidate_budget in args.candidate_budgets:
            candidates = _fixed_benchmark_candidates(
                active_mask,
                candidate_budget,
            )
            configurations = [
                ("sequential", 1),
                ("batched", None),
            ]
            if candidate_budget == max(args.candidate_budgets):
                configurations.extend(
                    ("chunked", chunk_size)
                    for chunk_size in args.chunk_sizes
                    if chunk_size < candidate_budget
                )

            reference = evaluate_batched_lookahead(
                model,
                input_ids,
                predicted_token_ids,
                candidates,
                base_metric_map=base_maps[metric],
                metric=metric,
                attention_mask=attention_mask,
                masked_active_mask=active_mask,
                candidate_chunk_size=1,
            )
            for mode, chunk_size in configurations:
                def operation():
                    return evaluate_batched_lookahead(
                        model,
                        input_ids,
                        predicted_token_ids,
                        candidates,
                        base_metric_map=base_maps[metric],
                        metric=metric,
                        attention_mask=attention_mask,
                        masked_active_mask=active_mask,
                        candidate_chunk_size=chunk_size,
                    )

                try:
                    observed = operation()
                    profile = _measure_cuda(
                        operation,
                        device=model_device,
                        warmup_runs=args.warmup_runs,
                        measurement_runs=args.measurement_runs,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    oom_case_count += 1
                    report["results"].append(
                        {
                            "status": "oom",
                            "metric": metric,
                            "mode": mode,
                            "candidate_budget": candidate_budget,
                            "candidate_chunk_size": chunk_size,
                            "expanded_batch_rows": (
                                candidate_budget * input_ids.shape[0]
                            ),
                            "error": str(error),
                        }
                    )
                    torch.cuda.empty_cache()
                    _save_report(args.output_path, report)
                    continue
                score_difference = _maximum_score_difference(
                    observed.scores,
                    reference.scores,
                )
                mean_score_difference = _maximum_mean_score_difference(
                    observed.scores,
                    reference.scores,
                    observed.heldout_counts,
                )
                metric_full_scale = (
                    math.log(vocabulary_size)
                    if metric == "entropy_drop"
                    else 1.0
                )
                indexes_equal = torch.equal(
                    observed.best_index,
                    reference.best_index,
                )
                masks_equal = torch.equal(
                    observed.best_mask,
                    reference.best_mask,
                )
                case_passed = (
                    score_difference <= args.score_atol
                    and indexes_equal
                    and masks_equal
                )
                equivalence_passed = equivalence_passed and case_passed
                median_ms = profile["wall_time"]["median_ms"]
                case = {
                    "status": "completed",
                    "metric": metric,
                    "mode": mode,
                    "candidate_budget": candidate_budget,
                    "candidate_chunk_size": chunk_size,
                    "model_calls": observed.model_calls,
                    "expanded_batch_rows": (
                        candidate_budget * input_ids.shape[0]
                    ),
                    "maximum_rows_per_model_call": (
                        min(chunk_size or candidate_budget, candidate_budget)
                        * input_ids.shape[0]
                    ),
                    "candidate_throughput_per_second": (
                        candidate_budget * input_ids.shape[0] * 1000.0 / median_ms
                    ),
                    "latency_and_memory": profile,
                    "max_abs_score_difference_vs_sequential": score_difference,
                    "max_abs_mean_score_difference_vs_sequential": (
                        mean_score_difference
                    ),
                    "max_mean_difference_fraction_of_metric_range": (
                        mean_score_difference / metric_full_scale
                    ),
                    "sequential_scores_by_candidate_and_batch": _json_scores(
                        reference.scores
                    ),
                    "observed_scores_by_candidate_and_batch": _json_scores(
                        observed.scores
                    ),
                    "heldout_counts_by_candidate_and_batch": (
                        observed.heldout_counts.detach().cpu().tolist()
                    ),
                    "sequential_selection_margins": _selection_margins(
                        reference.scores
                    ),
                    "observed_selection_margins": _selection_margins(
                        observed.scores
                    ),
                    "selected_indexes_equal": indexes_equal,
                    "selected_masks_equal": masks_equal,
                    "passed": case_passed,
                }
                report["results"].append(case)
                _save_report(args.output_path, report)

    if not equivalence_passed:
        report["status"] = "failed_equivalence"
    elif oom_case_count:
        report["status"] = "completed_with_oom"
    else:
        report["status"] = "completed"
    report["equivalence_passed"] = equivalence_passed
    report["oom_case_count"] = oom_case_count
    _save_report(args.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved benchmark report to {args.output_path.resolve()}")
    if not equivalence_passed:
        print(
            "Benchmark completed, but the declared scientific equivalence "
            "criterion did not pass; inspect the saved report."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
