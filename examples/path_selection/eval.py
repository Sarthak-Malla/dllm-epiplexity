"""
Run LLaDA evaluation with a selectable path-selection sampler.

From the repository root:

    source ~/.bashrc
    conda activate dllm
    accelerate launch --num_processes 1 \
        examples/path_selection/eval.py \
        --tasks gsm8k_cot \
        --model llada_path_selection \
        --apply_chat_template \
        --num_fewshot 5 \
        --limit 2 \
        --model_args "sampler_type=greedy,pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=32,steps=8,block_size=8,cfg_scale=0.0"

Use ``--model_args sampler_type=greedy`` for the native baseline.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import List

import torch
from tqdm import tqdm

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness, MDLMEvalSamplerConfig
from dllm.core.eval.diagnostic_retention import (
    DIAGNOSTIC_RETENTION_MODES,
    retain_diagnostics,
)
from dllm.core.eval.wandb_monitor import log_generation_progress
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.base import BaseSamplerOutput
import dllm.utils


# Global harness instance registry for candidate saving
_last_harness_instance = None


def _strip_wandb_arguments_for_non_main_rank(
    arguments: list[str],
    *,
    local_rank: int,
) -> list[str]:
    """Keep a single W&B run when Accelerate starts multiple evaluator ranks."""
    if local_rank == 0:
        return list(arguments)
    filtered = list(arguments)
    for option in ("--wandb_args", "--wandb_config_args"):
        while option in filtered:
            option_index = filtered.index(option)
            delete_stop = min(option_index + 2, len(filtered))
            del filtered[option_index:delete_stop]
    return filtered


@dataclass
class PathSelectionEvalConfig(MDLMEvalConfig):
    """Evaluation defaults shared by all path-selection samplers."""

    max_length: int = 4096


@dataclass
class PathSelectionSamplerConfig(MDLMEvalSamplerConfig):
    """Native MDLM settings shared by all path-selection samplers."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@register_model("llada_path_selection")
class LLaDAPathSelectionEvalHarness(MDLMEvalHarness):
    """Standard lm-eval harness with selectable path-selection samplers."""

    def __init__(self, sampler_type: str = "greedy", **kwargs):
        diagnostic_retention = kwargs.pop("diagnostic_retention", "full")
        if diagnostic_retention not in DIAGNOSTIC_RETENTION_MODES:
            raise ValueError(
                "diagnostic_retention must be one of "
                f"{DIAGNOSTIC_RETENTION_MODES}, got {diagnostic_retention!r}."
            )
        eval_config = PathSelectionEvalConfig()
        
        # Decide which sampler to use based on sampler_type string
        if sampler_type == "greedy":
            sampler_cls = MDLMSampler
            sampler_config = PathSelectionSamplerConfig()
        elif sampler_type in {"entropy_drop", "dependency_entropy"}:
            from dllm.core.samplers.entropy_drop import EntropyDropSampler, EntropyDropSamplerConfig

            sampler_cls = EntropyDropSampler
            # Extract oracle_candidate_strategy from kwargs if provided
            oracle_candidate_strategy = kwargs.pop("oracle_candidate_strategy", "mixed")
            if sampler_type == "dependency_entropy":
                kwargs.setdefault("proposal_strategy", "dependency")
                kwargs.setdefault("diagnostic_metadata", True)
            sampler_config = EntropyDropSamplerConfig(
                oracle_candidate_strategy=oracle_candidate_strategy,
                return_dict=True,  # Enable return_dict for candidate tracking
            )
        elif sampler_type == "dependency_non_lookahead":
            from dllm.core.samplers.dependency_non_lookahead import (
                DependencyNonLookaheadSampler,
                DependencyNonLookaheadSamplerConfig,
            )

            sampler_cls = DependencyNonLookaheadSampler
            sampler_config = DependencyNonLookaheadSamplerConfig(return_dict=True)
        elif sampler_type in {"risk_reduction", "dependency_risk"}:
            from dllm.core.samplers.risk_reduction import RiskReductionSampler, RiskReductionSamplerConfig

            sampler_cls = RiskReductionSampler
            if sampler_type == "dependency_risk":
                kwargs.setdefault("proposal_strategy", "dependency")
                kwargs.setdefault("diagnostic_metadata", True)
            sampler_config = RiskReductionSamplerConfig(
                return_dict=True,  # Enable return_dict for candidate tracking
            )
        else:
            available = (
                "greedy, entropy_drop, risk_reduction, dependency_entropy, "
                "dependency_risk, dependency_non_lookahead"
            )
            raise ValueError(
                f"Unknown sampler_type: {sampler_type}. Available: {available}."
            )

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )
        self.sampler_type = sampler_type
        self.selected_candidates_per_example = []
        self.diagnostics_per_example = []
        self.generation_batch_seconds = []
        self.example_index = 0
        self.diagnostic_retention = diagnostic_retention
        
        # Register this harness instance globally for later access
        global _last_harness_instance
        _last_harness_instance = self

    @torch.no_grad()
    def generate_until(self, requests: List) -> List[str]:
        """Generate until with candidate tracking for entropy_drop sampler."""
        out: List[str] = []
        generation_batch_seconds = []

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size), desc="Generating..."
        ):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                for ctx in contexts
            ]

            # Call sampler with return_dict=True to capture metadata
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            generation_started_at = time.perf_counter()
            sampler_output = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            generation_batch_seconds.append(
                time.perf_counter() - generation_started_at
            )
            batch_seconds = generation_batch_seconds[-1]

            # Handle both BaseSamplerOutput and raw tensor (for greedy fallback)
            if isinstance(sampler_output, BaseSamplerOutput):
                generated_ids = sampler_output.sequences
                selected_candidates = sampler_output.selected_candidates or [[] for _ in range(len(prompts))]
                diagnostics = sampler_output.diagnostics or [
                    [] for _ in range(len(prompts))
                ]
            else:
                generated_ids = sampler_output
                selected_candidates = [[] for _ in range(len(prompts))]
                diagnostics = [[] for _ in range(len(prompts))]

            # Track candidates per example
            retained_diagnostics = retain_diagnostics(
                diagnostics,
                self.diagnostic_retention,
            )
            for batch_index, (candidates, _example_diagnostics) in enumerate(
                zip(selected_candidates, diagnostics)
            ):
                self.selected_candidates_per_example.append({
                    "example_index": self.example_index,
                    "selected_candidates": candidates,
                })
                retained_example_diagnostics = retained_diagnostics[batch_index]
                self.diagnostics_per_example.append(
                    {
                        "example_index": self.example_index,
                        "steps": retained_example_diagnostics,
                    }
                )
                self.example_index += 1

            if self.rank == 0:
                cuda_memory = None
                if self.device.type == "cuda":
                    cuda_memory = {
                        "allocated_bytes": torch.cuda.memory_allocated(self.device),
                        "reserved_bytes": torch.cuda.memory_reserved(self.device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            self.device
                        ),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                            self.device
                        ),
                    }
                log_generation_progress(
                    examples_completed=min(batch_start + len(batch), len(requests)),
                    examples_total=len(requests),
                    batch_seconds=batch_seconds,
                    cumulative_seconds=(
                        sum(self.generation_batch_seconds)
                        + sum(generation_batch_seconds)
                    ),
                    diagnostics_by_example=diagnostics,
                    cuda_memory=cuda_memory,
                )

            generated_answers = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                [p.tolist() for p in prompts],
            )

            for answer, gen_kwargs in zip(generated_answers, gen_kwargs_list):
                for stop_seq in gen_kwargs["until"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0]
                out.append(answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        if generation_batch_seconds:
            self.generation_batch_seconds.extend(generation_batch_seconds)
            print(
                "Generation timing: "
                f"total_seconds={sum(generation_batch_seconds):.6f}, "
                f"first_batch_seconds={generation_batch_seconds[0]:.6f}, "
                "post_warmup_seconds="
                f"{sum(generation_batch_seconds[1:]):.6f}, "
                f"batch_count={len(generation_batch_seconds)}"
            )

        return out

    def save_selected_candidates(self, output_path: str):
        """Save the selected candidates per example to a JSON file alongside results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rank = self.rank
        world_size = self.world_size
        rank_suffix = (
            "" if world_size == 1 else f"_rank{rank:05d}-of-{world_size:05d}"
        )
        artifact_prefix = f"{output_path.name}_{self.sampler_type}"
        candidates_path = (
            output_path.parent / f"{artifact_prefix}_candidates{rank_suffix}.json"
        )
        with open(candidates_path, "w") as f:
            json.dump(self.selected_candidates_per_example, f, indent=2)
        diagnostics_path = (
            output_path.parent
            / f"{artifact_prefix}_diagnostics{rank_suffix}.json"
        )
        with open(diagnostics_path, "w") as f:
            json.dump(self.diagnostics_per_example, f, indent=2)
        runtime_path = (
            output_path.parent
            / f"{artifact_prefix}_runtime{rank_suffix}.json"
        )
        runtime = {
            "sampler_type": self.sampler_type,
            "rank": rank,
            "world_size": world_size,
            "generation_total_seconds": sum(self.generation_batch_seconds),
            "generation_batch_seconds": self.generation_batch_seconds,
            "generation_batch_count": len(self.generation_batch_seconds),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
            ),
        }
        with open(runtime_path, "w") as f:
            json.dump(runtime, f, indent=2)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if world_size > 1 and rank == 0:
            rank_runtime_paths = [
                output_path.parent
                / (
                    f"{artifact_prefix}_runtime_rank{rank_index:05d}"
                    f"-of-{world_size:05d}.json"
                )
                for rank_index in range(world_size)
            ]
            rank_runtimes = [
                json.loads(path.read_text()) for path in rank_runtime_paths
            ]
            generation_rank_seconds = [
                float(item["generation_total_seconds"])
                for item in rank_runtimes
            ]
            completion_path = (
                output_path.parent / f"{artifact_prefix}_runtime.json"
            )
            completion_runtime = {
                "sampler_type": self.sampler_type,
                "distributed": True,
                "world_size": world_size,
                "generation_total_seconds": max(generation_rank_seconds),
                "generation_work_seconds": sum(generation_rank_seconds),
                "generation_rank_seconds": generation_rank_seconds,
                "generation_batch_count": max(
                    int(item["generation_batch_count"])
                    for item in rank_runtimes
                ),
                "cuda_peak_allocated_bytes": max(
                    int(item["cuda_peak_allocated_bytes"] or 0)
                    for item in rank_runtimes
                ),
                "cuda_peak_reserved_bytes": max(
                    int(item["cuda_peak_reserved_bytes"] or 0)
                    for item in rank_runtimes
                ),
                "rank_runtime_paths": [str(path) for path in rank_runtime_paths],
            }
            completion_path.write_text(
                json.dumps(completion_runtime, indent=2) + "\n"
            )
            for artifact_kind in ("candidates", "diagnostics"):
                manifest_path = (
                    output_path.parent
                    / f"{artifact_prefix}_{artifact_kind}_manifest.json"
                )
                shard_paths = [
                    output_path.parent
                    / (
                        f"{artifact_prefix}_{artifact_kind}_rank{rank_index:05d}"
                        f"-of-{world_size:05d}.json"
                    )
                    for rank_index in range(world_size)
                ]
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "distributed": True,
                            "world_size": world_size,
                            "shards": [str(path) for path in shard_paths],
                        },
                        indent=2,
                    )
                    + "\n"
                )

        print(f"Selected candidates saved to {candidates_path}")
        print(f"Step diagnostics saved to {diagnostics_path}")
        print(f"Runtime metrics saved to {runtime_path}")


if __name__ == "__main__":
    import sys

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    sys.argv = _strip_wandb_arguments_for_non_main_rank(
        sys.argv,
        local_rank=local_rank,
    )

    process_started_at = time.perf_counter()
    cuda_metrics_enabled = torch.cuda.is_available()
    if cuda_metrics_enabled:
        torch.cuda.reset_peak_memory_stats()
    
    # Extract output_path from arguments
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            break
    
    cli_evaluate()
    
    # Save candidates if harness was used and output_path provided
    if output_path and _last_harness_instance is not None:
        try:
            _last_harness_instance.save_selected_candidates(output_path)
        except Exception as e:
            print(f"Error saving selected candidates: {e}")

    if cuda_metrics_enabled:
        torch.cuda.synchronize()
        gibibyte = 1024**3
        print(
            "CUDA memory: "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated() / gibibyte:.6f}, "
            f"peak_reserved_gib={torch.cuda.max_memory_reserved() / gibibyte:.6f}"
        )
    print(f"Process wall time: {time.perf_counter() - process_started_at:.6f} seconds")
