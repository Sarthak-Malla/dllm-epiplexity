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
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness, MDLMEvalSamplerConfig
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.base import BaseSamplerOutput
import dllm.utils


# Global harness instance registry for candidate saving
_last_harness_instance = None


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
        eval_config = PathSelectionEvalConfig()
        
        # Decide which sampler to use based on sampler_type string
        if sampler_type == "greedy":
            sampler_cls = MDLMSampler
            sampler_config = PathSelectionSamplerConfig()
        elif sampler_type == "entropy_drop":
            from dllm.core.samplers.entropy_drop import EntropyDropSampler, EntropyDropSamplerConfig

            sampler_cls = EntropyDropSampler
            # Extract oracle_candidate_strategy from kwargs if provided
            oracle_candidate_strategy = kwargs.pop("oracle_candidate_strategy", "mixed")
            sampler_config = EntropyDropSamplerConfig(
                oracle_candidate_strategy=oracle_candidate_strategy,
                return_dict=True,  # Enable return_dict for candidate tracking
            )
        elif sampler_type == "risk_reduction":
            from dllm.core.samplers.risk_reduction import RiskReductionSampler, RiskReductionSamplerConfig

            sampler_cls = RiskReductionSampler
            sampler_config = RiskReductionSamplerConfig(
                return_dict=True,  # Enable return_dict for candidate tracking
            )
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )
        self.sampler_type = sampler_type
        self.selected_candidates_per_example = []
        self.example_index = 0
        
        # Register this harness instance globally for later access
        global _last_harness_instance
        _last_harness_instance = self

    @torch.no_grad()
    def generate_until(self, requests: List) -> List[str]:
        """Generate until with candidate tracking for entropy_drop sampler."""
        out: List[str] = []

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
            sampler_output = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=True,
            )

            # Handle both BaseSamplerOutput and raw tensor (for greedy fallback)
            if isinstance(sampler_output, BaseSamplerOutput):
                generated_ids = sampler_output.sequences
                selected_candidates = sampler_output.selected_candidates or [[] for _ in range(len(prompts))]
            else:
                generated_ids = sampler_output
                selected_candidates = [[] for _ in range(len(prompts))]

            # Track candidates per example
            for candidates in selected_candidates:
                self.selected_candidates_per_example.append({
                    "example_index": self.example_index,
                    "selected_candidates": candidates,
                })
                self.example_index += 1

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

        return out

    def save_selected_candidates(self, output_path: str):
        """Save the selected candidates per example to a JSON file alongside results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as {output_path}_{sampler_type}_candidates.json
        candidates_path = output_path.parent / f"{output_path.name}_{self.sampler_type}_candidates.json"
        
        with open(candidates_path, "w") as f:
            json.dump(self.selected_candidates_per_example, f, indent=2)
        
        print(f"Selected candidates saved to {candidates_path}")


if __name__ == "__main__":
    import sys
    
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