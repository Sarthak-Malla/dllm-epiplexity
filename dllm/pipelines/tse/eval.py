"""lm-eval entrypoint for synchronized LLaDA Base/Instruct TSE."""

from dataclasses import dataclass

import accelerate
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from dllm.pipelines.tse.loader import load_tse_models
from dllm.pipelines.tse.models import TSEConfig
from dllm.pipelines.tse.sampler import TSESampler


@dataclass
class TSEEvalHarness(LM):
    """Run paired forwards with baseline or TSE selection."""

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get("model_a") is None or kwargs.get("model_b") is None:
            raise ValueError("TSE requires model_a and model_b model arguments")

        self.config = TSEConfig(
            model_a_path=kwargs["model_a"],
            model_b_path=kwargs["model_b"],
            model_a_device=kwargs.get("model_a_device", "cuda:0"),
            model_b_device=kwargs.get("model_b_device", "cuda:1"),
            dtype=kwargs.get("dtype", "bfloat16"),
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            steps=int(kwargs.get("steps", 128)),
            block_size=int(kwargs.get("block_size", 128)),
            temperature=float(kwargs.get("temperature", 0.0)),
            remasking=kwargs.get("remasking", "low_confidence"),
            stochastic_transfer=kwargs.get("stochastic_transfer", False),
            capture_logits=kwargs.get("capture_logits", False),
        )
        accelerator = accelerate.Accelerator()
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        if self._world_size != 1:
            raise ValueError(
                "TSE requires one process; assign models with "
                "model_a_device and model_b_device"
            )

        models = load_tse_models(self.config)
        self.tokenizer = models.tokenizer
        self.sampler = TSESampler(
            models.model_a,
            models.model_b,
            self.tokenizer,
            self.config.model_a_device,
            self.config.model_b_device,
        )
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.device = torch.device(self.config.model_a_device)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs = []
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            contexts, generation_kwargs = zip(*[instance.args for instance in batch])
            prompts = [
                self.tokenizer(context, return_tensors="pt")["input_ids"][0]
                for context in contexts
            ]
            generated = self.sampler.sample(
                prompts,
                max_new_tokens=self.config.max_new_tokens,
                steps=self.config.steps,
                block_size=self.config.block_size,
                temperature=self.config.temperature,
                remasking=self.config.remasking,
                stochastic_transfer=self.config.stochastic_transfer,
                capture_logits=self.config.capture_logits,
            )
            answers = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
            for answer, prompt, kwargs_for_generation in zip(
                answers, prompts, generation_kwargs
            ):
                answer = answer[len(self.tokenizer.decode(prompt, skip_special_tokens=False)) :]
                for stop_sequence in kwargs_for_generation["until"]:
                    if stop_sequence in answer:
                        answer = answer.split(stop_sequence)[0]
                outputs.append(answer)
        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError("TSE supports generation only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("TSE supports generation only")


register_model("tse_llada")(TSEEvalHarness)


if __name__ == "__main__":
    cli_evaluate()