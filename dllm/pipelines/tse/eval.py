"""lm-eval entrypoint for homogeneous or CTCA-aligned TSE.

Run with:
    python -m dllm.pipelines.tse.eval --model tse_llada --model_args ...
"""

from dataclasses import dataclass

import accelerate
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dllm.pipelines.tse.loader import load_tse_models
from dllm.pipelines.tse.models import TSEConfig
from dllm.pipelines.tse.ctca_sampler import CTCATSESampler
from dllm.pipelines.tse.sampler import TSESampler


def _as_bool(value) -> bool:
    if isinstance(value, str):
        if value.casefold() in {"true", "1", "yes"}:
            return True
        if value.casefold() in {"false", "0", "no"}:
            return False
        raise ValueError(f"Invalid boolean model argument: {value}")
    return bool(value)


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
            stochastic_transfer=_as_bool(kwargs.get("stochastic_transfer", False)),
            capture_logits=_as_bool(kwargs.get("capture_logits", False)),
            alpha=float(kwargs.get("alpha", 0.5)),
            temperature_a=float(kwargs.get("temperature_a", 1.0)),
            temperature_b=float(kwargs.get("temperature_b", 1.0)),
            epsilon=float(kwargs.get("epsilon", 1e-9)),
            fusion_device=kwargs.get("fusion_device", "cuda:0"),
            weighting_mode=kwargs.get("weighting_mode", "static"),
            weight_temperature=float(kwargs.get("weight_temperature", 1.0)),
            normalize_entropy=_as_bool(kwargs.get("normalize_entropy", True)),
            ctca_enabled=_as_bool(kwargs.get("ctca_enabled", False)),
            master_model=kwargs.get("master_model", "a"),
            ctca_cache_dir=kwargs.get("ctca_cache_dir", ".cache/ctca"),
            ctca_force_rebuild=_as_bool(kwargs.get("ctca_force_rebuild", False)),
            ctca_projection_temperature=float(
                kwargs.get("ctca_projection_temperature", 0.05)
            ),
            ctca_chunk_size=int(kwargs.get("ctca_chunk_size", 2500)),
            ctca_num_anchors=int(kwargs.get("ctca_num_anchors", 3000)),
            ctca_min_anchors=int(kwargs.get("ctca_min_anchors", 128)),
        )
        self.selection_mode = kwargs.get("selection_mode", "tse")
        self.baseline_model = kwargs.get("baseline_model", "b")
        accelerator = accelerate.Accelerator()
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        if self._world_size != 1:
            raise ValueError(
                "TSE requires one process; assign models with "
                "model_a_device and model_b_device"
            )

        models = load_tse_models(self.config)
        self.auxiliary_tokenizer = None
        if self.config.ctca_enabled:
            auxiliary_name = "b" if self.config.master_model == "a" else "a"
            self.tokenizer = models.tokenizer_for(self.config.master_model)
            self.auxiliary_tokenizer = models.tokenizer_for(auxiliary_name)
            master_device = getattr(
                self.config, f"model_{self.config.master_model}_device"
            )
            auxiliary_device = getattr(self.config, f"model_{auxiliary_name}_device")
            self.sampler = CTCATSESampler(
                models.model_for(self.config.master_model),
                models.model_for(auxiliary_name),
                self.tokenizer,
                self.auxiliary_tokenizer,
                master_device,
                auxiliary_device,
                master_id=self.config.master_model,
                auxiliary_id=auxiliary_name,
                cache_dir=self.config.ctca_cache_dir,
                force_rebuild=self.config.ctca_force_rebuild,
                projection_temperature=self.config.ctca_projection_temperature,
                projection_chunk_size=self.config.ctca_chunk_size,
                num_anchors=self.config.ctca_num_anchors,
                min_anchors=self.config.ctca_min_anchors,
                master_cache_id=(
                    self.config.model_a_path
                    if self.config.master_model == "a"
                    else self.config.model_b_path
                ),
                auxiliary_cache_id=(
                    self.config.model_b_path
                    if auxiliary_name == "b"
                    else self.config.model_a_path
                ),
            )
        else:
            self.tokenizer = models.tokenizer
            self.sampler = TSESampler(
                models.model_a,
                models.model_b,
                self.tokenizer,
                self.config.model_a_device,
                self.config.model_b_device,
            )
        self.batch_size = int(kwargs.get("batch_size", 1))
        active_device = (
            getattr(self.config, f"model_{self.config.master_model}_device")
            if self.config.ctca_enabled
            else self.config.model_a_device
        )
        self.device = torch.device(active_device)
        self._auxiliary_chat_prompts: dict[str, str] = {}

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
        master_prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        if self.auxiliary_tokenizer is not None:
            auxiliary_prompt = self.auxiliary_tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
            self._auxiliary_chat_prompts[master_prompt] = auxiliary_prompt
        return master_prompt

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs = []
        for start in tqdm(
            range(0, len(requests), self.batch_size),
            desc="TSE GSM8K generation",
        ):
            batch = requests[start : start + self.batch_size]
            contexts, generation_kwargs = zip(*[instance.args for instance in batch])
            prompts = [
                self.tokenizer(context, return_tensors="pt")["input_ids"][0]
                for context in contexts
            ]
            auxiliary_prompts = None
            if self.auxiliary_tokenizer is not None:
                auxiliary_prompts = [
                    self.auxiliary_tokenizer(
                        self._auxiliary_chat_prompts.get(context, context),
                        return_tensors="pt",
                    )["input_ids"][0]
                    for context in contexts
                ]
            sampler_inputs = (
                {"auxiliary_inputs": auxiliary_prompts}
                if auxiliary_prompts is not None
                else {}
            )
            generated = self.sampler.sample(
                prompts,
                **sampler_inputs,
                max_new_tokens=self.config.max_new_tokens,
                steps=self.config.steps,
                block_size=self.config.block_size,
                temperature=self.config.temperature,
                remasking=self.config.remasking,
                stochastic_transfer=self.config.stochastic_transfer,
                capture_logits=self.config.capture_logits,
                baseline_model=self.baseline_model,
                selection_mode=self.selection_mode,
                alpha=self.config.alpha,
                temperature_a=self.config.temperature_a,
                temperature_b=self.config.temperature_b,
                epsilon=self.config.epsilon,
                fusion_device=self.config.fusion_device,
                weighting_mode=self.config.weighting_mode,
                weight_temperature=self.config.weight_temperature,
                normalize_entropy=self.config.normalize_entropy,
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
