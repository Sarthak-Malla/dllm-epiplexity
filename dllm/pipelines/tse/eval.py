"""lm-eval entrypoint for homogeneous or CTCA-aligned TSE.

Run with:
    python -m dllm.pipelines.tse.eval --model tse_llada --model_args ...
"""

import atexit
import os
import time

import accelerate
import torch
from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.instance import Instance
from lm_eval.api.model import CacheHook, LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences
from tqdm import tqdm

from dllm.pipelines.tse.loader import (
    build_tse_sampler,
    load_tse_baseline,
    load_tse_models,
)
from dllm.pipelines.tse.models import TSEConfig


class _RenderedPrompt(str):
    """Keep both native chat prompts on the string passed through lm-eval."""

    auxiliary: str | None

    def __new__(cls, text: str, auxiliary: str | None = None):
        prompt = super().__new__(cls, text)
        prompt.auxiliary = auxiliary
        return prompt


def _as_bool(value) -> bool:
    if isinstance(value, str):
        if value.casefold() in {"true", "1", "yes"}:
            return True
        if value.casefold() in {"false", "0", "no"}:
            return False
        raise ValueError(f"Invalid boolean model argument: {value}")
    return bool(value)


class TSEEvalHarness(LM):
    """Evaluate a greedy single checkpoint or a paired ensemble."""

    def __init__(self, **kwargs):
        super().__init__()
        pretrained = kwargs.get("pretrained")
        model_a, model_b = kwargs.get("model_a"), kwargs.get("model_b")
        single_model = pretrained is not None
        if single_model:
            if model_a is not None or model_b is not None:
                raise ValueError(
                    "Use pretrained for one checkpoint or model_a and model_b for a pair"
                )
            model_a = pretrained
        elif model_a is None or model_b is None:
            raise ValueError("Specify pretrained or both model_a and model_b")

        default_device = (kwargs.get("device") or "cuda:0") if single_model else "cuda:0"
        self.config = TSEConfig(
            model_a_path=model_a,
            model_b_path=model_b,
            model_a_device=kwargs.get("model_a_device", default_device),
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
            ctca_force_rebuild=_as_bool(
                kwargs.get("ctca_force_rebuild", TSEConfig.ctca_force_rebuild)
            ),
            ctca_anchor_temperature=float(
                kwargs.get("ctca_anchor_temperature", 0.01)
            ),
            ctca_projection_temperature=float(
                kwargs.get("ctca_projection_temperature", 0.05)
            ),
            ctca_chunk_size=int(kwargs.get("ctca_chunk_size", 2500)),
            ctca_projection_mode=kwargs.get("ctca_projection_mode", "sparse_topk"),
            ctca_projection_top_k=int(kwargs.get("ctca_projection_top_k", 64)),
            ctca_num_anchors=kwargs.get("ctca_num_anchors", "auto"),
            ctca_min_anchors=int(kwargs.get("ctca_min_anchors", 128)),
        )
        self.selection_mode = kwargs.get(
            "selection_mode", "baseline" if single_model else "tse"
        )
        self.baseline_model = kwargs.get("baseline_model", "a" if single_model else "b")
        if single_model and (
            self.config.ctca_enabled
            or self.selection_mode != "baseline"
            or self.baseline_model != "a"
            or self.config.temperature != 0
            or self.config.remasking != "low_confidence"
            or self.config.stochastic_transfer
        ):
            raise ValueError(
                "The single-checkpoint control uses greedy tokens, max-probability "
                "position ranking, and deterministic transfers"
            )
        self._progress_wandb_enabled = _as_bool(
            kwargs.get("progress_wandb", False)
        )
        self._progress_wandb_project = kwargs.get(
            "progress_wandb_project", os.environ.get("WANDB_PROJECT", "dllm-ensemble")
        )
        self._progress_wandb_entity = kwargs.get(
            "progress_wandb_entity", os.environ.get("WANDB_ENTITY")
        )
        self._progress_wandb_run_name = kwargs.get(
            "progress_wandb_run_name", "tse-progress"
        )
        self._progress_wandb_log_interval = int(
            kwargs.get("progress_wandb_log_interval", 1)
        )
        if self._progress_wandb_log_interval < 1:
            raise ValueError("progress_wandb_log_interval must be positive")
        self._progress_wandb_run = None
        accelerator = accelerate.Accelerator()
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        if self._world_size != 1:
            raise ValueError(
                "TSE requires one process; assign models with "
                "model_a_device and model_b_device"
            )

        self.batch_size = int(kwargs.get("batch_size", 1))
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if single_model:
            self.sampler = load_tse_baseline(self.config)
        else:
            models = load_tse_models(self.config)
            self.sampler = build_tse_sampler(self.config, models)
        if self.config.ctca_enabled:
            self.tokenizer = self.sampler.master_tokenizer
            self.auxiliary_tokenizer = self.sampler.auxiliary_tokenizer
            self.device = self.sampler.master_device
        else:
            self.tokenizer = self.sampler.tokenizer
            self.auxiliary_tokenizer = None
            self.device = self.sampler.model_a_device
        if self._progress_wandb_enabled and self.rank == 0:
            self._init_progress_wandb()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        names = [self.tokenizer.name_or_path]
        if self.auxiliary_tokenizer is not None:
            names.append(self.auxiliary_tokenizer.name_or_path)
        return "tse-native-chat-v1-" + "-".join(name.replace("/", "__") for name in names)

    def set_cache_hook(self, cache_hook: CacheHook) -> None:
        if cache_hook.dbdict is not None:
            raise ValueError("TSE requires fresh decoding. Disable lm-eval response caching.")
        super().set_cache_hook(cache_hook)

    def _init_progress_wandb(self) -> None:
        import wandb

        mode = "tse"
        if self.config.ctca_enabled:
            mode = "ctca"
        elif self.config.model_b_path is None:
            mode = "single_model"
        config = {
            "mode": mode,
            "selection_mode": self.selection_mode,
            "baseline_model": self.baseline_model,
            "weighting_mode": self.config.weighting_mode,
            "model_a": self.config.model_a_path,
            "model_b": self.config.model_b_path,
            "batch_size": self.batch_size,
            "max_new_tokens": self.config.max_new_tokens,
            "steps": self.config.steps,
            "block_size": self.config.block_size,
            "ctca_enabled": self.config.ctca_enabled,
            "ctca_anchor_temperature": self.config.ctca_anchor_temperature,
            "ctca_projection_temperature": self.config.ctca_projection_temperature,
            "ctca_projection_mode": self.config.ctca_projection_mode,
            "ctca_projection_top_k": self.config.ctca_projection_top_k,
            "ctca_num_anchors": self.config.ctca_num_anchors,
            "ctca_min_anchors": self.config.ctca_min_anchors,
        }
        init_kwargs = {
            "project": self._progress_wandb_project,
            "name": self._progress_wandb_run_name,
            "config": config,
        }
        if self._progress_wandb_entity:
            init_kwargs["entity"] = self._progress_wandb_entity
        self._progress_wandb_run = wandb.init(**init_kwargs)
        atexit.register(self._finish_progress_wandb)

    def _finish_progress_wandb(self) -> None:
        if self._progress_wandb_run is not None:
            self._progress_wandb_run.finish()
            self._progress_wandb_run = None

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
        if self.auxiliary_tokenizer is None:
            return _RenderedPrompt(master_prompt)
        auxiliary_prompt = self.auxiliary_tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return _RenderedPrompt(master_prompt, auxiliary_prompt)

    def _encode_prompts(
        self, contexts: tuple[str, ...]
    ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        prompts = []
        auxiliary_prompts = []
        for context in contexts:
            rendered = isinstance(context, _RenderedPrompt)
            tokenize_kwargs = {
                "add_special_tokens": not rendered,
                "return_tensors": "pt",
            }
            encoded = self.tokenizer(str(context), **tokenize_kwargs)
            prompts.append(encoded["input_ids"][0])
            if self.auxiliary_tokenizer is not None:
                auxiliary_text = context.auxiliary if rendered else context
                if auxiliary_text is None:
                    raise ValueError("CTCA chat request is missing its auxiliary prompt")
                encoded = self.auxiliary_tokenizer(auxiliary_text, **tokenize_kwargs)
                auxiliary_prompts.append(encoded["input_ids"][0])
        return prompts, {"auxiliary_inputs": auxiliary_prompts} if auxiliary_prompts else {}

    def _decode_outputs(
        self,
        generated: torch.Tensor,
        prompts: list[torch.Tensor],
        generation_kwargs: tuple[dict, ...],
    ) -> list[str]:
        stop_ids = {
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "eot_token_id", None),
        }
        suffixes = []
        for row, prompt in zip(generated.tolist(), prompts):
            start = prompt.numel()
            suffix = row[start : start + self.config.max_new_tokens]
            end = next(
                (i for i, token_id in enumerate(suffix) if token_id in stop_ids),
                len(suffix),
            )
            suffixes.append(suffix[:end])

        answers = self.tokenizer.batch_decode(
            suffixes, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        eos = getattr(self.tokenizer, "eos_token", None)
        eot = getattr(self.tokenizer, "eot_token", None)
        outputs = []
        for answer, kwargs in zip(answers, generation_kwargs):
            until = kwargs.get("until")
            if isinstance(until, (list, tuple)):
                until = list(until)
            stops = handle_stop_sequences(until, eos)
            stops = handle_stop_sequences(stops, eot)
            end = min(
                (index for stop in stops if stop and (index := answer.find(stop)) >= 0),
                default=len(answer),
            )
            outputs.append(answer[:end])
        return outputs

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs = []
        request_count = len(requests)
        batch_starts = range(0, request_count, self.batch_size)
        total_batches = (request_count + self.batch_size - 1) // self.batch_size
        generation_start_time = time.perf_counter()
        for batch_index, start in enumerate(
            tqdm(batch_starts, desc="TSE generation"), start=1
        ):
            batch_start_time = time.perf_counter()
            batch = requests[start : start + self.batch_size]
            contexts, generation_kwargs = zip(*[instance.args for instance in batch])
            prompts, sampler_inputs = self._encode_prompts(contexts)
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
            batch_outputs = self._decode_outputs(generated, prompts, generation_kwargs)
            outputs.extend(batch_outputs)
            batch_seconds = time.perf_counter() - batch_start_time
            if self._progress_wandb_run is not None and (
                batch_index % self._progress_wandb_log_interval == 0
                or len(outputs) == request_count
            ):
                elapsed = time.perf_counter() - generation_start_time
                examples_done = len(outputs)
                generated_tokens = len(batch) * self.config.max_new_tokens
                average_seconds = elapsed / max(examples_done, 1)
                remaining = request_count - examples_done
                eta_seconds = average_seconds * remaining
                self._progress_wandb_run.log(
                    {
                        "progress/examples_done": examples_done,
                        "progress/examples_total": request_count,
                        "progress/batches_done": batch_index,
                        "progress/batches_total": total_batches,
                        "progress/percent_done": examples_done
                        / max(request_count, 1),
                        "throughput/batch_seconds": batch_seconds,
                        "throughput/examples_per_second": len(batch)
                        / max(batch_seconds, 1e-9),
                        "throughput/tokens_generated_per_second": generated_tokens
                        / max(batch_seconds, 1e-9),
                        "throughput/avg_seconds_per_example": average_seconds,
                        "throughput/eta_minutes": eta_seconds / 60,
                        "outputs/mean_answer_length_chars": (
                            sum(map(len, batch_outputs)) / len(batch_outputs)
                        ),
                    },
                    step=batch_index,
                )
        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError("TSE supports generation only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("TSE supports generation only")


register_model("tse_llada")(TSEEvalHarness)


def main() -> None:
    args = parse_eval_args(setup_parser())
    if args.model == "tse_llada":
        args.use_cache = None
        args.cache_requests = None
    cli_evaluate(args)


if __name__ == "__main__":
    main()
