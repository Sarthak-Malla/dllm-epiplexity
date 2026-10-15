"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import atexit
import dataclasses
from dataclasses import dataclass
import os
import time

import accelerate
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.eval.running_metrics import batch_running_accuracy
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))
        self._progress_wandb_enabled = self._as_bool(
            kwargs.get("progress_wandb", False)
        )
        self._progress_wandb_project = kwargs.get(
            "progress_wandb_project", os.environ.get("WANDB_PROJECT", "dllm-tse")
        )
        self._progress_wandb_entity = kwargs.get(
            "progress_wandb_entity", os.environ.get("WANDB_ENTITY")
        )
        self._progress_wandb_run_name = kwargs.get(
            "progress_wandb_run_name", "native-progress"
        )
        self._progress_wandb_group = kwargs.get("progress_wandb_group")
        self._progress_wandb_log_interval = int(
            kwargs.get("progress_wandb_log_interval", 1)
        )
        self._progress_wandb_run = None
        if self._progress_wandb_enabled and self.rank == 0:
            self._init_progress_wandb()

    @staticmethod
    def _as_bool(value) -> bool:
        """Parse a permissive boolean model argument."""
        if isinstance(value, str):
            if value.casefold() in {"true", "1", "yes"}:
                return True
            if value.casefold() in {"false", "0", "no"}:
                return False
            raise ValueError(f"Invalid boolean model argument: {value}")
        return bool(value)

    def _init_progress_wandb(self) -> None:
        """Open the lightweight batch-progress W&B run."""
        import wandb

        init_kwargs = {
            "project": self._progress_wandb_project,
            "name": self._progress_wandb_run_name,
            "config": {
                "model": self.model_args.model_name_or_path,
                "batch_size": self.batch_size,
                "sampler": dataclasses.asdict(self.sampler_config),
            },
        }
        if self._progress_wandb_entity:
            init_kwargs["entity"] = self._progress_wandb_entity
        if self._progress_wandb_group:
            init_kwargs["group"] = self._progress_wandb_group
        self._progress_wandb_run = wandb.init(**init_kwargs)
        atexit.register(self._finish_progress_wandb)

    def _finish_progress_wandb(self) -> None:
        """Close the progress run once, including during interpreter shutdown."""
        if self._progress_wandb_run is not None:
            self._progress_wandb_run.finish()
            self._progress_wandb_run = None

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
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    # ── Unified generate_until scaffolding ────────────────────────────

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        out: list[str] = []
        generation_start = time.perf_counter()
        running_correct = 0.0
        running_scored = 0
        total_batches = (len(requests) + self.batch_size - 1) // self.batch_size

        for batch_index, batch_start in enumerate(
            tqdm(range(0, len(requests), self.batch_size), desc="Generating..."),
            start=1,
        ):
            batch_timer = time.perf_counter()
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

            generated_ids = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=False,
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

            task_name = batch[0].task_name if batch else None
            batch_accuracy = batch_running_accuracy(
                task_name,
                [instance.doc for instance in batch],
                out[-len(batch) :],
            )
            if batch_accuracy is not None:
                running_correct += batch_accuracy * len(batch)
                running_scored += len(batch)
            if self._progress_wandb_run is not None and (
                batch_index % self._progress_wandb_log_interval == 0
                or len(out) == len(requests)
            ):
                elapsed = time.perf_counter() - generation_start
                batch_seconds = time.perf_counter() - batch_timer
                payload = {
                    "progress/examples_done": len(out),
                    "progress/examples_total": len(requests),
                    "progress/batches_done": batch_index,
                    "progress/batches_total": total_batches,
                    "progress/percent_done": len(out) / max(len(requests), 1),
                    "throughput/batch_seconds": batch_seconds,
                    "throughput/examples_per_second": len(batch)
                    / max(batch_seconds, 1e-9),
                    "throughput/eta_minutes": (
                        elapsed / max(len(out), 1) * (len(requests) - len(out)) / 60
                    ),
                }
                if batch_accuracy is not None:
                    payload["quality/batch_accuracy"] = batch_accuracy
                    payload["quality/running_accuracy"] = (
                        running_correct / running_scored
                    )
                self._progress_wandb_run.log(payload, step=batch_index)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
