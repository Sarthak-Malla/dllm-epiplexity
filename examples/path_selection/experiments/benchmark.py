"""Benchmark complete selector policies on a configured lm-eval generation task.

The user submits the generic launcher (it prepares the dllm environment):
    sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/full_benchmark.slurm.sh /absolute/path/to/task.json
Dataset size, prompts, filters, and scoring come from lm-eval. No jobs are submitted here.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import sys

ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
sys.path[:] = [str(ROOT), str(ROOT / "lm-evaluation-harness"),
               *(entry for entry in sys.path if entry not in {str(ROOT), str(ROOT / "lm-evaluation-harness")})]

from examples.path_selection.experiments.configs import CHECKPOINT, benchmark_configs, benchmark_selector_schedule
from examples.path_selection.experiments.runner import (
    EvaluationContext, assigned_document_ids, benchmark, checkpoint_identity, reset_generation_rng, source_hashes,
)
from examples.path_selection.experiments.artifacts import RunStore

ARMS = ("reference_cheap", "first_action_entropy", "reference_entropy")


def load_configuration(path):
    """Task-specific choices live in data, never in dataset-name branches."""
    settings = json.loads(path.read_text())
    allowed = {"task", "num_fewshot", "generation", "primary_metric", "primary_filter",
               "report_split_at", "confirm_run_unsafe_code", "apply_chat_template"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark settings: {sorted(unknown)}")
    for key in ("task", "primary_metric", "primary_filter"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f"Configuration needs a nonempty {key}.")
    generation = settings.get("generation", {})
    token_lists = {"suppress_tokens", "begin_suppress_tokens"}
    if set(generation) - {"max_new_tokens", "block_size", "steps", "max_length"} - token_lists:
        raise ValueError("Unsupported generation setting.")
    for key, value in generation.items():
        if key in token_lists:
            if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
                raise ValueError(f"generation.{key} must be a list of nonnegative token IDs.")
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"generation.{key} must be a positive integer.")
    for key in ("num_fewshot", "report_split_at"):
        if key in settings and (type(settings[key]) is not int or settings[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("confirm_run_unsafe_code", "apply_chat_template"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean.")
    return settings


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-stop", type=int,
                        help="Optional exclusive work endpoint; manifest always covers the full task.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "dllm-selection-ensemble"))
    parser.add_argument("--wandb-group")
    args = parser.parse_args()
    if not args.config.is_absolute() or not args.output_root.is_absolute():
        parser.error("--config and --output-root must be absolute.")
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("Invalid worker assignment.")
    if args.doc_start < 0 or (args.doc_stop is not None and args.doc_stop <= args.doc_start):
        parser.error("Require 0 <= doc-start < doc-stop when doc-stop is supplied.")
    return args


def task_requests(task, harness, settings):
    """Preserve lm-eval prompts and request semantics for every evaluation document."""
    if task.OUTPUT_TYPE != "generate_until" or task.config.repeats != 1:
        raise ValueError("This benchmark requires a generate_until task with repeats=1.")
    if getattr(task, "UNSAFE_CODE", False) and not settings.get("confirm_run_unsafe_code", False):
        raise ValueError("The task requires confirm_run_unsafe_code in its configuration.")
    if "num_fewshot" in settings:
        task.set_config(key="num_fewshot", value=settings["num_fewshot"])
    task.set_fewshot_seed(seed=1234)
    # The paired report supports per-document binary primary metrics and scalar
    # mean-aggregated secondary metrics; reject incompatible aggregation explicitly.
    if any(getattr(fn, "__name__", "") != "mean" for fn in task.aggregation().values()):
        raise ValueError("This benchmark currently supports mean-aggregated task metrics.")
    expected = tuple(range(len(task.eval_docs)))
    if not expected:
        raise ValueError("The selected task has no evaluation documents.")
    task.build_all_requests(
        limit=None, rank=0, world_size=1, cache_requests=False,
        apply_chat_template=settings.get("apply_chat_template", True), fewshot_as_multiturn=False,
        chat_template=harness.apply_chat_template, tokenizer_name=harness.tokenizer_name,
    )
    requests = {int(instance.doc_id): instance for instance in task.instances}
    if len(task.instances) != len(expected) or set(requests) != set(expected):
        raise ValueError("Expected exactly one generation request per evaluation document.")
    return expected, requests


class BenchmarkContext(EvaluationContext):
    """Reuse document identities while delegating all task-specific scoring to lm-eval."""

    def __init__(self, settings, device):
        import torch
        from lm_eval.tasks import TaskManager, get_task_dict
        from examples.path_selection.eval import LLaDAPathSelectionEvalHarness

        self.settings = settings
        reset_generation_rng()
        if settings.get("confirm_run_unsafe_code", False):
            # Some task utilities initialize their code evaluator during import.
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        manager = TaskManager()
        tasks = get_task_dict([settings["task"]], task_manager=manager)
        if set(tasks) != {settings["task"]}:
            raise ValueError("Choose one concrete lm-eval task rather than a group or tag.")
        self.task = tasks[settings["task"]]
        self.configs = {name: replace(benchmark_configs()[name], **settings.get("generation", {})) for name in ARMS}
        self.harness = LLaDAPathSelectionEvalHarness(
            pretrained=str(CHECKPOINT), dtype="bfloat16", load_in_4bit=False,
            batch_size=1, device=device, sampler_type="entropy_drop", diagnostic_retention="compact",
            **asdict(self.configs["reference_cheap"]),
        )
        if self.harness.world_size != 1 or self.harness.accelerator is not None:
            raise RuntimeError("Each benchmark worker must own one independent model.")
        self.document_ids, self.requests = task_requests(self.task, self.harness, settings)
        if settings.get("report_split_at", 0) >= len(self.document_ids):
            raise ValueError("report_split_at must leave a nonempty remainder of the task.")
        self.prompts = {
            doc_id: torch.tensor(self.harness.tokenizer(request.args[0])["input_ids"],
                                 dtype=torch.long, device=self.harness.device)
            for doc_id, request in self.requests.items()
        }
        task_path = Path(manager.task_index[settings["task"]]["yaml_path"])
        self.task_sources = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(task_path.parent.rglob("*")) if path.suffix in {".py", ".yaml", ".yml"}
        }

    def score_output(self, doc_id, sequences):
        import dllm.utils

        request = copy.deepcopy(self.requests[doc_id])
        answer = dllm.utils.sample_trim(
            self.harness.tokenizer, sequences.tolist(), [self.prompts[doc_id].tolist()],
        )[0]
        stops = request.args[1].get("until") or []
        for stop in [stops] if isinstance(stops, str) else stops:
            if stop and stop in answer:
                answer = answer.split(stop)[0]
        request.resps = [answer]
        request.filtered_resps = {}
        for filters in self.task._filters:
            filters.apply([request])
        metrics = {name: self.task.process_results(request.doc, [filtered])
                   for name, filtered in request.filtered_resps.items()}
        scores = {}
        for filter_name, values in metrics.items():
            for metric, value in values.items():
                if not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"Expected a finite scalar metric: {metric},{filter_name}.")
                scores[f"{metric},{filter_name}"] = float(value)
        primary = f"{self.settings['primary_metric']},{self.settings['primary_filter']}"
        if primary not in scores or scores[primary] not in (0.0, 1.0):
            raise ValueError(f"Primary metric {primary!r} must return per-document zero or one.")
        return {"response": answer, "metric_scores": scores, "primary_correct": bool(scores[primary])}


def _serialize_configuration(value):
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def main():
    args = parse_args()
    settings = load_configuration(args.config)
    if args.dry_run:
        print(json.dumps({"settings": settings, "arm": args.arm,
                          "selector_schedule": benchmark_selector_schedule(args.arm),
                          "document_count": "discovered from lm-eval at runtime",
                          "document_work_range": [args.doc_start, args.doc_stop]}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run on a compute node using the documented Slurm launcher.")
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("Use one Slurm task with independent model workers.")
    os.environ.update({"HF_HOME": "/home/sarthak.malla/.cache/huggingface", "HF_HUB_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_DATASETS_TRUST_REMOTE_CODE": "True",
                       "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": args.wandb_mode})
    import torch
    import datasets
    import transformers
    from examples.path_selection.experiments.accounting import ForwardAccounting
    from examples.path_selection.experiments.telemetry import ExperimentLogger

    if not torch.cuda.is_available() or int(args.device.split(":")[1]) >= torch.cuda.device_count():
        raise SystemExit("Requested CUDA device is unavailable in this allocation.")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(12 if args.worker_count == 2 else 24)
    context = BenchmarkContext(settings, args.device)
    stop = len(context.document_ids) if args.doc_stop is None else args.doc_stop
    if not 0 <= args.doc_start < stop <= len(context.document_ids):
        raise ValueError("Requested work range lies outside the task's evaluation documents.")
    documents = context.document_records()
    properties = torch.cuda.get_device_properties(context.harness.device)
    manifest = {
        "kind": "policy_benchmark", "schema_version": 1, "suite": "first_action_benchmark",
        "task": settings["task"], "settings": settings,
        "num_fewshot": context.task.config.num_fewshot,
        "primary_metric": settings["primary_metric"], "primary_filter": settings["primary_filter"],
        "document_ids": list(context.document_ids), "worker_count": args.worker_count,
        "generation_seed": 42, "dtype": "bfloat16", "batch_size": 1,
        "evaluation_split": context.task.config.test_split or context.task.config.validation_split,
        "task_configuration": json.loads(json.dumps(context.task.dump_config(), default=_serialize_configuration)),
        "checkpoint": str(CHECKPOINT), "checkpoint_identity": checkpoint_identity(),
        "source_hashes": {**source_hashes(), **context.task_sources},
        "benchmarks": {arm: asdict(config) for arm, config in context.configs.items()},
        "benchmark_selector_schedules": {arm: benchmark_selector_schedule(arm) for arm in ARMS},
        "documents": [{key: value for key, value in document.items()
                       if key not in {"prompt", "target", "generation_kwargs"}} for document in documents],
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "datasets": datasets.__version__,
                    "gpu_name": properties.name, "gpu_memory_bytes": properties.total_memory},
    }
    store = RunStore(args.output_root, manifest, resume=args.resume)
    expected = assigned_document_ids(args.worker_index, args.worker_count, context.document_ids)
    work = [doc_id for doc_id in expected if args.doc_start <= doc_id < stop]
    stage = f"benchmark_{args.arm}"
    logging_config = {**manifest, "sampler": asdict(context.configs[args.arm]),
                      "selector_schedule": benchmark_selector_schedule(args.arm)}
    with ExperimentLogger(root=store.root, configuration=logging_config, stage=stage,
                          worker_index=args.worker_index, worker_count=args.worker_count,
                          mode=args.wandb_mode, project=args.wandb_project,
                          group=args.wandb_group or args.output_root.name, compact_state=True) as logger, \
            torch.no_grad(), ForwardAccounting(context.harness.model) as accounting:
        store.on_write = logger.log_unit
        benchmark(context, store, accounting, work, args.arm, expected_document_ids=expected,
                  configuration=context.configs[args.arm], save_actions=False)


if __name__ == "__main__":
    main()
