"""Run one worker of the 100-document diagnostic suite in a Slurm allocation.

Source /home/sarthak.malla/.zshrc and activate the dllm conda environment, then use:
    srun -p "$PARTITION" -q "$QUOTATYPE" --gres=gpu:2 --cpus-per-task=24 --time=03:00:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py -- --resume collect
Later stages require --resume before the subcommand. --dry-run prints settings
without loading a model or writing artifacts. This program never submits jobs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
_IMPORT_ROOTS = (str(ROOT), str(ROOT / "lm-evaluation-harness"))
# Existing PYTHONPATH entries must be moved ahead of installed packages too.
sys.path[:] = [*_IMPORT_ROOTS, *(entry for entry in sys.path if entry not in _IMPORT_ROOTS)]

from examples.path_selection.experiments.configs import (
    CHECKPOINT, DEFAULT_OUTPUT, DOCUMENT_IDS, EXPERIMENTS, SNAPSHOT_THRESHOLDS,
    benchmark_configs, benchmark_selector_schedule, reference_config, suite_configuration,
)


def parse_args():
    """Separate optional workload slicing from the selected immutable corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "dllm-selection-ensemble"))
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-stop", type=int, default=100,
                        help="Exclusive endpoint; slicing preserves the 100-document diagnostic manifest.")
    subcommands = parser.add_subparsers(dest="stage", required=True)
    subcommands.add_parser("collect")
    diagnose = subcommands.add_parser("diagnose")
    diagnose.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("--arm", choices=tuple(benchmark_configs()), required=True)
    args = parser.parse_args()
    if not args.output_root.is_absolute():
        parser.error("--output-root must be absolute.")
    if not 0 <= args.doc_start < args.doc_stop <= 100:
        parser.error("Require 0 <= --doc-start < --doc-stop <= 100.")
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("Require 0 <= --worker-index < --worker-count.")
    if args.device not in {"cuda:0", "cuda:1"}:
        parser.error("--device must name cuda:0 or cuda:1 explicitly.")
    return args


def source_hashes():
    """Identify executable sampler, evaluation, model, and task sources."""
    paths = set()
    for directory in (
        ROOT / "dllm/core/samplers", ROOT / "dllm/core/eval",
        ROOT / "dllm/core/schedulers", ROOT / "dllm/pipelines/llada/models",
        ROOT / "dllm/utils", Path(__file__).resolve().parent,
        ROOT / "lm-evaluation-harness/lm_eval/api",
        ROOT / "lm-evaluation-harness/lm_eval/filters",
    ):
        paths.update(directory.glob("*.py"))
    paths.add(ROOT / "examples/path_selection/eval.py")
    paths.add(ROOT / "examples/__init__.py")
    paths.add(ROOT / "examples/path_selection/__init__.py")
    paths.update((ROOT / "lm-evaluation-harness/lm_eval/tasks/gsm8k").glob("*.yaml"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def assigned_document_ids(worker_index, worker_count, document_ids=DOCUMENT_IDS):
    """Return a stable disjoint shard of the selected corpus."""
    if worker_count not in (1, 2) or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid worker assignment.")
    return tuple(doc_id for doc_id in document_ids if doc_id % worker_count == worker_index)


def checkpoint_identity():
    """Record the pinned revision, metadata bytes, and exact weight inventory."""
    return {
        "path": str(CHECKPOINT), "revision": CHECKPOINT.name,
        "metadata_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(CHECKPOINT.glob("*.json"))
        },
        "weight_files": {
            path.name: {"bytes": path.stat().st_size, "resolved_path": str(path.resolve())}
            for pattern in ("*.safetensors", "*.bin") for path in sorted(CHECKPOINT.glob(pattern))
        },
    }


def reset_generation_rng():
    """Make independent problem units replayable regardless of resume order."""
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class EvaluationContext:
    """Use lm-eval's actual requests, filters, and task metrics without response caching."""

    def __init__(self, device="cuda:0"):
        import numpy as np
        import torch
        from lm_eval.tasks import TaskManager, get_task_dict
        from examples.path_selection.eval import LLaDAPathSelectionEvalHarness

        random.seed(0)
        np.random.seed(1234)
        torch.manual_seed(1234)
        self.harness = LLaDAPathSelectionEvalHarness(
            pretrained=str(CHECKPOINT), dtype="bfloat16", load_in_4bit=False,
            batch_size=1, device=device, sampler_type="entropy_drop",
            diagnostic_retention="full", **asdict(reference_config()),
        )
        if self.harness.world_size != 1 or self.harness.accelerator is not None:
            raise RuntimeError("Each experiment worker must own one independent model without distributed collectives.")
        self.task = get_task_dict(["gsm8k_cot"], task_manager=TaskManager())["gsm8k_cot"]
        self.task.set_config(key="num_fewshot", value=5)
        self.task.set_fewshot_seed(seed=1234)
        self.task.build_all_requests(
            limit=100, rank=0, world_size=1, cache_requests=False,
            apply_chat_template=True, fewshot_as_multiturn=False,
            chat_template=self.harness.apply_chat_template,
            tokenizer_name=self.harness.tokenizer_name,
        )
        self.requests = {int(instance.doc_id): instance for instance in self.task.instances}
        if set(self.requests) != set(DOCUMENT_IDS):
            raise RuntimeError("The task did not construct exactly GSM8K test documents 0–99.")
        self.prompts = {
            doc_id: torch.tensor(self.harness.tokenizer(instance.args[0])["input_ids"],
                                 dtype=torch.long, device=self.harness.device)
            for doc_id, instance in self.requests.items()
        }

    def document_records(self):
        from examples.path_selection.experiments.artifacts import stable_hash

        return [{
            "doc_id": doc_id, "prompt": request.args[0],
            "target": self.task.doc_to_target(request.doc),
            "document_hash": stable_hash(request.doc),
            "prompt_hash": stable_hash(request.args[0]),
            "prompt_token_hash": stable_hash(self.prompts[doc_id]),
            "target_hash": stable_hash(self.task.doc_to_target(request.doc)),
            "generation_kwargs": request.args[1],
        } for doc_id, request in sorted(self.requests.items())]

    def score_output(self, doc_id, sequences):
        import dllm.utils

        request = copy.deepcopy(self.requests[doc_id])
        answer = dllm.utils.sample_trim(
            self.harness.tokenizer, sequences.tolist(), [self.prompts[doc_id].tolist()],
        )[0]
        for stop_sequence in request.args[1]["until"]:
            if stop_sequence in answer:
                answer = answer.split(stop_sequence)[0]
        request.resps = [answer]
        request.filtered_resps = {}
        for filter_ensemble in self.task._filters:
            filter_ensemble.apply([request])
        metrics = {
            name: self.task.process_results(request.doc, [filtered])
            for name, filtered in request.filtered_resps.items()
        }
        return {
            "response": answer, "filtered_responses": request.filtered_resps,
            "metrics": metrics,
            "flexible_correct": bool(metrics["flexible-extract"]["exact_match"]),
            "strict_correct": bool(metrics["strict-match"]["exact_match"]),
            "generated_token_ids": sequences[0, len(self.prompts[doc_id]):].tolist(),
        }


@contextmanager
def ledger_transaction(store, accounting, phase, doc_id, state_key=None):
    """Persist disjoint actual-work totals even if a caught exception interrupts a unit."""
    before = accounting.snapshot()
    started = time.perf_counter()
    error = None
    try:
        yield
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        store.put_json("ledger", uuid.uuid4().hex, {
            "phase": phase, "doc_id": doc_id, "state_key": state_key,
            "accounting": accounting.delta(before), "wall_seconds": time.perf_counter() - started,
            "completed": error is None, "error": error,
        })


def collect(context, store, accounting, document_ids, *, expected_document_ids=DOCUMENT_IDS):
    """Capture all threshold states on the adaptive entropy reference trajectory."""
    import torch
    from examples.path_selection.experiments.artifacts import stable_hash
    from examples.path_selection.experiments.diagnostics import candidate_records, prepare_pools

    sampler = context.harness.sampler
    configuration = reference_config()
    for doc_id in document_ids:
        document_key = f"doc{doc_id:05d}"
        if store.has("collection", document_key):
            store.record_reuse("collection", document_key, document_key)
            continue
        with ledger_transaction(store, accounting, "collect", doc_id):
            reset_generation_rng()
            state = sampler.initialize_state([context.prompts[doc_id]], configuration)
            before = accounting.snapshot()
            started = time.perf_counter()
            reached = set()

            def observe(prepared):
                revealed = int((prepared.state.response_mask &
                                (prepared.state.input_ids != context.harness.tokenizer.mask_token_id)).sum())
                pending = [threshold for threshold in SNAPSHOT_THRESHOLDS
                           if threshold not in reached and revealed >= threshold]
                if not pending:
                    return
                pools, _, component_timings = prepare_pools(prepared)
                pool_records = {name: candidate_records(pool) for name, pool in pools.items()}
                for threshold in pending:
                    key = f"{document_key}-r{threshold:03d}"
                    snapshot = prepared.state.state_dict()
                    snapshot_hash = stable_hash(snapshot)
                    if store.has("states", key):
                        previous = store.get_json("states", key)
                        from examples.path_selection.experiments.artifacts import jsonable
                        if previous["state_hash"] != snapshot_hash or previous["pools"] != jsonable(pool_records):
                            raise RuntimeError(f"Reference replay changed snapshot {key}.")
                        store.record_reuse("states", key, key)
                    else:
                        store.put_snapshot(key, snapshot)
                        store.put_json("states", key, {
                            "doc_id": doc_id, "state_key": key, "threshold": threshold,
                            "revealed": revealed, "decision_index": prepared.state.global_step_index,
                            "block_index": prepared.state.block_index,
                            "config_hash": stable_hash(asdict(configuration)),
                            "state_hash": snapshot_hash, "pools": pool_records,
                            **component_timings,
                        })
                    reached.add(threshold)

            with accounting.scope("collection_reference"):
                output = sampler.continue_from_state(state, configuration, observer=observe)
            if reached != set(SNAPSHOT_THRESHOLDS):
                raise RuntimeError(f"Missing threshold states for document {doc_id}: {reached}.")
            store.put_json("collection", document_key, {
                "doc_id": doc_id, "state_keys": [f"{document_key}-r{x:03d}" for x in SNAPSHOT_THRESHOLDS],
                "accounting": accounting.delta(before), "wall_seconds": time.perf_counter() - started,
                **context.score_output(doc_id, output.sequences),
            })
        print(f"Collected document {doc_id}; three reference snapshots saved.", flush=True)
    if all(store.has("collection", f"doc{doc_id:05d}") for doc_id in expected_document_ids):
        store.mark_complete("collect", {"documents": len(expected_document_ids),
                                         "states": 3 * len(expected_document_ids),
                                         "document_ids": list(expected_document_ids)})


def diagnose(context, store, accounting, document_ids, experiment, *, expected_document_ids=DOCUMENT_IDS):
    """Replay each stored state, then intervene without changing the shared corpus."""
    from examples.path_selection.experiments.diagnostics import DiagnosticRunner, candidate_records, prepare_pools
    from dllm.core.samplers.decoding_state import DecodeState

    sampler = context.harness.sampler
    diagnostic = DiagnosticRunner(sampler, store, accounting, context.score_output)
    for doc_id in document_ids:
        for threshold in SNAPSHOT_THRESHOLDS:
            state_key = f"doc{doc_id:05d}-r{threshold:03d}"
            key = f"{experiment}-{state_key}"
            if store.has("diagnostics", key):
                store.record_reuse("diagnostics", key, key)
                continue
            if not store.has("states", state_key):
                raise RuntimeError(f"Collect the shared state before diagnosing it: {state_key}.")
            with ledger_transaction(store, accounting, f"diagnose_{experiment}", doc_id, state_key):
                before = accounting.snapshot()
                started = time.perf_counter()
                state = DecodeState.from_state_dict(store.get_snapshot(state_key),
                                                    device=context.harness.device)
                with accounting.scope("diagnostic_replay_base"):
                    prepared = sampler.prepare_step(state, retain_capture=True)
                    if prepared is None:
                        raise RuntimeError(f"Saved state is already complete: {state_key}.")
                    pools, absolute_dependency, component_timings = prepare_pools(
                        prepared, include_confidence=experiment == "proposals",
                        include_mass=experiment == "attention_mass",
                    )
                original = store.get_json("states", state_key)
                from examples.path_selection.experiments.artifacts import jsonable
                for pool in ("adaptive", "fixed4"):
                    if jsonable(candidate_records(pools[pool])) != original["pools"][pool]:
                        raise RuntimeError(f"Prepared candidate replay changed {state_key}/{pool}.")
                shared = {"state_key": state_key, "doc_id": doc_id}
                if experiment == "selectors":
                    results = diagnostic.selectors(pools, **shared)
                elif experiment in {"precedence", "attention_mass"}:
                    results = diagnostic.precedence(pools, absolute_dependency=absolute_dependency, **shared)
                else:
                    results = diagnostic.proposals(pools, **shared)
                store.put_json("diagnostics", key, {
                    "experiment": experiment, "doc_id": doc_id, "state_key": state_key,
                    "stage": threshold, "revealed": original["revealed"],
                    "decision_index": original["decision_index"], "state_hash": original["state_hash"],
                    "candidate_pools": {name: candidate_records(pool) for name, pool in pools.items()},
                    "accounting": accounting.delta(before), "wall_seconds": time.perf_counter() - started,
                    **component_timings,
                    **results,
                })
                del pools, prepared, state
            print(f"Completed {experiment}: {state_key}.", flush=True)
    if all(store.has("diagnostics", f"{experiment}-doc{doc_id:05d}-r{threshold:03d}")
           for doc_id in expected_document_ids for threshold in SNAPSHOT_THRESHOLDS):
        store.mark_complete(f"diagnose_{experiment}", {"documents": len(expected_document_ids),
                                                      "states": 3 * len(expected_document_ids),
                                                      "document_ids": list(expected_document_ids)})


def benchmark(context, store, accounting, document_ids, arm, *, expected_document_ids=DOCUMENT_IDS,
              configuration=None, save_actions=True):
    """Measure the deployed policy with no extra forwards or diagnostic branches."""
    import torch

    sampler = context.harness.sampler
    configuration = replace(configuration or benchmark_configs()[arm], diagnostic_metadata=False)
    selector_schedule = benchmark_selector_schedule(arm)
    for doc_id in document_ids:
        key = f"{arm}-doc{doc_id:05d}"
        if store.has("benchmarks", key):
            store.record_reuse("benchmarks", key, key)
            continue
        with ledger_transaction(store, accounting, f"benchmark_{arm}", doc_id):
            reset_generation_rng()
            before = accounting.snapshot()
            started = time.perf_counter()
            state = sampler.initialize_state([context.prompts[doc_id]], configuration)
            actions = []
            action_count = committed_positions = 0
            selector_counts = {}
            proposal_seconds = reconstruction_seconds = 0.0
            with accounting.scope(f"benchmark_{arm}"):
                while True:
                    prepared = sampler.prepare_step(state, configuration)
                    if prepared is None:
                        break
                    selector = selector_schedule[
                        "first_action" if state.global_step_index == 0 else "remaining_actions"
                    ]
                    selection = sampler.select_step(prepared, selector=selector)
                    values = None
                    if configuration.commit_mode == "seed_first":
                        values = sampler.probe_seed_first(prepared, int(selection.best_index[0]))
                    sampler.commit_step(
                        state, prepared, selection.best_mask,
                        token_ids=None if values is None else values.token_ids,
                        confidence=None if values is None else values.confidence,
                        selection=selection,
                    )
                    action_count += 1
                    committed_positions += int(selection.best_mask.sum().item())
                    selector_counts[selection.selector] = selector_counts.get(selection.selector, 0) + 1
                    if save_actions:
                        positions = torch.where(selection.best_mask[0])[0]
                        actions.append({"positions": positions.tolist(), "size": int(positions.numel()),
                                        "committed_token_ids": state.input_ids[0, positions].tolist(),
                                        "selected_candidate": selection.best_names[0],
                                        "selector": selection.selector})
                    proposal_seconds += prepared.proposal_seconds
                    reconstruction_seconds += prepared.reconstruction_seconds
                torch.cuda.synchronize(context.harness.device)
            # Correctness extraction is outside generation latency and performs
            # exactly the same trim/filter/metric operations as all other arms.
            generation_seconds = time.perf_counter() - started
            store.put_json("benchmarks", key, {
                "arm": arm, "doc_id": doc_id, "configuration": asdict(configuration),
                "selector_schedule": selector_schedule,
                "accounting": accounting.delta(before), "wall_seconds": generation_seconds,
                "generation_seconds": generation_seconds,
                "action_count": action_count, "committed_positions": committed_positions,
                "selector_counts": selector_counts,
                **({"actions": actions} if save_actions else {}),
                "proposal_seconds": proposal_seconds, "reconstruction_seconds": reconstruction_seconds,
                **context.score_output(doc_id, state.input_ids),
            })
        print(f"Completed benchmark {arm}: document {doc_id}.", flush=True)
    if all(store.has("benchmarks", f"{arm}-doc{doc_id:05d}") for doc_id in expected_document_ids):
        store.mark_complete(f"benchmark_{arm}", {"documents": len(expected_document_ids),
                                                "document_ids": list(expected_document_ids)})


def main():
    """Validate execution workflow before model loading or output creation."""
    invocation_started = time.perf_counter()
    args = parse_args()
    if args.dry_run:
        preview = {"configuration": suite_configuration(), "source_hashes": source_hashes(),
                   "requested_stage": args.stage, "document_work_range": [args.doc_start, args.doc_stop],
                   "worker_index": args.worker_index, "worker_count": args.worker_count,
                   "device": args.device, "wandb_mode": args.wandb_mode,
                   "output_root": str(args.output_root)}
        print(json.dumps(preview, indent=2, default=str))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run on a compute node using the documented srun command; no login-node inference.")
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("Use one Slurm task; the two-GPU launcher starts independent workers, without DDP.")
    if not CHECKPOINT.is_dir():
        raise SystemExit(f"Pinned checkpoint is unavailable: {CHECKPOINT}")
    os.environ.update({"HF_HOME": "/home/sarthak.malla/.cache/huggingface",
                       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "HF_DATASETS_TRUST_REMOTE_CODE": "True", "WANDB_MODE": args.wandb_mode,
                       "TOKENIZERS_PARALLELISM": "false"})
    import torch
    from examples.path_selection.experiments.accounting import ForwardAccounting
    from examples.path_selection.experiments.artifacts import RunStore
    from examples.path_selection.experiments.telemetry import ExperimentLogger

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required in the Slurm allocation.")
    if int(args.device.split(":")[1]) >= torch.cuda.device_count():
        raise SystemExit(f"Requested device {args.device} is unavailable in this allocation.")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(12 if args.worker_count == 2 else 24)
    context = EvaluationContext(device=args.device)
    documents = context.document_records()
    import datasets
    import transformers
    device_properties = torch.cuda.get_device_properties(context.harness.device)
    manifest = {**suite_configuration(), "source_hashes": source_hashes(),
                "worker_count": args.worker_count, "sharding": "doc_id_modulo_worker_count",
                "checkpoint_identity": checkpoint_identity(),
                "runtime": {
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "datasets": datasets.__version__,
                    "gpu_name": device_properties.name,
                    "gpu_memory_bytes": device_properties.total_memory,
                    "compute_capability": [device_properties.major, device_properties.minor],
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                },
                "documents": [{key: value for key, value in document.items()
                               if key not in {"prompt", "target", "generation_kwargs"}}
                              for document in documents]}
    store = RunStore(args.output_root, manifest, resume=args.resume)
    expected_document_ids = assigned_document_ids(args.worker_index, args.worker_count)
    store.put_json("worker", "assignment", {"worker_index": args.worker_index,
                                             "worker_count": args.worker_count,
                                             "document_ids": list(expected_document_ids)})
    for document in documents:
        store.put_json("documents", f"doc{document['doc_id']:05d}", document)
    setup_seconds = time.perf_counter() - invocation_started
    phase = args.stage
    if args.stage == "diagnose":
        phase += f"_{args.experiment}"
    elif args.stage == "benchmark":
        phase += f"_{args.arm}"
    with ExperimentLogger(
        root=store.root, configuration=manifest, stage=phase,
        worker_index=args.worker_index, worker_count=args.worker_count,
        mode=args.wandb_mode, project=args.wandb_project,
        group=args.wandb_group or args.output_root.name, name=args.wandb_name,
    ) as logger, torch.no_grad(), ForwardAccounting(context.harness.model) as accounting:
        store.on_write = logger.log_unit
        error = None
        try:
            document_ids = [doc_id for doc_id in expected_document_ids
                            if args.doc_start <= doc_id < args.doc_stop]
            if args.stage == "collect":
                collect(context, store, accounting, document_ids, expected_document_ids=expected_document_ids)
            elif args.stage == "diagnose":
                diagnose(context, store, accounting, document_ids, args.experiment,
                         expected_document_ids=expected_document_ids)
            else:
                benchmark(context, store, accounting, document_ids, args.arm,
                          expected_document_ids=expected_document_ids)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            store.put_json("invocations", uuid.uuid4().hex, {
                "stage": args.stage, "arm": getattr(args, "arm", None),
                "experiment": getattr(args, "experiment", None),
                "worker_index": args.worker_index, "worker_count": args.worker_count,
                "device": args.device,
                "wandb_mode": args.wandb_mode, "wandb_project": args.wandb_project,
                "wandb_group": args.wandb_group or args.output_root.name,
                "document_work_range": [args.doc_start, args.doc_stop],
                "setup_seconds": setup_seconds,
                "wall_seconds": time.perf_counter() - invocation_started,
                "accounting": accounting.snapshot(), "completed": error is None, "error": error,
                "note": "Uncatchable termination can leave in-flight work unrecorded; ledger units are disjoint.",
            })


if __name__ == "__main__":
    main()
