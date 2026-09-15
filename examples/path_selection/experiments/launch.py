"""Launch two independent experiment workers inside one two-GPU Slurm allocation.

After sourcing /home/sarthak.malla/.zshrc and activating dllm, the user runs:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:2 --cpus-per-task=24 --time=03:00:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py -- --resume collect
Pass runner arguments after --. This launcher never submits a Slurm job. Its
--dry-run prints both commands without starting workers or writing artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
# Prioritize the checkout even when it already appears later in PYTHONPATH.
# The local examples package must resolve before importing third-party modules.
sys.path[:] = [str(ROOT), *(entry for entry in sys.path if entry != str(ROOT))]

from examples.path_selection.experiments.artifacts import atomic_json

RUNNER = ROOT / "examples/path_selection/experiments/runner.py"
BENCHMARK_RUNNER = ROOT / "examples/path_selection/experiments/benchmark.py"
DEFAULT_OUTPUT = ROOT / "eval_results/path_selection/experiments/training_free_v1_two_gpu"
WORKER_COUNT = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-config", type=Path,
                        help="Use the task-generic full-policy worker with this JSON task configuration.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "dllm-selection-ensemble"))
    parser.add_argument("--wandb-group")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    if not args.runner_args:
        parser.error("Pass a runner stage after --, for example: -- --resume collect")
    if not args.output_root.is_absolute():
        parser.error("--output-root must be absolute.")
    if args.benchmark_config is not None and not args.benchmark_config.is_absolute():
        parser.error("--benchmark-config must be absolute.")
    protected = ("--output-root", "--worker-index", "--worker-count", "--device")
    if any(value.split("=", 1)[0] in protected for value in args.runner_args):
        parser.error("The launcher owns output-root, worker-index/count, and device; do not pass them after --.")
    return args


def worker_commands(args):
    """Shard problem assignment while retaining the same task identity in both workers."""
    benchmark_config = getattr(args, "benchmark_config", None)
    entrypoint = BENCHMARK_RUNNER if benchmark_config is not None else RUNNER
    config_args = ["--config", str(benchmark_config)] if benchmark_config is not None else []
    supplied = {value.split("=", 1)[0] for value in args.runner_args if value.startswith("--")}
    logging_args = []
    for option, value in (("--wandb-mode", args.wandb_mode), ("--wandb-project", args.wandb_project),
                          ("--wandb-group", args.wandb_group or args.output_root.name)):
        if option not in supplied:
            logging_args.extend((option, value))
    return [[
        sys.executable, str(entrypoint), *config_args,
        "--output-root", str(args.output_root / "workers" / f"worker{index}"),
        "--worker-index", str(index), "--worker-count", str(WORKER_COUNT),
        "--device", f"cuda:{index}",
        *logging_args, *args.runner_args,
    ] for index in range(WORKER_COUNT)]


def independent_worker_environment():
    """Remove inherited distributed-launch identity before constructing Accelerate."""
    environment = os.environ.copy()
    exact = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
             "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK"}
    prefixes = ("OMPI_", "PMI_", "PMIX_", "MV2_", "TORCHELASTIC_", "ACCELERATE_")
    for name in list(environment):
        if name in exact or name.startswith(prefixes):
            environment.pop(name)
    environment.update({"OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    return environment


def stop_workers(processes):
    """Stop all workers after interruption/failure so no orphan continues inference."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 20
    while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()


def main():
    args = parse_args()
    commands = worker_commands(args)
    if args.dry_run:
        print(json.dumps({"worker_count": WORKER_COUNT, "output_root": str(args.output_root),
                          "commands": commands, "document_assignment": "doc_id % 2 == worker_index",
                          "wandb_mode": args.wandb_mode}, indent=2))
        for command in commands:
            print(shlex.join(command))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Launch this program inside the documented two-GPU Slurm allocation.")
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise SystemExit("Use one Slurm task with two GPUs; this program owns the two worker processes.")
    if (args.output_root / "manifest.json").exists():
        raise SystemExit("Choose a separate two-GPU run root; this directory contains a legacy single-worker manifest.")
    import torch

    if torch.cuda.device_count() < WORKER_COUNT:
        raise SystemExit("The allocation must expose two CUDA GPUs.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    invocation_id = uuid.uuid4().hex
    log_directory = args.output_root / "launches"
    log_directory.mkdir(exist_ok=True)
    started = time.perf_counter()
    processes = []
    record = {"invocation_id": invocation_id, "worker_count": WORKER_COUNT,
              "started_at": datetime.now(timezone.utc).isoformat(), "commands": commands,
              "runner_args": args.runner_args, "completed": False}
    lock_path = args.output_root / "launch.lock"
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another launcher is already writing this run: {args.output_root}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        try:
            environment = independent_worker_environment()
            for index, command in enumerate(commands):
                log_path = log_directory / f"{invocation_id}.worker{index}.log"
                stream = stack.enter_context(log_path.open("w"))
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
                processes.append(process)
                print(f"Worker {index}: device cuda:{index}, documents {index}, {index + 2}, ...; log {log_path}", flush=True)
            while any(process.poll() is None for process in processes):
                failed = [process for process in processes if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError("An experiment worker failed; inspect its log before resuming.")
                time.sleep(0.5)
            record["return_codes"] = [process.returncode for process in processes]
            if any(record["return_codes"]):
                raise RuntimeError("At least one worker failed; the launch is incomplete.")
            record["completed"] = True
        except BaseException as exc:
            stop_workers(processes)
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["return_codes"] = [process.returncode for process in processes]
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            atomic_json(log_directory / f"{invocation_id}.json", record)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    print(f"Both workers completed successfully. Run root: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
