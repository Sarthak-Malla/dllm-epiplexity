"""Launch one fixed-size candidate-search arm on two GPUs on a Slurm compute node.

Prepare the environment, then launch an arm (D, C, CD, I, IE, or CS with N4/N8):
    if [ -f /home/sarthak.malla/.zshrc ]; then
        source /home/sarthak.malla/.zshrc
    else
        source /apps/local/conda_init.sh
    fi
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --gres=gpu:2 --cpus-per-task=24 --time=04:00:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_ablation.py --arm D4 --limit 300

Use --dry-run to print the configuration without loading a model or writing files.
The default subset is GSM8K test documents 0 through 299, shared by all arms.
One launcher starts two evaluation workers, with 150 documents on each GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
DEFAULT_OUTPUT_ROOT = ROOT / "eval_results/path_selection/ablation/7_fixed_size_candidate_search"
CHECKPOINT = Path(
    "/home/sarthak.malla/.cache/huggingface/hub/"
    "models--GSAI-ML--LLaDA-8B-Instruct/snapshots/"
    "08b83a6feb34df1a6011b80c3c00c7563e963b07"
)
ARMS = {
    "D4": ("soft_full", 4, 0.0),
    "D8": ("soft_full", 8, 0.0),
    "C4": ("top_confidence", 4, 0.0),
    "C8": ("top_confidence", 8, 0.0),
    "CD4": ("soft_full", 4, 1.0),
    "CD8": ("soft_full", 8, 1.0),
    "I4": ("soft_full", 4, 1.0),
    "I8": ("soft_full", 8, 1.0),
    "IE4": ("soft_full", 4, 1.0),
    "IE8": ("soft_full", 8, 1.0),
    "CS4": ("soft_full", 4, 1.0),
    "CS8": ("soft_full", 8, 1.0),
}
SEED_SETTINGS = {
    "I4": ("incoming", 0.0), "I8": ("incoming", 0.0),
    "IE4": ("incoming", 1.0), "IE8": ("incoming", 1.0),
    "CS4": ("confidence", 0.0), "CS8": ("confidence", 0.0),
}
EVALUATION_SEED = "0,1234,1234,1234"
WORKER_COUNT = 2


def model_arguments(arm: str, generation_seed: int) -> dict[str, str]:
    """Set explicit arm controls while keeping the shared evaluation fixed."""
    variant, count, confidence_exponent = ARMS[arm]
    seed_strategy, seed_entropy_weight = SEED_SETTINGS.get(arm, ("legacy", 0.0))
    return {
        "pretrained": str(CHECKPOINT),
        "dtype": "bfloat16",
        "load_in_4bit": "false",
        "max_length": "4096",
        "max_new_tokens": "256",
        "steps": "64",
        "block_size": "64",
        "temperature": "0.0",
        "cfg_scale": "0.0",
        "stochastic_transfer": "false",
        "return_dict": "true",
        "diagnostic_retention": "full",
        "sampler_type": "entropy_drop",
        "proposal_strategy": "dependency",
        "candidate_budget": str(count),
        "candidate_chunk_size": "1",
        "dependency_commit_k": "4",
        "dependency_parallel_variant": variant,
        "dependency_cardinality_strategy": "fixed",
        "dependency_max_action_size": "4",
        # Keep a string: the harness parses a bare "4" as an integer.
        "dependency_action_sizes": "1|2|4",
        "dependency_size_scoring": "per_token",
        "dependency_utility_threshold": "0.0",
        # Recorded for reproducibility; the fixed policy does not use this bound.
        "dependency_entropy_budget": "2.0",
        "dependency_immediate_cost_weight": "1.0",
        "dependency_size_penalty": "0.0",
        "dependency_last_n_layers": "4",
        "dependency_direction": "outgoing",
        "dependency_target_weighting": "entropy",
        "dependency_position_temperature": "1.0",
        "dependency_confidence_exponent": str(confidence_exponent),
        "dependency_seed_strategy": seed_strategy,
        "dependency_seed_entropy_weight": str(seed_entropy_weight),
        "dependency_generation_seed": str(generation_seed),
        "dependency_sink_filter_enabled": "true",
        "dependency_sink_quantile": "0.99",
        "dependency_zero_diagonal": "true",
        "dependency_renormalize_selected_keys": "true",
        "dependency_fallback_strategy": "dependency_only",
        "dependency_conflict_normalization": "max",
        "dependency_conflict_penalty": "1.0",
        "dependency_hard_conflict_threshold": "0.25",
        "dependency_anchor_support_weight": "1.0",
        "dependency_anchor_confidence_threshold": "0.8",
        "diagnostic_metadata": "true",
    }


def parse_args() -> argparse.Namespace:
    """Parse only experiment identity and evaluation-size controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--run-tag", default="fixed_k_v1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=300,
                        help="Even number of GSM8K test documents from the start of the split (default: 300).")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag):
        parser.error("--run-tag must start with a letter/digit and contain only letters, digits, '.', '_', '-'.")
    if args.generation_seed < 0:
        parser.error("--generation-seed must be nonnegative.")
    if not 2 <= args.limit <= 1318 or args.limit % WORKER_COUNT:
        parser.error("--limit must be an even integer from 2 to 1318, giving each GPU the same number of documents.")
    if not args.output_root.is_absolute():
        parser.error("--output-root must be an absolute path.")
    if "," in str(args.output_root):
        parser.error("--output-root cannot contain commas.")
    return args


def source_hashes() -> dict[str, str]:
    """Record sampler and evaluation sources without executing them."""
    paths = sorted((ROOT / "dllm/core/samplers").glob("*.py"))
    paths.extend(sorted((ROOT / "dllm/core/eval").glob("*.py")))
    paths.extend(sorted((ROOT / "dllm/core/schedulers").glob("*.py")))
    paths.extend((Path(__file__).resolve(), ROOT / "examples/path_selection/eval.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main() -> None:
    """Start two evaluation workers and require diagnostics from both ranks."""
    args = parse_args()
    scope = f"limit{args.limit}"
    run_directory = (
        args.output_root / args.run_tag / "gsm8k_cot" / scope
        / f"seed{args.generation_seed}" / args.arm
    )
    settings = model_arguments(args.arm, args.generation_seed)
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", str(WORKER_COUNT), "--num_machines", "1",
        "--gpu_ids", "all", "--main_process_port", "0",
        "--mixed_precision", "no", "--dynamo_backend", "no",
        "--num_cpu_threads_per_process", "12",
        str(ROOT / "examples/path_selection/eval.py"),
        "--model", "llada_path_selection",
        "--model_args", ",".join(f"{key}={value}" for key, value in settings.items()),
        "--tasks", "gsm8k_cot", "--batch_size", "1", "--device", "cuda",
        "--seed", EVALUATION_SEED, "--apply_chat_template", "--num_fewshot", "5",
        "--log_samples", "--output_path", str(run_directory / "results.json"),
        "--limit", str(args.limit),
    ]
    if args.wandb_mode != "disabled":
        command.extend((
            "--wandb_args",
            f"project=dllm-selection-ensemble,group=ablation-7-{args.run_tag}-{scope},"
            f"name=ablation-7-{args.arm}-{scope}-s{args.generation_seed},"
            f"mode={args.wandb_mode},dir={run_directory}",
        ))
    manifest = {
        "arm": args.arm,
        "task": "gsm8k_cot",
        "num_fewshot": 5,
        "evaluation_seed": EVALUATION_SEED,
        "generation_seed": args.generation_seed,
        "limit": args.limit,
        "checkpoint": str(CHECKPOINT),
        "model_args": settings,
        "command": command,
        "run_directory": str(run_directory),
        "world_size": WORKER_COUNT,
        "response_cache": False,
        "source_hashes": source_hashes(),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this launcher inside a Slurm compute allocation using srun.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise SystemExit("Start one launcher in a single Slurm task with two GPUs; it creates both evaluation workers.")
    if not CHECKPOINT.is_dir():
        raise SystemExit(f"Pinned checkpoint is unavailable: {CHECKPOINT}")
    # A fresh directory prevents cached/mixed runs and protects incomplete artifacts.
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join(filter(None, (
            str(ROOT), str(ROOT / "lm-evaluation-harness"), environment.get("PYTHONPATH"),
        ))),
        "HF_HOME": "/home/sarthak.malla/.cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_DATASETS_TRUST_REMOTE_CODE": "True",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DIR": str(run_directory),
        "WANDB_MODE": args.wandb_mode,
        "MPLCONFIGDIR": str(run_directory / "matplotlib"),
        "TMPDIR": str(run_directory / "tmp"),
    })
    for key in ("MPLCONFIGDIR", "TMPDIR"):
        Path(environment[key]).mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    artifact_prefix = "results.json_entropy_drop"
    required_artifacts = [
        run_directory / f"{artifact_prefix}_runtime.json",
        run_directory / f"{artifact_prefix}_diagnostics_manifest.json",
    ]
    for rank in range(WORKER_COUNT):
        for kind in ("diagnostics", "runtime"):
            required_artifacts.append(
                run_directory / f"{artifact_prefix}_{kind}_rank{rank:05d}-of-{WORKER_COUNT:05d}.json"
            )
    for artifact in required_artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Evaluator did not produce required artifact: {artifact}")
    runtime = json.loads(required_artifacts[0].read_text())
    if runtime.get("world_size") != WORKER_COUNT or runtime.get("distributed") is not True:
        raise RuntimeError("Evaluator did not report a completed two-GPU run.")
    for pattern in ("results_*.json", "samples_gsm8k_cot_*.jsonl"):
        if len(list(run_directory.glob(pattern))) != 1:
            raise RuntimeError(f"Expected exactly one {pattern} below {run_directory}.")
    completed = {
        "arm": args.arm,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "returncode": 0,
        "world_size": WORKER_COUNT,
    }
    (run_directory / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(f"Completed {args.arm}: {run_directory}", flush=True)


if __name__ == "__main__":
    main()
