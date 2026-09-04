"""Create one deterministic Phase-8 lm-eval sample manifest.

Run from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/prepare_p8_manifest.py \
        --stage smoke --task gsm8k_cot \
        --output /tmp/p8_gsm8k_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_PLAN = Path(
    "/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/"
    "dependency_guided/p8_evaluation_plan.json"
)


def load_plan(path: Path) -> dict[str, object]:
    """Load and minimally validate the frozen Phase-8 plan."""
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError("Phase-8 plan must use schema_version=1.")
    if not isinstance(plan.get("tasks"), dict) or not plan["tasks"]:
        raise ValueError("Phase-8 plan must define at least one task.")
    return plan


def stage_bounds(
    plan: dict[str, object],
    *,
    stage: str,
    task: str,
    shard_index: int = 0,
) -> tuple[int, int, int]:
    """Return inclusive start, exclusive stop, and total shard count."""
    tasks = plan["tasks"]
    if task not in tasks:
        raise ValueError(f"Unknown Phase-8 task {task!r}.")
    task_config = tasks[task]
    if stage == "smoke":
        if shard_index != 0:
            raise ValueError("The smoke stage has only shard_index=0.")
        start = int(task_config["smoke_sample"])
        return start, start + 1, 1
    if stage == "primary":
        start_key, stop_key = "primary_start", "primary_stop"
    elif stage == "analysis":
        start_key, stop_key = "analysis_start", "analysis_stop"
    else:
        raise ValueError("stage must be smoke, primary, or analysis.")

    stage_config = plan["sample_stages"][stage]
    shard_size = int(stage_config["shard_size"])
    selection_start = int(task_config[start_key])
    selection_stop = int(task_config[stop_key])
    total = selection_stop - selection_start
    shard_count = math.ceil(total / shard_size)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count - 1}] for "
            f"{stage}/{task}, got {shard_index}."
        )
    start = selection_start + shard_index * shard_size
    stop = min(start + shard_size, selection_stop)
    return start, stop, shard_count


def build_manifest(
    plan: dict[str, object],
    *,
    stage: str,
    task: str,
    shard_index: int = 0,
) -> tuple[dict[str, list[int]], dict[str, int | str]]:
    """Build an lm-eval manifest and its auditable range metadata."""
    start, stop, shard_count = stage_bounds(
        plan,
        stage=stage,
        task=task,
        shard_index=shard_index,
    )
    total_examples = int(plan["tasks"][task]["total_examples"])
    if start < 0 or stop > total_examples or start >= stop:
        raise ValueError(
            f"Invalid frozen range [{start}, {stop}) for {task} with "
            f"{total_examples} examples."
        )
    return (
        {task: list(range(start, stop))},
        {
            "stage": stage,
            "task": task,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "sample_start": start,
            "sample_stop": stop,
            "sample_count": stop - start,
        },
    )


def main() -> None:
    """Parse arguments, write one manifest atomically, and print metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--stage", choices=("smoke", "primary", "analysis"), required=True
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load_plan(args.plan.resolve())
    manifest, metadata = build_manifest(
        plan,
        stage=args.stage,
        task=args.task,
        shard_index=args.shard_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
