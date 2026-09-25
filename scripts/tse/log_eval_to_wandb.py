"""Log an lm-evaluation-harness JSON result to Weights & Biases.

Run from the repository root with:

    python scripts/tse/log_eval_to_wandb.py \
        --result-path .logs/eval.json \
        --project dllm-tse \
        --run-name tse-gsm8k-static \
        --mode tse \
        --weighting-mode static
"""

import argparse
import json
import os
from numbers import Number
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument(
        "--project", default=os.environ.get("WANDB_PROJECT", "dllm-tse")
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--weighting-mode", default="none")
    parser.add_argument("--model-a", default="")
    parser.add_argument("--model-b", default="")
    return parser.parse_args()


def flatten_numbers(value, prefix=""):
    metrics = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            metrics.update(flatten_numbers(item, child_prefix))
    elif isinstance(value, Number) and not isinstance(value, bool):
        metrics[prefix] = float(value)
    return metrics


def main() -> None:
    args = parse_args()
    if not args.result_path.is_file():
        raise FileNotFoundError(f"lm-eval result file not found: {args.result_path}")

    with args.result_path.open() as result_file:
        result = json.load(result_file)

    config = {
        "mode": args.mode,
        "weighting_mode": args.weighting_mode,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "result_path": str(args.result_path),
    }
    init_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "config": config,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity

    run = wandb.init(**init_kwargs)
    metrics = flatten_numbers(result.get("results", result))
    if metrics:
        run.log(metrics)
        for key, value in metrics.items():
            run.summary[key] = value
    run.summary["result_path"] = str(args.result_path)
    run.finish()
    print(f"Logged {len(metrics)} metrics to W&B run {args.run_name}")


if __name__ == "__main__":
    main()
