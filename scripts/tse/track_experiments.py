"""Inspect or validate the TSE experiment ledger.

Run from any directory with:

    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/track_experiments.py
    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/track_experiments.py --check
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path(
    "/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json"
)
VALID_STATUSES = {"planned", "running", "completed", "failed", "cancelled"}
VALID_GROUPS = {"tse", "ctca"}
PRIMARY_METRIC = "exact_match,flexible-extract"
SECONDARY_METRIC = "exact_match,strict-match"
METRIC_STDERR = {
    PRIMARY_METRIC: "exact_match_stderr,flexible-extract",
    SECONDARY_METRIC: "exact_match_stderr,strict-match",
}
PROTOCOL_REQUIREMENTS = {
    "tse-gsm8k-v1": {
        "task": "gsm8k_cot",
        "samples": 1319,
        "num_fewshot": 5,
        "metadata": {"max_new_tokens": 512, "steps": 128},
    },
    "ctca-gsm8k-v1": {
        "task": "gsm8k_cot",
        "samples": 1319,
        "num_fewshot": 5,
        "metadata": {"max_new_tokens": 384, "steps": 64},
    },
    "ctca-gsm8k300-v2": {
        "task": "gsm8k_cot",
        "samples": 300,
        "num_fewshot": 5,
        "metadata": {"max_new_tokens": 512, "steps": 128},
    },
    "ctca-gsm8k300-dream-maskgit-zero-v1": {
        "task": "gsm8k_cot",
        "samples": 300,
        "num_fewshot": 0,
        "metadata": {"max_new_tokens": 512, "steps": 128},
    },
    "ctca-gsm8k300-model-prompt-v1": {
        "task": "gsm8k_cot",
        "samples": 300,
        "num_fewshot": 5,
        "metadata": {"max_new_tokens": 512, "steps": 128},
    },
    "ctca-humaneval-v1": {
        "task": "humaneval_instruct",
        "samples": 164,
        "num_fewshot": 0,
        "metadata": {"max_new_tokens": 768, "steps": 192},
    },
}
REQUIRED_FIELDS = {
    "id",
    "status",
    "group",
    "method",
    "models",
    "protocol_id",
    "config_path",
    "result_glob",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate metadata and completed-result artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print resolved records as JSON instead of a Markdown table.",
    )
    parser.add_argument(
        "--group",
        choices=sorted(VALID_GROUPS),
        help="Show only the TSE or CTCA research group.",
    )
    return parser.parse_args()


def load_ledger(path: Path) -> list[dict[str, Any]]:
    """Load experiment records from the versioned JSON ledger."""
    with path.open(encoding="utf-8") as ledger_file:
        payload = json.load(ledger_file)
    return payload["experiments"]


def resolve_result(record: dict[str, Any]) -> Path | None:
    """Return the newest lm-eval result matching an experiment record."""
    matches = [Path(match) for match in glob.glob(record["result_glob"])]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_result(result_path: Path | None) -> dict[str, Any]:
    """Read an lm-eval result, returning an empty mapping when unresolved."""
    if result_path is None:
        return {}
    with result_path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def read_metrics(result_path: Path | None) -> dict[str, float]:
    """Read numeric metrics from a single-task lm-eval result."""
    payload = read_result(result_path)
    tasks = payload.get("results", {})
    if len(tasks) != 1:
        return {}
    task_metrics = next(iter(tasks.values()))
    return {
        key: float(value)
        for key, value in task_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def resolved_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach the resolved artifact and standard metrics to a ledger record."""
    result_path = resolve_result(record)
    payload = read_result(result_path)
    duration = payload.get("total_evaluation_time_seconds")
    return {
        **record,
        "resolved_result_path": str(result_path) if result_path else None,
        "metrics": read_metrics(result_path),
        "duration_hours": float(duration) / 3600.0 if duration else None,
    }


def validate(records: list[dict[str, Any]]) -> list[str]:
    """Return ledger errors; warnings remain visible in normal table output."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        label = record.get("id", f"record {index}")
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue
        if record["id"] in seen_ids:
            errors.append(f"{label}: duplicate experiment id")
        seen_ids.add(record["id"])
        if record["status"] not in VALID_STATUSES:
            errors.append(f"{label}: invalid status {record['status']!r}")
        if record["group"] not in VALID_GROUPS:
            errors.append(f"{label}: invalid group {record['group']!r}")
        if not Path(record["config_path"]).is_file():
            errors.append(f"{label}: config does not exist: {record['config_path']}")
        result_path = resolve_result(record)
        if record["status"] == "completed" and result_path is None:
            errors.append(f"{label}: completed experiment has no result artifact")
        if result_path is not None and not read_metrics(result_path):
            errors.append(
                f"{label}: result has no standard task metrics: {result_path}"
            )
        if result_path is not None:
            payload = read_result(result_path)
            requirements = PROTOCOL_REQUIREMENTS.get(record["protocol_id"])
            if requirements is None:
                errors.append(
                    f"{label}: protocol has no declared requirements: "
                    f"{record['protocol_id']}"
                )
                continue
            task = requirements["task"]
            effective_samples = (
                payload.get("n-samples", {}).get(task, {}).get("effective")
            )
            if effective_samples != requirements["samples"]:
                errors.append(
                    f"{label}: expected {requirements['samples']} effective samples, got "
                    f"{effective_samples!r}"
                )
            n_shot = payload.get("n-shot", {}).get(task)
            if n_shot != requirements["num_fewshot"]:
                errors.append(
                    f"{label}: expected {requirements['num_fewshot']}-shot "
                    f"evaluation, got {n_shot!r}"
                )
            metadata = payload.get("configs", {}).get(task, {}).get("metadata", {})
            for key, expected in requirements["metadata"].items():
                actual = metadata.get(key)
                if actual != expected:
                    errors.append(
                        f"{label}: protocol requires {key}={expected!r}, "
                        f"result has {actual!r}"
                    )
    return errors


def format_score(metrics: dict[str, float], metric: str) -> str:
    """Format a fractional lm-eval score as a percentage."""
    value = metrics.get(metric)
    stderr = metrics.get(METRIC_STDERR[metric])
    if value is None:
        return "—"
    if stderr is None:
        return f"{100.0 * value:.2f}"
    return f"{100.0 * value:.2f} ± {100.0 * stderr:.2f}"


def print_markdown(records: list[dict[str, Any]]) -> None:
    """Print a compact, copyable experiment status table."""
    print(
        "| Experiment | Group | Status | Protocol | "
        "Flexible EM | Strict EM | Pass@1 | Hours | Job |"
    )
    print("|---|---|---|---|---|---:|---:|---:|---|")
    for record in records:
        metrics = record["metrics"]
        duration = record["duration_hours"]
        formatted_duration = "—" if duration is None else f"{duration:.2f}"
        pass_at_1 = next(
            (value for key, value in metrics.items() if key.startswith("pass@1")),
            None,
        )
        formatted_pass_at_1 = "—" if pass_at_1 is None else f"{100 * pass_at_1:.2f}"
        print(
            f"| {record['id']} | {record['group']} | {record['status']} | "
            f"{record['protocol_id']} | {format_score(metrics, PRIMARY_METRIC)} | "
            f"{format_score(metrics, SECONDARY_METRIC)} | "
            f"{formatted_pass_at_1} | "
            f"{formatted_duration} | "
            f"{record.get('job_id') or '—'} |"
        )


def main() -> int:
    """Render the ledger and optionally enforce its invariants."""
    args = parse_args()
    records = load_ledger(args.ledger)
    if args.group:
        records = [record for record in records if record["group"] == args.group]
    resolved = [resolved_record(record) for record in records]
    if args.json:
        print(json.dumps(resolved, indent=2))
    else:
        print_markdown(resolved)

    if args.check:
        errors = validate(records)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"\nValidated {len(records)} experiment records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
