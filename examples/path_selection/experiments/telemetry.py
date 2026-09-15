"""Log scalar experiment progress to W&B without exporting prompts or token vectors.

Use through the experiment runner after sourcing ~/.zshrc and activating dllm:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:2 --cpus-per-task=24 --time=04:00:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py -- --resume collect
Set WANDB_MODE=offline to save locally for later synchronization, or disabled to
skip W&B. Online mode requires the user's configured W&B authentication.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import warnings

from examples.path_selection.experiments.artifacts import atomic_json, stable_hash


TELEMETRY_DEFAULTS = {"mode": "online", "project": "dllm-selection-ensemble"}
_KINDS = ("ledger", "collection", "states", "diagnostics", "benchmarks")
_ACCOUNTING = ("model_calls", "evaluated_rows", "input_tokens", "model_seconds")


def _number(value):
    """Accept finite scalar measurements, excluding text and vectors."""
    return isinstance(value, (bool, int, float)) and math.isfinite(float(value))


def _public_configuration(configuration):
    """Allow only experiment settings; dataset text never enters W&B config."""
    allowed = (
        "schema_version", "suite", "task", "num_fewshot", "generation_seed",
        "checkpoint", "dtype", "batch_size", "evaluation_seeds",
        "snapshot_thresholds", "response_cache", "request_cache",
        "primary_metric", "primary_filter", "evaluation_split",
    )
    result = {key: configuration[key] for key in allowed if key in configuration}
    if "document_ids" in configuration:
        result["document_count"] = len(configuration["document_ids"])
    # Sampler dataclasses contain public settings, but only retain scalar values
    # and short scalar lists, never arbitrary nested dictionaries or text inputs.
    for section in ("reference", "continuation", "fixed4", "sampler", "selector_schedule"):
        if isinstance(configuration.get(section), dict):
            result[section] = {
                key: value for key, value in configuration[section].items()
                if value is None or isinstance(value, (str, bool, int, float))
                or (isinstance(value, (tuple, list)) and len(value) <= 16
                    and all(_number(item) for item in value))
            }
    return result


class ExperimentLogger:
    """A persistent W&B run per worker and stage, rebuilt from completed units.

    Completed artifact identities contribute once to cumulative metrics. On
    restart the totals are recomputed from local artifacts, including units
    written just before a logging failure. W&B may contain a repeated snapshot
    after a crash, but that snapshot does not increment these cumulative totals.
    Ledger records alone supply actual-work totals: nested branch/unit records
    are never added a second time. No artifact upload or model watching occurs.
    """

    def __init__(
        self, root, configuration, stage, worker_index=0, worker_count=1, *,
        mode=None, project="dllm-selection-ensemble", group=None, name=None,
        run_id=None, compact_state=False,
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage):
            raise ValueError("Telemetry stage must be a safe artifact name.")
        if not isinstance(worker_count, int) or worker_count < 1 or not 0 <= worker_index < worker_count:
            raise ValueError("Telemetry worker index must lie inside its worker count.")
        self.root = Path(root).resolve()
        self.compact_state = compact_state
        self.configuration = _public_configuration(configuration)
        self.stage = stage
        self.worker_index, self.worker_count = worker_index, worker_count
        self.mode = mode or os.environ.get("WANDB_MODE", TELEMETRY_DEFAULTS["mode"])
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline, or disabled.")
        self.project = project
        self.group = group or f"{configuration.get('suite', 'training_free')}-{self.root.parent.name}"
        self.name = name or f"{stage}-worker{worker_index:02d}-of-{worker_count:02d}"
        self.identity = {
            "root": str(self.root), "configuration": self.configuration, "stage": stage,
            "worker_index": worker_index, "worker_count": worker_count,
        }
        self.config_hash = stable_hash(self.identity)
        self.path = self.root / "telemetry" / f"{stage}-worker{worker_index:02d}.json"
        self.run_id = run_id or stable_hash(self.identity)[:24]
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old["config_hash"] != self.config_hash:
                raise ValueError("Telemetry resume rejected: stage, worker layout, or configuration changed.")
            if run_id is not None and old["run_id"] != run_id:
                raise ValueError("Telemetry resume requires the existing W&B run ID.")
            if old["project"] != project:
                raise ValueError("Telemetry resume requires the existing W&B project.")
            self.run_id = old["run_id"]
            self.group = old["group"] if group is None else group
            self.name = old["name"] if name is None else name
        self.units = {}
        self.run = None
        self._finished = False
        self._entered = False

    def _belongs(self, kind, record):
        if kind == "ledger":
            return record.get("phase") == self.stage
        if kind in {"collection", "states"}:
            return self.stage == "collect"
        if kind == "diagnostics":
            return self.stage == f"diagnose_{record.get('experiment')}"
        if kind == "benchmarks":
            return self.stage == f"benchmark_{record.get('arm')}"
        return False

    def _branch_correct(self, key):
        if not key:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(key)):
            raise ValueError("Invalid branch identity in telemetry record.")
        path = self.root / "branches" / f"{key}.json"
        if not path.is_file():
            raise ValueError(f"Completed diagnostic references missing branch: {path}")
        return bool(json.loads(path.read_text())["flexible_correct"])

    def _paired(self, metrics, prefix, baseline, alternative):
        left, right = self._branch_correct(baseline), self._branch_correct(alternative)
        if left is None or right is None:
            return
        metrics[f"{prefix}/pairs"] += 1
        metrics[f"{prefix}/wins"] += int(right and not left)
        metrics[f"{prefix}/losses"] += int(left and not right)
        metrics[f"{prefix}/both_correct"] += int(left and right)
        metrics[f"{prefix}/both_wrong"] += int(not left and not right)

    def _metrics(self, kind, record):
        metrics = defaultdict(float)
        if kind == "ledger":
            for field in _ACCOUNTING:
                value = record.get("accounting", {}).get(field, 0)
                if _number(value):
                    metrics[f"work/{field}"] += float(value)
            metrics["work/ledger_units"] = 1
            metrics["work/failed_units"] = int(not record.get("completed", True))
            if _number(record.get("wall_seconds")):
                metrics["work/wall_seconds"] = float(record["wall_seconds"])
            return dict(metrics)
        metrics["progress/completed_units"] = 1
        if kind == "states":
            metrics["progress/states"] = 1
        if kind in {"collection", "benchmarks"}:
            metrics["progress/documents"] = 1
            for filter_name, source in (("flexible", "flexible_correct"), ("strict", "strict_correct"),
                                        ("primary", "primary_correct")):
                if source in record:
                    metrics[f"accuracy/{filter_name}_correct"] = int(bool(record[source]))
                    metrics[f"accuracy/{filter_name}_count"] = 1
            for name, value in record.get("metric_scores", {}).items():
                if _number(value):
                    metrics[f"task_metrics/{name}/sum"] += float(value)
                    metrics[f"task_metrics/{name}/count"] += 1
        if kind == "diagnostics":
            metrics["progress/states"] = 1
        duration = record.get("generation_seconds", record.get("wall_seconds"))
        if _number(duration):
            metrics["latency/unit_seconds"] = float(duration)
            metrics["latency/measured_units"] = 1
        for field in ("proposal_seconds", "reconstruction_seconds"):
            if _number(record.get(field)):
                metrics[f"latency/{field}"] = float(record[field])
        for action in record.get("actions", []):
            if _number(action.get("size")):
                metrics["actions/count"] += 1
                metrics["actions/revealed_positions"] += action["size"]
        if "actions" not in record and "action_count" in record:
            metrics["actions/count"] += record["action_count"]
            metrics["actions/revealed_positions"] += record["committed_positions"]
        for selector, count in record.get("selector_counts", {}).items():
            metrics[f"actions/selector/{selector}"] += count
        for comparison in record.get("comparisons", []):
            prefix = f"selectors/{comparison.get('pool', 'all')}"
            metrics[f"{prefix}/comparisons"] += 1
            metrics[f"{prefix}/agreements"] += int(comparison.get("exact_agreement", False))
            for field in ("jaccard", "entropy_size", "cheap_size"):
                if _number(comparison.get(field)):
                    metrics[f"{prefix}/{field}_sum"] += comparison[field]
            self._paired(metrics, prefix, comparison.get("entropy_branch"), comparison.get("cheap_branch"))
        for probe in record.get("probes", []):
            prefix = f"seed/{probe.get('pool', 'all')}"
            metrics[f"{prefix}/groups"] += 1
            metrics[f"{prefix}/groups_with_flip"] += int(probe.get("any_flip", False))
            for feature in probe.get("features", []):
                metrics[f"{prefix}/companion_observations"] += 1
                metrics[f"{prefix}/flips"] += int(feature.get("flipped", False))
                if _number(feature.get("probability_loss")):
                    metrics[f"{prefix}/probability_loss_sum"] += feature["probability_loss"]
        for group in record.get("groups", []):
            prefix = f"precedence/{group.get('pool', 'all')}/{group.get('label', 'natural')}"
            self._paired(metrics, f"{prefix}/seed_first", group.get("simultaneous_branch"), group.get("seed_first_branch"))
            if group.get("reverse_branch"):
                self._paired(metrics, f"{prefix}/reverse", group.get("simultaneous_branch"), group["reverse_branch"])
        return dict(metrics)

    def _add(self, kind, record, unit_id):
        if not self._belongs(kind, record):
            return False
        key = f"{kind}/{unit_id}"
        metrics = self._metrics(kind, record)
        fingerprint = stable_hash(metrics)
        if key in self.units:
            if self.units[key]["fingerprint"] != fingerprint:
                raise ValueError(f"Telemetry completed unit changed: {key}")
            return False
        self.units[key] = {"kind": kind, "fingerprint": fingerprint, "metrics": metrics,
                           "doc_id": record.get("doc_id")}
        return True

    def _summary(self):
        totals = defaultdict(float)
        docs = set()
        for unit in self.units.values():
            for key, value in unit["metrics"].items():
                totals[key] += value
            if unit["kind"] != "ledger" and unit["doc_id"] is not None:
                docs.add(unit["doc_id"])
        totals["progress/unique_documents"] = len(docs)
        for key, count in list(totals.items()):
            if key.startswith("accuracy/") and key.endswith("_count") and count:
                prefix = key[:-len("_count")]
                totals[f"{prefix}_rate"] = totals[f"{prefix}_correct"] / count
            if key.startswith("task_metrics/") and key.endswith("/count") and count:
                prefix = key[:-len("/count")]
                totals[f"{prefix}/mean"] = totals[f"{prefix}/sum"] / count
            if key.startswith("selectors/") and key.endswith("/comparisons") and count:
                prefix = key[:-len("/comparisons")]
                totals[f"{prefix}/agreement_rate"] = totals[f"{prefix}/agreements"] / count
                for field in ("jaccard", "entropy_size", "cheap_size"):
                    totals[f"{prefix}/mean_{field}"] = totals[f"{prefix}/{field}_sum"] / count
            if key.startswith("seed/") and key.endswith("/companion_observations") and count:
                prefix = key[:-len("/companion_observations")]
                totals[f"{prefix}/flip_rate"] = totals[f"{prefix}/flips"] / count
                totals[f"{prefix}/mean_probability_loss"] = totals[f"{prefix}/probability_loss_sum"] / count
            if key.startswith("seed/") and key.endswith("/groups") and count:
                prefix = key[:-len("/groups")]
                totals[f"{prefix}/group_flip_rate"] = totals[f"{prefix}/groups_with_flip"] / count
        if totals.get("actions/count"):
            totals["actions/mean_size"] = totals["actions/revealed_positions"] / totals["actions/count"]
        if totals.get("latency/measured_units"):
            totals["latency/mean_unit_seconds"] = totals["latency/unit_seconds"] / totals["latency/measured_units"]
        return dict(totals)

    def _persist(self):
        atomic_json(self.path, {
            "run_id": self.run_id, "project": self.project, "group": self.group,
            "name": self.name, "config_hash": self.config_hash, "stage": self.stage,
            "worker_index": self.worker_index, "worker_count": self.worker_count,
            "mode": self.mode, "completed_units": len(self.units), "totals": self._summary(),
            **({} if self.compact_state else {"units": self.units}),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def __enter__(self):
        if self._entered:
            raise RuntimeError("ExperimentLogger cannot be entered twice.")
        self._entered = True
        for kind in _KINDS:
            for path in sorted((self.root / kind).glob("*.json")):
                self._add(kind, json.loads(path.read_text()), path.stem)
        self._persist()
        if self.mode == "disabled":
            return self
        try:
            import wandb

            directory = self.root / "telemetry" / "wandb"
            directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=self.project, group=self.group, name=self.name, id=self.run_id,
                resume="allow", mode=self.mode, dir=str(directory),
                config={**self.configuration, "stage": self.stage,
                        "worker_index": self.worker_index, "worker_count": self.worker_count},
                settings=wandb.Settings(init_timeout=60),
            )
            if self.run is None:
                raise RuntimeError("wandb.init returned no run.")
            self.run.log({**self._summary(), "telemetry/resumed_completed_units": len(self.units)})
        except Exception as exc:
            if self.run is not None:
                try:
                    self.run.finish(exit_code=1)
                except Exception:
                    pass
            raise RuntimeError(
                "W&B initialization failed. Configure W&B authentication/network access, "
                "or rerun with --wandb-mode offline (WANDB_MODE=offline); "
                "use disabled only when tracking should be skipped."
            ) from exc
        return self

    def log_unit(self, kind, record, *, unit_id=None):
        """Log a completed artifact callback once, using only scalar summaries."""
        if self._finished:
            raise RuntimeError("Cannot log after ExperimentLogger.finish.")
        if unit_id is None:
            unit_id = stable_hash({"kind": kind, "stage": self.stage,
                                   "doc_id": record.get("doc_id"), "state_key": record.get("state_key"),
                                   "arm": record.get("arm"), "phase": record.get("phase")})
        if not self._add(kind, record, str(unit_id)):
            return False
        self._persist()
        if self.run is not None:
            values = {**self._summary(), "unit/kind": kind}
            values.update({f"unit_metrics/{key}": value
                           for key, value in self.units[f"{kind}/{unit_id}"]["metrics"].items()})
            for field in ("doc_id", "revealed", "decision_index", "group_size"):
                if _number(record.get(field)):
                    values[f"unit/{field}"] = record[field]
            if isinstance(record.get("state_key"), str):
                values["unit/state_key"] = record["state_key"]
            for field in _ACCOUNTING:
                value = record.get("accounting", {}).get(field)
                if _number(value):
                    values[f"unit/{field}"] = value
            try:
                self.run.log(values)
            except Exception as exc:
                raise RuntimeError("W&B logging failed; completed artifacts remain resumable. Check connectivity or resume in offline mode.") from exc
        return True

    def finish(self, summary=None, error=None):
        """Finish once, retaining cumulative totals and sanitized invocation status."""
        if self._finished:
            return
        self._finished = True
        self._persist()
        if self.run is None:
            return
        values = {**self._summary(), "invocation/succeeded": error is None}
        for key, value in (summary or {}).items():
            if _number(value):
                values[f"invocation/{key}"] = value
        if error is not None:
            values["invocation/error_type"] = type(error).__name__
        try:
            self.run.summary.update(values)
            self.run.finish(exit_code=1 if error is not None else 0)
        except Exception as exc:
            if error is None:
                raise RuntimeError("W&B finalization failed; local telemetry and completed artifacts are preserved.") from exc
            warnings.warn(
                "W&B finalization also failed while handling the experiment error; "
                "local telemetry and completed artifacts are preserved.",
                RuntimeWarning, stacklevel=2,
            )

    def __exit__(self, exc_type, exc, traceback):
        self.finish(error=exc)
