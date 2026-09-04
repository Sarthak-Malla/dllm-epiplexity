"""Analyze saved dependency-proxy states without loading a model.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_analysis.py -v
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dllm.core.samplers.diagnostic_io import configuration_fingerprint


ANALYSIS_SCHEMA_VERSION = 1
DEPENDENCY_DIRECTION = "matrix[query, key]: query uses key as context"
ORACLE_FIELDS = {
    "entropy_drop": "entropy_drop_per_heldout",
    "risk_reduction": "risk_reduction_per_heldout",
}
PROPOSAL_NAMES = (
    "confidence",
    "negative_entropy",
    "random",
    "left_to_right",
    "dependency_degree",
    "confidence_x_dependency_degree",
    "proposed_dependency",
)


@dataclass(frozen=True)
class AnalyzedProxyState:
    """Aligned proposal and oracle scores for one saved proxy state."""

    state_id: str
    example_id: str
    task: str
    state_index: int
    mask_ratio: float
    positions: tuple[int, ...]
    proposal_scores: Mapping[str, tuple[float, ...]]
    oracle_scores: Mapping[str, tuple[float, ...]]


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Return a mapping or raise a field-specific validation error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_string(value: object, *, name: str) -> str:
    """Return a nonempty string or raise a field-specific error."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _integer_positions(value: object, *, name: str) -> tuple[int, ...]:
    """Read a nonempty sequence of unique integer positions."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    positions = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in positions):
        raise ValueError(f"{name} must contain integers.")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{name} must not contain duplicates.")
    return positions


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    """Read a finite one-dimensional floating-point vector."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _finite_matrix(value: object, *, name: str) -> np.ndarray:
    """Read a finite square floating-point matrix."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _align_vector(
    values: np.ndarray,
    source_positions: Sequence[int],
    target_positions: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    """Align one vector from source-position order into target order."""
    if values.size != len(source_positions):
        raise ValueError(f"{name} length does not match its positions.")
    if set(source_positions) != set(target_positions):
        raise ValueError(f"{name} positions do not match the oracle positions.")
    index = {position: offset for offset, position in enumerate(source_positions)}
    return values[[index[position] for position in target_positions]]


def _random_score(*, seed: int, state_id: str, position: int) -> float:
    """Return a stable pseudo-random score independent of process hash state."""
    payload = f"{seed}:{state_id}:{position}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def analyze_proxy_state(
    record: Mapping[str, object],
    *,
    dependency_field: str,
    random_seed: int,
    confidence_exponent: float,
) -> AnalyzedProxyState:
    """Validate one saved state and derive every predeclared proposal score."""
    if dependency_field not in {"before_sink", "after_sink"}:
        raise ValueError("dependency_field must be before_sink or after_sink.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    if not math.isfinite(confidence_exponent) or confidence_exponent < 0:
        raise ValueError("confidence_exponent must be finite and nonnegative.")

    state_id = _require_string(record.get("state_id"), name="state_id")
    example_id = _require_string(record.get("example_id"), name="example_id")
    source = _require_mapping(record.get("source"), name="source")
    task = _require_string(source.get("dataset_label"), name="source.dataset_label")
    state_index = record.get("state_index")
    if isinstance(state_index, bool) or not isinstance(state_index, int):
        raise ValueError("state_index must be an integer.")
    mask_ratio = record.get("target_mask_ratio")
    if (
        isinstance(mask_ratio, bool)
        or not isinstance(mask_ratio, (int, float))
        or not math.isfinite(float(mask_ratio))
    ):
        raise ValueError("target_mask_ratio must be finite.")

    active_positions = _integer_positions(
        record.get("active_positions"), name="active_positions"
    )
    active_confidence = _finite_vector(
        record.get("active_confidence"), name="active_confidence"
    )
    active_entropy = _finite_vector(
        record.get("active_entropy"), name="active_entropy"
    )
    if active_confidence.size != len(active_positions):
        raise ValueError("active_confidence length does not match active_positions.")
    if active_entropy.size != len(active_positions):
        raise ValueError("active_entropy length does not match active_positions.")
    if np.any(active_confidence < 0) or np.any(active_confidence > 1):
        raise ValueError("active_confidence must lie in [0, 1].")
    if np.any(active_entropy < 0):
        raise ValueError("active_entropy must be nonnegative.")

    dependency = _require_mapping(record.get("dependency"), name="dependency")
    if dependency.get("direction_convention") != DEPENDENCY_DIRECTION:
        raise ValueError("The dependency direction convention is unexpected.")
    dependency_positions = _integer_positions(
        dependency.get("positions"), name="dependency.positions"
    )
    dependency_matrix = _finite_matrix(
        dependency.get(dependency_field),
        name=f"dependency.{dependency_field}",
    )
    if dependency_matrix.shape != (
        len(dependency_positions),
        len(dependency_positions),
    ):
        raise ValueError("The dependency matrix does not match dependency.positions.")

    oracle = _require_mapping(record.get("oracle"), name="oracle")
    positions = _integer_positions(oracle.get("positions"), name="oracle.positions")
    confidence = _align_vector(
        active_confidence,
        active_positions,
        positions,
        name="active_confidence",
    )
    entropy = _align_vector(
        active_entropy,
        active_positions,
        positions,
        name="active_entropy",
    )
    if set(dependency_positions) != set(positions):
        raise ValueError("dependency.positions do not match oracle.positions.")
    dependency_index = {
        position: offset for offset, position in enumerate(dependency_positions)
    }
    dependency_order = [dependency_index[position] for position in positions]
    dependency_matrix = dependency_matrix[np.ix_(dependency_order, dependency_order)]

    oracle_scores: dict[str, tuple[float, ...]] = {}
    for oracle_name, field_name in ORACLE_FIELDS.items():
        values = _finite_vector(oracle.get(field_name), name=f"oracle.{field_name}")
        if values.size != len(positions):
            raise ValueError(f"oracle.{field_name} length does not match positions.")
        oracle_scores[oracle_name] = tuple(float(value) for value in values)

    # D[j, i] means query j uses candidate/key i. Column sums therefore measure
    # the candidate's incoming influence over the other masked queries.
    dependency_degree = dependency_matrix.sum(axis=0)
    entropy_weighted_dependency = dependency_matrix.T @ entropy
    proposal_scores = {
        "confidence": confidence,
        "negative_entropy": -entropy,
        "random": np.asarray(
            [
                _random_score(seed=random_seed, state_id=state_id, position=position)
                for position in positions
            ],
            dtype=np.float64,
        ),
        "left_to_right": -np.asarray(positions, dtype=np.float64),
        "dependency_degree": dependency_degree,
        "confidence_x_dependency_degree": confidence * dependency_degree,
        "proposed_dependency": (
            np.power(confidence, confidence_exponent)
            * entropy_weighted_dependency
        ),
    }
    return AnalyzedProxyState(
        state_id=state_id,
        example_id=example_id,
        task=task,
        state_index=state_index,
        mask_ratio=float(mask_ratio),
        positions=positions,
        proposal_scores={
            name: tuple(float(value) for value in proposal_scores[name])
            for name in PROPOSAL_NAMES
        },
        oracle_scores=oracle_scores,
    )


def descending_average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return one-based descending ranks, averaging exact ties."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a nonempty finite vector.")
    order = np.argsort(-array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return tuple(float(rank) for rank in ranks)


def spearman_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Compute Spearman correlation with average ranks and defined nulls."""
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must have the same nonzero length.")
    left_ranks = np.asarray(descending_average_ranks(left), dtype=np.float64)
    right_ranks = np.asarray(descending_average_ranks(right), dtype=np.float64)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def top_indices(
    scores: Sequence[float], positions: Sequence[int], k: int
) -> tuple[int, ...]:
    """Select descending top-k indices with absolute-position tie breaking."""
    if len(scores) != len(positions) or not scores:
        raise ValueError("scores and positions must have the same nonzero length.")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 < k <= len(scores):
        raise ValueError("k must be a positive integer no larger than the state.")
    return tuple(
        sorted(range(len(scores)), key=lambda index: (-scores[index], positions[index]))[
            :k
        ]
    )


def state_ranking_metrics(
    proposal_scores: Sequence[float],
    oracle_scores: Sequence[float],
    positions: Sequence[int],
    *,
    recall_ks: Sequence[int],
) -> dict[str, float | None]:
    """Compute within-state correlation, retrieval, and candidate-set regret."""
    if len(proposal_scores) != len(oracle_scores):
        raise ValueError("proposal and oracle score lengths must match.")
    result: dict[str, float | None] = {
        "spearman": spearman_correlation(proposal_scores, oracle_scores)
    }
    oracle_best = max(oracle_scores)
    for k in recall_ks:
        proposal_top = top_indices(proposal_scores, positions, k)
        oracle_top = set(top_indices(oracle_scores, positions, k))
        result[f"recall_at_{k}"] = len(set(proposal_top) & oracle_top) / k
        result[f"regret_at_{k}"] = oracle_best - max(
            oracle_scores[index] for index in proposal_top
        )
    return result


def build_analysis_records(
    states: Sequence[AnalyzedProxyState],
    *,
    recall_ks: Sequence[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build auditable position, state, and flattened metric records."""
    position_records: list[dict[str, object]] = []
    state_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    for state in states:
        proposal_ranks = {
            name: descending_average_ranks(scores)
            for name, scores in state.proposal_scores.items()
        }
        oracle_ranks = {
            name: descending_average_ranks(scores)
            for name, scores in state.oracle_scores.items()
        }
        for index, position in enumerate(state.positions):
            position_records.append(
                {
                    "state_id": state.state_id,
                    "example_id": state.example_id,
                    "task": state.task,
                    "state_index": state.state_index,
                    "mask_ratio": state.mask_ratio,
                    "position": position,
                    "proposal_scores": {
                        name: scores[index]
                        for name, scores in state.proposal_scores.items()
                    },
                    "proposal_ranks": {
                        name: ranks[index]
                        for name, ranks in proposal_ranks.items()
                    },
                    "oracle_scores": {
                        name: scores[index]
                        for name, scores in state.oracle_scores.items()
                    },
                    "oracle_ranks": {
                        name: ranks[index]
                        for name, ranks in oracle_ranks.items()
                    },
                }
            )
        for proposal_name, proposal_scores in state.proposal_scores.items():
            for oracle_name, oracle_scores in state.oracle_scores.items():
                metrics = state_ranking_metrics(
                    proposal_scores,
                    oracle_scores,
                    state.positions,
                    recall_ks=recall_ks,
                )
                state_record = {
                    "state_id": state.state_id,
                    "example_id": state.example_id,
                    "task": state.task,
                    "state_index": state.state_index,
                    "mask_ratio": state.mask_ratio,
                    "position_count": len(state.positions),
                    "proposal": proposal_name,
                    "oracle": oracle_name,
                    "metrics": metrics,
                }
                state_records.append(state_record)
                for metric_name, value in metrics.items():
                    metric_records.append(
                        {
                            "state_id": state.state_id,
                            "example_id": state.example_id,
                            "task": state.task,
                            "state_index": state.state_index,
                            "mask_ratio": state.mask_ratio,
                            "proposal": proposal_name,
                            "oracle": oracle_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )
    return position_records, state_records, metric_records


def _group_seed(base_seed: int, identity: Sequence[object]) -> int:
    """Derive an order-independent NumPy seed for one aggregate row."""
    payload = json.dumps([base_seed, *identity], sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_example_mean(
    values_by_example: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Estimate a mean and percentile interval by resampling examples."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer.")
    example_means = np.asarray(
        [
            float(np.mean(np.asarray(values, dtype=np.float64)))
            for _, values in sorted(values_by_example.items())
            if values
        ],
        dtype=np.float64,
    )
    if example_means.size == 0:
        raise ValueError("At least one example with a finite value is required.")
    if not np.all(np.isfinite(example_means)):
        raise ValueError("Bootstrap inputs must be finite.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        example_means.size,
        size=(samples, example_means.size),
    )
    bootstrap_means = example_means[indices].mean(axis=1)
    return {
        "mean": float(example_means.mean()),
        "ci_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci_high": float(np.percentile(bootstrap_means, 97.5)),
        "example_count": int(example_means.size),
    }


def aggregate_metrics(
    metric_records: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Aggregate state metrics with examples as the resampling unit."""
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    state_counts: dict[tuple[object, ...], int] = defaultdict(int)
    seen_states: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for record in metric_records:
        value = record["value"]
        if value is None:
            continue
        identity = tuple(record[field] for field in group_fields) + (
            record["proposal"],
            record["oracle"],
            record["metric"],
        )
        example_id = str(record["example_id"])
        grouped[identity][example_id].append(float(value))
        state_id = str(record["state_id"])
        if state_id not in seen_states[identity]:
            seen_states[identity].add(state_id)
            state_counts[identity] += 1

    rows = []
    for identity in sorted(grouped, key=lambda value: tuple(map(str, value))):
        group_values = identity[: len(group_fields)]
        proposal, oracle, metric = identity[len(group_fields) :]
        summary = bootstrap_example_mean(
            grouped[identity],
            samples=bootstrap_samples,
            seed=_group_seed(bootstrap_seed, identity),
        )
        row: dict[str, object] = {
            field: value for field, value in zip(group_fields, group_values)
        }
        row.update(
            {
                "proposal": proposal,
                "oracle": oracle,
                "metric": metric,
                **summary,
                "state_count": state_counts[identity],
            }
        )
        rows.append(row)
    return rows


def compare_all_mask_to_later(
    metric_records: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Compute paired example-level later-minus-all-mask differences."""
    grouped: dict[
        tuple[str, str, str, str], dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in metric_records:
        value = record["value"]
        if value is None:
            continue
        stage = "all_mask" if math.isclose(float(record["mask_ratio"]), 1.0) else "later"
        for task_group in ("overall", str(record["task"])):
            identity = (
                task_group,
                str(record["proposal"]),
                str(record["oracle"]),
                str(record["metric"]),
            )
            grouped[identity][str(record["example_id"])][stage].append(float(value))

    rows = []
    for identity in sorted(grouped):
        differences: dict[str, list[float]] = {}
        all_values = []
        later_values = []
        for example_id, stages in grouped[identity].items():
            if not stages["all_mask"] or not stages["later"]:
                continue
            all_mean = float(np.mean(stages["all_mask"]))
            later_mean = float(np.mean(stages["later"]))
            all_values.append(all_mean)
            later_values.append(later_mean)
            differences[example_id] = [later_mean - all_mean]
        if not differences:
            continue
        summary = bootstrap_example_mean(
            differences,
            samples=bootstrap_samples,
            seed=_group_seed(bootstrap_seed, ("all_vs_later", *identity)),
        )
        task, proposal, oracle, metric = identity
        rows.append(
            {
                "task": task,
                "proposal": proposal,
                "oracle": oracle,
                "metric": metric,
                "all_mask_mean": float(np.mean(all_values)),
                "later_mean": float(np.mean(later_values)),
                "difference_later_minus_all": summary["mean"],
                "difference_ci_low": summary["ci_low"],
                "difference_ci_high": summary["ci_high"],
                "example_count": summary["example_count"],
                "higher_is_better": not str(metric).startswith("regret_"),
            }
        )
    return rows


def _write_json(path: Path, value: object) -> None:
    """Atomically write strict, human-readable JSON."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Atomically write strict canonical JSONL."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary_path.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write a nonempty table with stable column order."""
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    fieldnames = list(rows[0])
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def read_proxy_states(
    path: Path,
    *,
    expected_state_count: int,
    expected_configuration_fingerprint: str,
    expected_manifest_sha256: str,
) -> list[dict[str, object]]:
    """Read and validate the complete frozen state JSONL."""
    records = []
    state_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from error
            if not isinstance(record, dict):
                raise ValueError(f"State at {path}:{line_number} is not an object.")
            state_id = _require_string(record.get("state_id"), name="state_id")
            if state_id in state_ids:
                raise ValueError(f"Duplicate state_id {state_id!r}.")
            state_ids.add(state_id)
            if record.get("status") != "completed":
                raise ValueError(f"State {state_id} is not completed.")
            if record.get("configuration_fingerprint") != expected_configuration_fingerprint:
                raise ValueError(f"State {state_id} has an unexpected configuration.")
            source = _require_mapping(record.get("source"), name="source")
            if source.get("manifest_sha256") != expected_manifest_sha256:
                raise ValueError(f"State {state_id} has an unexpected manifest SHA-256.")
            records.append(record)
    if len(records) != expected_state_count:
        raise ValueError(
            f"Expected {expected_state_count} states but found {len(records)}."
        )
    return records


def _sha256_file(path: Path) -> str:
    """Hash one input artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(values: Sequence[float]) -> np.ndarray:
    """Standardize a state-local vector, mapping constants to zero."""
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std())
    if standard_deviation == 0:
        return np.zeros_like(array)
    return (array - array.mean()) / standard_deviation


def write_analysis_plots(
    output_directory: Path,
    states: Sequence[AnalyzedProxyState],
) -> list[Path]:
    """Write multi-panel state-standardized scatter and normalized-rank plots."""
    import matplotlib.pyplot as plt

    plot_paths = []
    for oracle_name in ORACLE_FIELDS:
        figure, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        for axis, proposal_name in zip(axes.flat, PROPOSAL_NAMES):
            proposal_values = []
            oracle_values = []
            for state in states:
                proposal_values.extend(_normalized(state.proposal_scores[proposal_name]))
                oracle_values.extend(_normalized(state.oracle_scores[oracle_name]))
            axis.scatter(proposal_values, oracle_values, s=4, alpha=0.15)
            axis.axhline(0, color="black", linewidth=0.5)
            axis.axvline(0, color="black", linewidth=0.5)
            axis.set_title(proposal_name.replace("_", " "), fontsize=9)
            axis.set_xlabel("proposal z-score within state")
            axis.set_ylabel(f"{oracle_name.replace('_', ' ')} z-score")
        axes.flat[-1].axis("off")
        figure.suptitle(f"Proposal score versus {oracle_name.replace('_', ' ')}")
        path = output_directory / f"scatter_{oracle_name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plot_paths.append(path)

        figure, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        for axis, proposal_name in zip(axes.flat, PROPOSAL_NAMES):
            proposal_ranks = []
            oracle_ranks = []
            for state in states:
                denominator = max(len(state.positions) - 1, 1)
                proposal_ranks.extend(
                    (np.asarray(descending_average_ranks(state.proposal_scores[proposal_name])) - 1)
                    / denominator
                )
                oracle_ranks.extend(
                    (np.asarray(descending_average_ranks(state.oracle_scores[oracle_name])) - 1)
                    / denominator
                )
            axis.scatter(proposal_ranks, oracle_ranks, s=4, alpha=0.15)
            axis.plot([0, 1], [0, 1], color="black", linewidth=0.7)
            axis.set_title(proposal_name.replace("_", " "), fontsize=9)
            axis.set_xlabel("normalized proposal rank (0 = best)")
            axis.set_ylabel("normalized oracle rank (0 = best)")
        axes.flat[-1].axis("off")
        figure.suptitle(f"Proposal rank versus {oracle_name.replace('_', ' ')} rank")
        path = output_directory / f"rank_{oracle_name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plot_paths.append(path)
    return plot_paths


def run_proxy_analysis(
    *,
    states_path: Path,
    source_configuration_path: Path,
    output_directory: Path,
    expected_state_count: int,
    expected_manifest_sha256: str,
    dependency_field: str,
    confidence_exponent: float,
    random_seed: int,
    recall_ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Validate frozen inputs and write the complete P2.6 analysis package."""
    recall_ks = tuple(recall_ks)
    if not recall_ks or tuple(sorted(set(recall_ks))) != recall_ks:
        raise ValueError("recall_ks must be unique and strictly increasing.")
    source_metadata = json.loads(source_configuration_path.read_text())
    if not isinstance(source_metadata, dict):
        raise ValueError("Source configuration metadata must be an object.")
    source_fingerprint = _require_string(
        source_metadata.get("configuration_fingerprint"),
        name="configuration_fingerprint",
    )
    source_configuration = _require_mapping(
        source_metadata.get("configuration"), name="configuration"
    )
    if source_configuration.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Source configuration has an unexpected manifest SHA-256.")
    failures_path = states_path.parent / "failures.jsonl"
    if failures_path.exists() and failures_path.read_text().strip():
        raise ValueError("The source directory contains recorded failures.")

    records = read_proxy_states(
        states_path,
        expected_state_count=expected_state_count,
        expected_configuration_fingerprint=source_fingerprint,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    states = [
        analyze_proxy_state(
            record,
            dependency_field=dependency_field,
            random_seed=random_seed,
            confidence_exponent=confidence_exponent,
        )
        for record in records
    ]
    if any(max(recall_ks) > len(state.positions) for state in states):
        raise ValueError("A requested Recall@K exceeds a state's position count.")

    analysis_configuration = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "states_path": str(states_path.resolve()),
        "states_sha256": _sha256_file(states_path),
        "source_configuration_path": str(source_configuration_path.resolve()),
        "source_configuration_fingerprint": source_fingerprint,
        "expected_state_count": expected_state_count,
        "expected_manifest_sha256": expected_manifest_sha256,
        "dependency_field": dependency_field,
        "dependency_direction": DEPENDENCY_DIRECTION,
        "confidence_exponent": confidence_exponent,
        "random_seed": random_seed,
        "recall_ks": list(recall_ks),
        "oracle_fields": ORACLE_FIELDS,
        "proposals": {
            "confidence": "c_i",
            "negative_entropy": "-H_i",
            "random": "SHA256(seed, state_id, absolute_position)",
            "left_to_right": "-absolute_position",
            "dependency_degree": "sum_j D[j,i]",
            "confidence_x_dependency_degree": "c_i * sum_j D[j,i]",
            "proposed_dependency": "c_i^eta * sum_j D[j,i] * H_j",
        },
        "bootstrap": {
            "unit": "example",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
            "state_aggregation": "mean_within_example",
        },
        "regret_definition": (
            "max oracle over all positions minus max oracle in proposal top-k"
        ),
    }
    analysis_fingerprint = configuration_fingerprint(analysis_configuration)
    analysis_metadata = {
        "analysis_configuration_fingerprint": analysis_fingerprint,
        "configuration": analysis_configuration,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration_path = output_directory / "analysis_configuration.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text())
        if existing != analysis_metadata:
            raise ValueError("Output directory contains a different analysis configuration.")
    else:
        _write_json(configuration_path, analysis_metadata)

    position_records, state_records, metric_records = build_analysis_records(
        states, recall_ks=recall_ks
    )
    overall = aggregate_metrics(
        metric_records,
        group_fields=(),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_mask_ratio = aggregate_metrics(
        metric_records,
        group_fields=("mask_ratio",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_task = aggregate_metrics(
        metric_records,
        group_fields=("task",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_task_mask_ratio = aggregate_metrics(
        metric_records,
        group_fields=("task", "mask_ratio"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    all_mask_vs_later = compare_all_mask_to_later(
        metric_records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    _write_jsonl(output_directory / "position_scores.jsonl", position_records)
    _write_jsonl(output_directory / "per_state_metrics.jsonl", state_records)
    _write_csv(output_directory / "overall.csv", overall)
    _write_csv(output_directory / "by_mask_ratio.csv", by_mask_ratio)
    _write_csv(output_directory / "by_task.csv", by_task)
    _write_csv(output_directory / "by_task_mask_ratio.csv", by_task_mask_ratio)
    _write_csv(output_directory / "all_mask_vs_later.csv", all_mask_vs_later)
    plot_paths = write_analysis_plots(output_directory, states)

    summary: dict[str, object] = {
        "status": "completed",
        "analysis_configuration_fingerprint": analysis_fingerprint,
        "source_configuration_fingerprint": source_fingerprint,
        "states_sha256": analysis_configuration["states_sha256"],
        "state_count": len(states),
        "example_count": len({state.example_id for state in states}),
        "position_count": len(position_records),
        "task_counts": {
            task: len([state for state in states if state.task == task])
            for task in sorted({state.task for state in states})
        },
        "mask_ratio_counts": {
            str(mask_ratio): len(
                [state for state in states if state.mask_ratio == mask_ratio]
            )
            for mask_ratio in sorted({state.mask_ratio for state in states}, reverse=True)
        },
        "proposals": list(PROPOSAL_NAMES),
        "oracles": list(ORACLE_FIELDS),
        "recall_ks": list(recall_ks),
        "bootstrap_samples": bootstrap_samples,
        "output_files": sorted(
            [
                str(configuration_path.resolve()),
                str((output_directory / "position_scores.jsonl").resolve()),
                str((output_directory / "per_state_metrics.jsonl").resolve()),
                str((output_directory / "overall.csv").resolve()),
                str((output_directory / "by_mask_ratio.csv").resolve()),
                str((output_directory / "by_task.csv").resolve()),
                str((output_directory / "by_task_mask_ratio.csv").resolve()),
                str((output_directory / "all_mask_vs_later.csv").resolve()),
                *(str(path.resolve()) for path in plot_paths),
            ]
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary
