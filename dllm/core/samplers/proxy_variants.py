"""Analyze predeclared dependency-score variants on frozen proxy states.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_variant_analysis.py -v
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import numpy as np

from dllm.core.samplers.diagnostic_io import configuration_fingerprint
from dllm.core.samplers.proxy_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    DEPENDENCY_DIRECTION,
    ORACLE_FIELDS,
    AnalyzedProxyState,
    _align_vector,
    _finite_matrix,
    _finite_vector,
    _group_seed,
    _integer_positions,
    _require_mapping,
    _require_string,
    _sha256_file,
    _write_csv,
    _write_json,
    _write_jsonl,
    aggregate_metrics,
    analyze_proxy_state,
    bootstrap_example_mean,
    build_analysis_records,
    read_proxy_states,
)


VARIANT_ANALYSIS_SCHEMA_VERSION = 1
BASELINE_NAMES = (
    "confidence",
    "negative_entropy",
    "random",
    "left_to_right",
)
DEPENDENCY_FIELDS = ("before_sink", "after_sink")
DEPENDENCY_DIRECTIONS = ("incoming", "outgoing", "symmetric")
TARGET_WEIGHTINGS = ("uniform", "entropy")
PAIRED_METRICS = ("spearman", "recall_at_4", "regret_at_4")


def _format_exponent(value: float) -> str:
    """Format a nonnegative exponent as a stable proposal-name component."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("confidence exponents must be finite and nonnegative.")
    return format(value, "g").replace(".", "p")


def dependency_variant_name(
    *,
    dependency_field: str,
    direction: str,
    confidence_exponent: float,
    target_weighting: str,
) -> str:
    """Return a stable, human-readable name for one dependency-score variant."""
    if dependency_field not in DEPENDENCY_FIELDS:
        raise ValueError(f"Unsupported dependency field: {dependency_field!r}.")
    if direction not in DEPENDENCY_DIRECTIONS:
        raise ValueError(f"Unsupported dependency direction: {direction!r}.")
    if target_weighting not in TARGET_WEIGHTINGS:
        raise ValueError(f"Unsupported target weighting: {target_weighting!r}.")
    sink_name = dependency_field.removesuffix("_sink")
    eta = _format_exponent(float(confidence_exponent))
    return f"dep_{sink_name}_{direction}_eta{eta}_{target_weighting}"


def build_variant_catalog(
    *,
    dependency_fields: Sequence[str],
    directions: Sequence[str],
    confidence_exponents: Sequence[float],
    target_weightings: Sequence[str],
) -> list[dict[str, object]]:
    """Validate a factorial grid and return its ordered variant metadata."""
    dependency_fields = tuple(dependency_fields)
    directions = tuple(directions)
    confidence_exponents = tuple(float(value) for value in confidence_exponents)
    target_weightings = tuple(target_weightings)
    dimensions = {
        "dependency_fields": dependency_fields,
        "directions": directions,
        "confidence_exponents": confidence_exponents,
        "target_weightings": target_weightings,
    }
    for name, values in dimensions.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and contain no duplicates.")
    if any(value not in DEPENDENCY_FIELDS for value in dependency_fields):
        raise ValueError("dependency_fields contains an unsupported value.")
    if any(value not in DEPENDENCY_DIRECTIONS for value in directions):
        raise ValueError("directions contains an unsupported value.")
    if any(value not in TARGET_WEIGHTINGS for value in target_weightings):
        raise ValueError("target_weightings contains an unsupported value.")
    for value in confidence_exponents:
        _format_exponent(value)

    catalog = []
    for dependency_field in dependency_fields:
        for direction in directions:
            for confidence_exponent in confidence_exponents:
                for target_weighting in target_weightings:
                    proposal = dependency_variant_name(
                        dependency_field=dependency_field,
                        direction=direction,
                        confidence_exponent=confidence_exponent,
                        target_weighting=target_weighting,
                    )
                    if direction == "incoming":
                        influence = "sum_j D[j,i] w_j"
                    elif direction == "outgoing":
                        influence = "sum_j D[i,j] w_j"
                    else:
                        influence = "sum_j 0.5*(D[j,i]+D[i,j]) w_j"
                    catalog.append(
                        {
                            "proposal": proposal,
                            "dependency_field": dependency_field,
                            "direction": direction,
                            "confidence_exponent": confidence_exponent,
                            "target_weighting": target_weighting,
                            "formula": f"c_i^{confidence_exponent:g} * {influence}",
                        }
                    )
    proposal_names = [str(row["proposal"]) for row in catalog]
    if len(set(proposal_names)) != len(proposal_names):
        raise ValueError("Variant grid produced duplicate proposal names.")
    return catalog


def _aligned_state_inputs(
    record: Mapping[str, object],
    positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object], list[int]]:
    """Read confidence, entropy, and matrix alignment shared by all variants."""
    active_positions = _integer_positions(
        record.get("active_positions"), name="active_positions"
    )
    confidence = _align_vector(
        _finite_vector(record.get("active_confidence"), name="active_confidence"),
        active_positions,
        positions,
        name="active_confidence",
    )
    entropy = _align_vector(
        _finite_vector(record.get("active_entropy"), name="active_entropy"),
        active_positions,
        positions,
        name="active_entropy",
    )
    dependency = _require_mapping(record.get("dependency"), name="dependency")
    if dependency.get("direction_convention") != DEPENDENCY_DIRECTION:
        raise ValueError("The dependency direction convention is unexpected.")
    dependency_positions = _integer_positions(
        dependency.get("positions"), name="dependency.positions"
    )
    if set(dependency_positions) != set(positions):
        raise ValueError("dependency.positions do not match oracle.positions.")
    dependency_index = {
        position: offset for offset, position in enumerate(dependency_positions)
    }
    dependency_order = [dependency_index[position] for position in positions]
    return confidence, entropy, dependency, dependency_order


def analyze_proxy_state_variants(
    record: Mapping[str, object],
    *,
    variant_catalog: Sequence[Mapping[str, object]],
    random_seed: int,
) -> AnalyzedProxyState:
    """Derive frozen baselines and all predeclared variants for one state."""
    if not variant_catalog:
        raise ValueError("variant_catalog must not be empty.")
    base = analyze_proxy_state(
        record,
        dependency_field="after_sink",
        random_seed=random_seed,
        confidence_exponent=1.0,
    )
    proposal_scores: dict[str, tuple[float, ...]] = {
        name: base.proposal_scores[name] for name in BASELINE_NAMES
    }
    confidence, entropy, dependency, dependency_order = _aligned_state_inputs(
        record, base.positions
    )
    matrices: dict[str, np.ndarray] = {}

    for specification in variant_catalog:
        proposal = _require_string(specification.get("proposal"), name="proposal")
        dependency_field = _require_string(
            specification.get("dependency_field"), name="dependency_field"
        )
        direction = _require_string(
            specification.get("direction"), name="direction"
        )
        target_weighting = _require_string(
            specification.get("target_weighting"), name="target_weighting"
        )
        confidence_exponent = specification.get("confidence_exponent")
        if (
            isinstance(confidence_exponent, bool)
            or not isinstance(confidence_exponent, (int, float))
        ):
            raise ValueError("confidence_exponent must be numeric.")
        expected_name = dependency_variant_name(
            dependency_field=dependency_field,
            direction=direction,
            confidence_exponent=float(confidence_exponent),
            target_weighting=target_weighting,
        )
        if proposal != expected_name:
            raise ValueError("Variant proposal name does not match its specification.")
        if proposal in proposal_scores:
            raise ValueError(f"Duplicate proposal {proposal!r}.")

        if dependency_field not in matrices:
            matrix = _finite_matrix(
                dependency.get(dependency_field),
                name=f"dependency.{dependency_field}",
            )
            expected_shape = (len(dependency_order), len(dependency_order))
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"dependency.{dependency_field} does not match its positions."
                )
            matrices[dependency_field] = matrix[
                np.ix_(dependency_order, dependency_order)
            ]
        matrix = matrices[dependency_field]
        weights = entropy if target_weighting == "entropy" else np.ones_like(entropy)
        if direction == "incoming":
            influence = matrix.T @ weights
        elif direction == "outgoing":
            influence = matrix @ weights
        else:
            influence = (0.5 * (matrix + matrix.T)) @ weights
        scores = np.power(confidence, float(confidence_exponent)) * influence
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Variant {proposal!r} produced a nonfinite score.")
        proposal_scores[proposal] = tuple(float(value) for value in scores)

    return AnalyzedProxyState(
        state_id=base.state_id,
        example_id=base.example_id,
        task=base.task,
        state_index=base.state_index,
        mask_ratio=base.mask_ratio,
        positions=base.positions,
        proposal_scores=proposal_scores,
        oracle_scores=base.oracle_scores,
    )


def paired_variant_comparisons(
    metric_records: Sequence[Mapping[str, object]],
    *,
    variant_names: Sequence[str],
    baseline_names: Sequence[str],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Compute paired advantages, with examples as the resampling unit."""
    variant_names = tuple(variant_names)
    baseline_names = tuple(baseline_names)
    metrics = tuple(metrics)
    if not variant_names or not baseline_names or not metrics:
        raise ValueError("Variants, baselines, and metrics must be nonempty.")

    index: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for record in metric_records:
        value = record.get("value")
        if value is None:
            continue
        identity = (
            str(record["state_id"]),
            str(record["proposal"]),
            str(record["oracle"]),
            str(record["metric"]),
        )
        if identity in index:
            raise ValueError(f"Duplicate metric record {identity!r}.")
        index[identity] = record

    grouped: dict[
        tuple[object, ...], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple[object, ...], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "ties": 0, "losses": 0, "state_count": 0}
    )
    for identity, variant_record in index.items():
        state_id, proposal, oracle, metric = identity
        if proposal not in variant_names or metric not in metrics:
            continue
        for baseline in baseline_names:
            baseline_record = index.get((state_id, baseline, oracle, metric))
            if baseline_record is None:
                raise ValueError(
                    f"Missing paired baseline {baseline!r} for state {state_id!r}."
                )
            group_values = tuple(variant_record[field] for field in group_fields)
            group_identity = group_values + (proposal, baseline, oracle, metric)
            variant_value = float(variant_record["value"])
            baseline_value = float(baseline_record["value"])
            if metric.startswith("regret_"):
                advantage = baseline_value - variant_value
            else:
                advantage = variant_value - baseline_value
            grouped[group_identity][str(variant_record["example_id"])].append(
                advantage
            )
            counts[group_identity]["state_count"] += 1
            if advantage > 1e-12:
                counts[group_identity]["wins"] += 1
            elif advantage < -1e-12:
                counts[group_identity]["losses"] += 1
            else:
                counts[group_identity]["ties"] += 1

    rows = []
    for identity in sorted(grouped, key=lambda value: tuple(map(str, value))):
        group_values = identity[: len(group_fields)]
        proposal, baseline, oracle, metric = identity[len(group_fields) :]
        summary = bootstrap_example_mean(
            grouped[identity],
            samples=bootstrap_samples,
            seed=_group_seed(bootstrap_seed, ("paired", *identity)),
        )
        row: dict[str, object] = {
            field: value for field, value in zip(group_fields, group_values)
        }
        row.update(
            {
                "proposal": proposal,
                "baseline": baseline,
                "oracle": oracle,
                "metric": metric,
                "mean_advantage": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "example_count": summary["example_count"],
                **counts[identity],
                "positive_favors_proposal": True,
            }
        )
        rows.append(row)
    return rows


def _recoverability_accounting() -> dict[str, object]:
    """Describe exactly which P2.7 dimensions are present in frozen records."""
    return {
        "recoverable": [
            {
                "dimension": "sink_filtering",
                "values": list(DEPENDENCY_FIELDS),
                "source": "Both matrices were saved per state.",
            },
            {
                "dimension": "direction_and_symmetry",
                "values": list(DEPENDENCY_DIRECTIONS),
                "source": "The full directed active-mask matrix was saved.",
            },
            {
                "dimension": "confidence_exponent",
                "source": "Aligned active confidence was saved.",
            },
            {
                "dimension": "entropy_weighting",
                "values": list(TARGET_WEIGHTINGS),
                "source": "Aligned active entropy was saved.",
            },
        ],
        "not_recoverable": [
            {
                "dimension": "last_n_layers",
                "requested_values": [1, 2, 4],
                "reason": "Only the last-four-layer aggregate was saved.",
                "requires_new_capture": True,
            },
            {
                "dimension": "selected_key_renormalization",
                "requested_values": [False, True],
                "reason": "Only selected-key-renormalized matrices were saved.",
                "requires_new_capture": True,
            },
            {
                "dimension": "prompt_or_committed_key_inclusion",
                "requested_values": [False, True],
                "reason": "Only active masked positions were saved as keys.",
                "requires_new_capture": True,
            },
        ],
    }


def run_proxy_variant_analysis(
    *,
    states_path: Path,
    source_configuration_path: Path,
    output_directory: Path,
    expected_state_count: int,
    expected_manifest_sha256: str,
    dependency_fields: Sequence[str],
    directions: Sequence[str],
    confidence_exponents: Sequence[float],
    target_weightings: Sequence[str],
    random_seed: int,
    recall_ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Validate inputs and write the complete frozen-state P2.7 package."""
    recall_ks = tuple(recall_ks)
    if not recall_ks or tuple(sorted(set(recall_ks))) != recall_ks:
        raise ValueError("recall_ks must be unique and strictly increasing.")
    if 4 not in recall_ks:
        raise ValueError("P2.7 paired comparisons require Recall@4 and regret@4.")
    catalog = build_variant_catalog(
        dependency_fields=dependency_fields,
        directions=directions,
        confidence_exponents=confidence_exponents,
        target_weightings=target_weightings,
    )
    variant_names = tuple(str(row["proposal"]) for row in catalog)

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
        analyze_proxy_state_variants(
            record,
            variant_catalog=catalog,
            random_seed=random_seed,
        )
        for record in records
    ]
    if any(max(recall_ks) > len(state.positions) for state in states):
        raise ValueError("A requested Recall@K exceeds a state's position count.")

    recoverability = _recoverability_accounting()
    analysis_configuration = {
        "schema_version": VARIANT_ANALYSIS_SCHEMA_VERSION,
        "base_analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_label": "P2.7 exploratory frozen-state variants",
        "states_path": str(states_path.resolve()),
        "states_sha256": _sha256_file(states_path),
        "source_configuration_path": str(source_configuration_path.resolve()),
        "source_configuration_fingerprint": source_fingerprint,
        "expected_state_count": expected_state_count,
        "expected_manifest_sha256": expected_manifest_sha256,
        "dependency_direction_convention": DEPENDENCY_DIRECTION,
        "dependency_fields": list(dependency_fields),
        "directions": list(directions),
        "confidence_exponents": [float(value) for value in confidence_exponents],
        "target_weightings": list(target_weightings),
        "variant_count": len(catalog),
        "baselines": list(BASELINE_NAMES),
        "paired_baselines": ["random", "confidence"],
        "paired_metrics": list(PAIRED_METRICS),
        "random_seed": random_seed,
        "recall_ks": list(recall_ks),
        "oracle_fields": ORACLE_FIELDS,
        "oracle_interpretation": (
            "One-step mechanistic entropy/risk targets, not final-answer reward."
        ),
        "bootstrap": {
            "unit": "example",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
            "state_aggregation": "mean_within_example",
        },
        "regret_definition": (
            "max target over all positions minus max target in proposal top-k"
        ),
        "paired_advantage_definition": (
            "proposal minus baseline for Spearman/Recall; baseline minus proposal "
            "for regret; positive always favors proposal"
        ),
        "exploratory_warning": (
            "The grid is evaluated on the same examples as P2.6; any selected "
            "variant requires confirmation on fresh disjoint states."
        ),
        "recoverability": recoverability,
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
            raise ValueError(
                "Output directory contains a different analysis configuration."
            )
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
    by_task = aggregate_metrics(
        metric_records,
        group_fields=("task",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_mask_ratio = aggregate_metrics(
        metric_records,
        group_fields=("mask_ratio",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_task_mask_ratio = aggregate_metrics(
        metric_records,
        group_fields=("task", "mask_ratio"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired_overall = paired_variant_comparisons(
        metric_records,
        variant_names=variant_names,
        baseline_names=("random", "confidence"),
        group_fields=(),
        metrics=PAIRED_METRICS,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired_by_mask_ratio = paired_variant_comparisons(
        metric_records,
        variant_names=variant_names,
        baseline_names=("random", "confidence"),
        group_fields=("mask_ratio",),
        metrics=PAIRED_METRICS,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    _write_jsonl(output_directory / "position_scores.jsonl", position_records)
    _write_jsonl(output_directory / "per_state_metrics.jsonl", state_records)
    _write_csv(output_directory / "variant_catalog.csv", catalog)
    _write_csv(output_directory / "overall.csv", overall)
    _write_csv(output_directory / "by_task.csv", by_task)
    _write_csv(output_directory / "by_mask_ratio.csv", by_mask_ratio)
    _write_csv(output_directory / "by_task_mask_ratio.csv", by_task_mask_ratio)
    _write_csv(output_directory / "paired_overall.csv", paired_overall)
    _write_csv(
        output_directory / "paired_by_mask_ratio.csv", paired_by_mask_ratio
    )
    _write_json(output_directory / "recoverability.json", recoverability)

    output_names = [
        "analysis_configuration.json",
        "position_scores.jsonl",
        "per_state_metrics.jsonl",
        "variant_catalog.csv",
        "overall.csv",
        "by_task.csv",
        "by_mask_ratio.csv",
        "by_task_mask_ratio.csv",
        "paired_overall.csv",
        "paired_by_mask_ratio.csv",
        "recoverability.json",
    ]
    summary: dict[str, object] = {
        "status": "completed",
        "analysis_configuration_fingerprint": analysis_fingerprint,
        "source_configuration_fingerprint": source_fingerprint,
        "states_sha256": analysis_configuration["states_sha256"],
        "state_count": len(states),
        "example_count": len({state.example_id for state in states}),
        "position_count": len(position_records),
        "variant_count": len(catalog),
        "baseline_count": len(BASELINE_NAMES),
        "proposal_count": len(states[0].proposal_scores),
        "task_counts": {
            task: len([state for state in states if state.task == task])
            for task in sorted({state.task for state in states})
        },
        "mask_ratio_counts": {
            str(mask_ratio): len(
                [state for state in states if state.mask_ratio == mask_ratio]
            )
            for mask_ratio in sorted(
                {state.mask_ratio for state in states}, reverse=True
            )
        },
        "oracles": list(ORACLE_FIELDS),
        "recall_ks": list(recall_ks),
        "bootstrap_samples": bootstrap_samples,
        "exploratory": True,
        "output_files": sorted(
            str((output_directory / name).resolve()) for name in output_names
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    return summary
