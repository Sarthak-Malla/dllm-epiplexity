"""Test P2.7 dependency variants and paired frozen-baseline comparisons.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_variant_analysis.py -v
"""

import csv
import json

import pytest

from dllm.core.samplers.diagnostic_io import configuration_fingerprint
from dllm.core.samplers.proxy_analysis import (
    analyze_proxy_state,
    build_analysis_records,
)
from dllm.core.samplers.proxy_variants import (
    BASELINE_NAMES,
    analyze_proxy_state_variants,
    build_variant_catalog,
    dependency_variant_name,
    paired_variant_comparisons,
    run_proxy_variant_analysis,
)


MANIFEST_SHA256 = "b" * 64


def _state_record(
    *,
    state_id="state-0",
    example_id="example-0",
    task="gsm8k",
    state_index=0,
    mask_ratio=1.0,
    configuration_fingerprint_value="configuration",
    position_count=3,
):
    """Build one aligned synthetic state with different sink matrices."""
    positions = list(range(10, 10 + position_count))
    if position_count == 3:
        confidence = [0.5, 0.25, 1.0]
        entropy = [1.0, 2.0, 3.0]
        before_sink = [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0],
        ]
        after_sink = [[value / 2 for value in row] for row in before_sink]
    else:
        confidence = [(index + 1) / position_count for index in range(position_count)]
        entropy = [float(position_count - index) for index in range(position_count)]
        before_sink = [
            [
                0.0 if row == column else (column + 1) / position_count
                for column in range(position_count)
            ]
            for row in range(position_count)
        ]
        after_sink = [[value / 2 for value in row] for row in before_sink]
    return {
        "status": "completed",
        "configuration_fingerprint": configuration_fingerprint_value,
        "state_id": state_id,
        "example_id": example_id,
        "state_index": state_index,
        "target_mask_ratio": mask_ratio,
        "source": {
            "dataset_label": task,
            "manifest_sha256": MANIFEST_SHA256,
        },
        "active_positions": positions,
        "active_confidence": confidence,
        "active_entropy": entropy,
        "dependency": {
            "direction_convention": (
                "matrix[query, key]: query uses key as context"
            ),
            "positions": positions,
            "before_sink": before_sink,
            "after_sink": after_sink,
        },
        "oracle": {
            "positions": positions,
            "entropy_drop_per_heldout": [
                float(index) for index in range(position_count)
            ],
            "risk_reduction_per_heldout": [
                float(position_count - index) for index in range(position_count)
            ],
        },
    }


def test_factorial_variants_follow_direction_weight_and_confidence_formulas():
    catalog = build_variant_catalog(
        dependency_fields=("before_sink", "after_sink"),
        directions=("incoming", "outgoing", "symmetric"),
        confidence_exponents=(0.0, 1.0),
        target_weightings=("uniform", "entropy"),
    )
    assert len(catalog) == 24
    state = analyze_proxy_state_variants(
        _state_record(), variant_catalog=catalog, random_seed=42
    )
    assert tuple(state.proposal_scores)[:4] == BASELINE_NAMES

    def scores(field, direction, eta, weighting):
        name = dependency_variant_name(
            dependency_field=field,
            direction=direction,
            confidence_exponent=eta,
            target_weighting=weighting,
        )
        return state.proposal_scores[name]

    assert scores("before_sink", "incoming", 0, "uniform") == pytest.approx(
        (8, 7, 6)
    )
    assert scores("before_sink", "outgoing", 0, "uniform") == pytest.approx(
        (3, 7, 11)
    )
    assert scores("before_sink", "symmetric", 0, "uniform") == pytest.approx(
        (5.5, 7, 8.5)
    )
    assert scores("before_sink", "incoming", 1, "entropy") == pytest.approx(
        (10.5, 4.75, 10)
    )
    assert scores("after_sink", "incoming", 0, "uniform") == pytest.approx(
        (4, 3.5, 3)
    )

    p2_6 = analyze_proxy_state(
        _state_record(),
        dependency_field="after_sink",
        random_seed=42,
        confidence_exponent=1.0,
    )
    assert scores("after_sink", "incoming", 0, "uniform") == pytest.approx(
        p2_6.proposal_scores["dependency_degree"]
    )
    assert scores("after_sink", "incoming", 1, "uniform") == pytest.approx(
        p2_6.proposal_scores["confidence_x_dependency_degree"]
    )
    assert scores("after_sink", "incoming", 1, "entropy") == pytest.approx(
        p2_6.proposal_scores["proposed_dependency"]
    )


def test_paired_advantage_reverses_regret_and_clusters_examples():
    catalog = build_variant_catalog(
        dependency_fields=("before_sink",),
        directions=("incoming",),
        confidence_exponents=(0.0,),
        target_weightings=("uniform",),
    )
    states = []
    for example_index in range(2):
        for state_index in range(2):
            states.append(
                analyze_proxy_state_variants(
                    _state_record(
                        state_id=f"state-{example_index}-{state_index}",
                        example_id=f"example-{example_index}",
                        state_index=state_index,
                        position_count=4,
                    ),
                    variant_catalog=catalog,
                    random_seed=42,
                )
            )
    _, _, metric_records = build_analysis_records(states, recall_ks=(2, 4))
    variant_name = str(catalog[0]["proposal"])
    rows = paired_variant_comparisons(
        metric_records,
        variant_names=(variant_name,),
        baseline_names=("confidence",),
        group_fields=(),
        metrics=("recall_at_4", "regret_at_4"),
        bootstrap_samples=50,
        bootstrap_seed=42,
    )
    assert {row["metric"] for row in rows} == {"recall_at_4", "regret_at_4"}
    assert all(row["example_count"] == 2 for row in rows)
    assert all(row["state_count"] == 4 for row in rows)
    assert all(row["positive_favors_proposal"] is True for row in rows)


def test_complete_variant_analysis_writes_fingerprinted_tables(tmp_path):
    source_configuration = {"manifest_sha256": MANIFEST_SHA256}
    source_fingerprint = configuration_fingerprint(source_configuration)
    source_configuration_path = tmp_path / "configuration.json"
    source_configuration_path.write_text(
        json.dumps(
            {
                "configuration": source_configuration,
                "configuration_fingerprint": source_fingerprint,
            }
        )
    )
    states_path = tmp_path / "states.jsonl"
    records = []
    for example_index, task in enumerate(("gsm8k", "humaneval")):
        for state_index, mask_ratio in enumerate((1.0, 0.5)):
            records.append(
                _state_record(
                    state_id=f"state-{example_index}-{state_index}",
                    example_id=f"example-{example_index}",
                    task=task,
                    state_index=state_index,
                    mask_ratio=mask_ratio,
                    configuration_fingerprint_value=source_fingerprint,
                    position_count=8,
                )
            )
    states_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    output_directory = tmp_path / "variants"
    summary = run_proxy_variant_analysis(
        states_path=states_path,
        source_configuration_path=source_configuration_path,
        output_directory=output_directory,
        expected_state_count=4,
        expected_manifest_sha256=MANIFEST_SHA256,
        dependency_fields=("before_sink", "after_sink"),
        directions=("incoming", "outgoing", "symmetric"),
        confidence_exponents=(0.0, 1.0),
        target_weightings=("uniform", "entropy"),
        random_seed=42,
        recall_ks=(2, 4, 8),
        bootstrap_samples=20,
        bootstrap_seed=42,
    )
    assert summary["status"] == "completed"
    assert summary["variant_count"] == 24
    assert summary["proposal_count"] == 28
    assert summary["state_count"] == 4
    expected_files = {
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
        "summary.json",
    }
    assert expected_files == {path.name for path in output_directory.iterdir()}
    with (output_directory / "variant_catalog.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 24
    recoverability = json.loads(
        (output_directory / "recoverability.json").read_text()
    )
    assert {row["dimension"] for row in recoverability["not_recoverable"]} == {
        "last_n_layers",
        "selected_key_renormalization",
        "prompt_or_committed_key_inclusion",
    }

    with pytest.raises(ValueError, match="different analysis configuration"):
        run_proxy_variant_analysis(
            states_path=states_path,
            source_configuration_path=source_configuration_path,
            output_directory=output_directory,
            expected_state_count=4,
            expected_manifest_sha256=MANIFEST_SHA256,
            dependency_fields=("after_sink",),
            directions=("incoming",),
            confidence_exponents=(0.0,),
            target_weightings=("uniform",),
            random_seed=42,
            recall_ks=(2, 4, 8),
            bootstrap_samples=20,
            bootstrap_seed=42,
        )
