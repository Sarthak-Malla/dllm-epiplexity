"""Test the frozen-state dependency-proxy analysis and its artifacts.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_analysis.py -v
"""

import csv
import json

import pytest

from dllm.core.samplers.diagnostic_io import configuration_fingerprint
from dllm.core.samplers.proxy_analysis import (
    PROPOSAL_NAMES,
    aggregate_metrics,
    analyze_proxy_state,
    bootstrap_example_mean,
    descending_average_ranks,
    run_proxy_analysis,
    spearman_correlation,
    state_ranking_metrics,
)


MANIFEST_SHA256 = "a" * 64


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
    positions = list(range(10, 10 + position_count))
    confidence = [0.5, 0.25, 1.0] if position_count == 3 else [
        (index + 1) / position_count for index in range(position_count)
    ]
    entropy = [1.0, 2.0, 3.0] if position_count == 3 else [
        float(position_count - index) for index in range(position_count)
    ]
    if position_count == 3:
        dependency_matrix = [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0],
        ]
    else:
        dependency_matrix = [
            [
                0.0 if row == column else (column + 1) / position_count
                for column in range(position_count)
            ]
            for row in range(position_count)
        ]
    oracle_entropy = [float(index) for index in range(position_count)]
    oracle_risk = [float(position_count - index) for index in range(position_count)]
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
            "before_sink": dependency_matrix,
            "after_sink": dependency_matrix,
        },
        "oracle": {
            "positions": positions,
            "entropy_drop_per_heldout": oracle_entropy,
            "risk_reduction_per_heldout": oracle_risk,
        },
    }


def test_proposal_formulas_follow_query_key_direction():
    state = analyze_proxy_state(
        _state_record(),
        dependency_field="after_sink",
        random_seed=42,
        confidence_exponent=1.0,
    )

    assert tuple(state.proposal_scores) == PROPOSAL_NAMES
    assert state.proposal_scores["dependency_degree"] == pytest.approx((8, 7, 6))
    assert state.proposal_scores["confidence_x_dependency_degree"] == pytest.approx(
        (4, 1.75, 6)
    )
    assert state.proposal_scores["proposed_dependency"] == pytest.approx(
        (10.5, 4.75, 10)
    )
    assert state.proposal_scores["negative_entropy"] == (-1.0, -2.0, -3.0)
    assert state.proposal_scores["left_to_right"] == (-10.0, -11.0, -12.0)


def test_ranks_spearman_recall_and_regret_have_fixed_tie_rules():
    assert descending_average_ranks((3.0, 1.0, 1.0)) == (1.0, 2.5, 2.5)
    assert spearman_correlation((3.0, 2.0, 1.0), (1.0, 2.0, 3.0)) == pytest.approx(
        -1.0
    )
    assert spearman_correlation((1.0, 1.0, 1.0), (1.0, 2.0, 3.0)) is None

    metrics = state_ranking_metrics(
        (0.9, 0.8, 0.7, 0.6),
        (0.1, 0.4, 0.3, 1.0),
        (10, 11, 12, 13),
        recall_ks=(2,),
    )
    assert metrics["recall_at_2"] == 0.5
    assert metrics["regret_at_2"] == pytest.approx(0.6)


def test_bootstrap_clusters_state_values_by_example():
    summary = bootstrap_example_mean(
        {"example-a": [0.0, 2.0], "example-b": [4.0]},
        samples=200,
        seed=42,
    )
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["example_count"] == 2
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]

    rows = aggregate_metrics(
        [
            {
                "state_id": "a-0",
                "example_id": "a",
                "task": "gsm8k",
                "proposal": "confidence",
                "oracle": "entropy_drop",
                "metric": "spearman",
                "value": 0.0,
            },
            {
                "state_id": "a-1",
                "example_id": "a",
                "task": "gsm8k",
                "proposal": "confidence",
                "oracle": "entropy_drop",
                "metric": "spearman",
                "value": 2.0,
            },
            {
                "state_id": "b-0",
                "example_id": "b",
                "task": "gsm8k",
                "proposal": "confidence",
                "oracle": "entropy_drop",
                "metric": "spearman",
                "value": 4.0,
            },
        ],
        group_fields=("task",),
        bootstrap_samples=200,
        bootstrap_seed=42,
    )
    assert rows[0]["mean"] == pytest.approx(2.5)
    assert rows[0]["state_count"] == 3
    assert rows[0]["example_count"] == 2


def test_complete_analysis_writes_auditable_tables_and_plots(tmp_path):
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

    output_directory = tmp_path / "analysis"
    summary = run_proxy_analysis(
        states_path=states_path,
        source_configuration_path=source_configuration_path,
        output_directory=output_directory,
        expected_state_count=4,
        expected_manifest_sha256=MANIFEST_SHA256,
        dependency_field="after_sink",
        confidence_exponent=1.0,
        random_seed=42,
        recall_ks=(2, 4, 8),
        bootstrap_samples=50,
        bootstrap_seed=42,
    )

    assert summary["status"] == "completed"
    assert summary["state_count"] == 4
    assert summary["example_count"] == 2
    assert summary["position_count"] == 32
    expected_files = {
        "analysis_configuration.json",
        "position_scores.jsonl",
        "per_state_metrics.jsonl",
        "overall.csv",
        "by_mask_ratio.csv",
        "by_task.csv",
        "by_task_mask_ratio.csv",
        "all_mask_vs_later.csv",
        "scatter_entropy_drop.png",
        "rank_entropy_drop.png",
        "scatter_risk_reduction.png",
        "rank_risk_reduction.png",
        "summary.json",
    }
    assert expected_files <= {path.name for path in output_directory.iterdir()}
    assert len((output_directory / "position_scores.jsonl").read_text().splitlines()) == 32
    assert len((output_directory / "per_state_metrics.jsonl").read_text().splitlines()) == 56
    with (output_directory / "overall.csv").open(newline="") as handle:
        overall_rows = list(csv.DictReader(handle))
    assert {row["proposal"] for row in overall_rows} == set(PROPOSAL_NAMES)
    assert {row["oracle"] for row in overall_rows} == {
        "entropy_drop",
        "risk_reduction",
    }

    with pytest.raises(ValueError, match="different analysis configuration"):
        run_proxy_analysis(
            states_path=states_path,
            source_configuration_path=source_configuration_path,
            output_directory=output_directory,
            expected_state_count=4,
            expected_manifest_sha256=MANIFEST_SHA256,
            dependency_field="after_sink",
            confidence_exponent=0.5,
            random_seed=42,
            recall_ks=(2, 4, 8),
            bootstrap_samples=50,
            bootstrap_seed=42,
        )
