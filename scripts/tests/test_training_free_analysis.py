"""Test paired inference, incomplete-stage handling, and synthetic experiment reports.

After sourcing /home/sarthak.malla/.zshrc and activating dllm, run on compute:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:20:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_analysis.py
No model or GPU is required for these synthetic artifact tests.
"""

import pytest

from examples.path_selection.experiments.analyze import (
    benchmark_summaries, build_report, diagnostic_summaries, paired_summary,
    stability_summaries, mechanism_findings,
)
from examples.path_selection.experiments.artifacts import RunStore, atomic_json


def test_cluster_bootstrap_weights_problems_instead_of_companion_count():
    records = [{"doc_id": 0, "left": True, "right": False} for _ in range(30)]
    records.append({"doc_id": 1, "left": False, "right": True})
    result = paired_summary(records, repeats=100)
    assert result["difference_pp"] == 0
    assert result["problems"] == 2
    assert result["observations"] == 31
    assert result["exact_mcnemar_p"] is None
    assert "does not certify" in result["inference"]


def test_missing_stages_remain_unavailable_not_negative_evidence(tmp_path):
    RunStore(tmp_path, {"document_ids": [0, 1], "checkpoint": "synthetic"})
    report = build_report(tmp_path)
    assert report["coverage"]["diagnostics"] == 0
    assert report["paired_diagnostics"] == {}
    assert report["stability"]["flip_rate"] is None
    assert report["recorded_physical_work"] is None
    assert report["development_nomination"] is None
    assert (tmp_path / "analysis" / "report.md").is_file()


def test_frozen_state_and_continuation_identity_are_required():
    branches = {
        "entropy": {"doc_id": 0, "state_key": "first", "continuation_config_hash": "cheap", "flexible_correct": True, "strict_correct": True},
        "cheap": {"doc_id": 0, "state_key": "another", "continuation_config_hash": "cheap", "flexible_correct": False, "strict_correct": False},
    }
    diagnostics = {"state0": {"experiment": "selectors", "comparisons": [{
        "pool": "adaptive", "entropy_branch": "entropy", "cheap_branch": "cheap", "exact_agreement": False,
        "jaccard": 0.5, "entropy_size": 3, "cheap_size": 2,
    }]}}
    with pytest.raises(ValueError, match="frozen state"):
        diagnostic_summaries(diagnostics, branches)
    branches["cheap"]["state_key"] = "first"
    branches["cheap"]["continuation_config_hash"] = "another-policy"
    with pytest.raises(ValueError, match="continuation policy"):
        diagnostic_summaries(diagnostics, branches)


def test_development_nomination_never_certifies_noninferiority():
    records = {}
    for arm, evaluations in (("reference_entropy", 20), ("reference_cheap", 5)):
        for doc_id in (0, 1):
            records[f"{arm}-{doc_id}"] = {
                "arm": arm, "doc_id": doc_id, "flexible_correct": True,
                "strict_correct": False, "accounting": {"evaluated_rows": evaluations, "model_calls": evaluations},
                "wall_seconds": evaluations / 10, "actions": [{"size": 2}],
            }
    summaries, comparisons, nomination, frontier = benchmark_summaries(records, [0, 1])
    assert summaries["reference_cheap"]["complete"]
    assert nomination["arm"] == "reference_cheap"
    assert nomination["confirmation_required"]
    assert not nomination["noninferiority_established"]
    assert frontier == ["reference_cheap"]
    del records["reference_cheap-1"]
    _, _, nomination, _ = benchmark_summaries(records, [0, 1])
    assert nomination["arm"] == "reference_entropy"


def test_identical_probe_labels_are_not_duplicated_across_mass_and_precedence():
    feature = {
        "seed": 1, "companion": 2, "confidence": 0.9, "seed_confidence": 0.95,
        "entropy": 0.2, "top2_margin": 0.8, "flipped": True,
        "original_value_probability": 0.9, "refreshed_original_value_probability": 0.1,
        "probability_loss": 0.8, "total_variation": 0.8,
        "companion_to_seed": 0.4, "seed_to_companion": 0.2,
        "symmetric_conflict": 0.5, "conflict_scale": 0.6,
        "confidence_weight": 0.1, "penalty": 0.05, "selected_key_mass": 1.0,
    }
    group = {"pool": "adaptive", "candidate_index": 0, "probe_key": "probe",
             "features": [feature], "group_size": 2, "any_flip": True}
    record = {"doc_id": 0, "state_key": "state", "probes": [group]}
    result = stability_summaries({"probe": {}}, {"state": {"threshold": 85}},
                                {"precedence": record, "mass": record})
    assert result["companions"] == 1
    assert result["groups"] == 1
    assert result["flip_rate"] == 1


def test_two_worker_report_merges_documents_and_counts_concurrent_time(tmp_path):
    manifest = {"document_ids": [0, 1], "checkpoint": "synthetic", "worker_count": 2}
    for index in (0, 1):
        store = RunStore(tmp_path / "workers" / f"worker{index}", manifest)
        # Both manifests know every document; identical document metadata dedups.
        for doc_id in (0, 1):
            store.put_json("documents", f"doc{doc_id}", {"doc_id": doc_id})
        store.put_json("collection", f"doc{index}", {"doc_id": index})
        store.mark_complete("collect", {"documents": 1, "document_ids": [index]})
        store.put_json("ledger", f"cost{index}", {
            "phase": "collect", "accounting": {"model_calls": 3, "evaluated_rows": 4},
        })
        store.put_json("invocations", f"invocation{index}", {"wall_seconds": 20, "setup_seconds": 5})
    atomic_json(tmp_path / "launches" / "invocation.json", {"wall_seconds": 23, "completed": True})
    report = build_report(tmp_path)
    assert report["coverage"]["documents"] == 2
    assert report["coverage"]["collection"] == 2
    assert set(report["completed"]) == {"worker0:collect", "worker1:collect"}
    assert report["recorded_physical_work"]["evaluated_rows"] == 8
    assert report["invocation_worker_seconds"] == 40
    assert report["model_setup_worker_seconds"] == 10
    assert report["launch_elapsed_seconds"] == 23


def test_two_worker_merge_rejects_changed_configuration(tmp_path):
    RunStore(tmp_path / "workers" / "worker0", {"checkpoint": "synthetic-a"})
    RunStore(tmp_path / "workers" / "worker1", {"checkpoint": "synthetic-b"})
    with pytest.raises(ValueError, match="different configurations"):
        build_report(tmp_path)


def test_two_worker_merge_rejects_conflicting_document_metadata(tmp_path):
    for index in (0, 1):
        store = RunStore(tmp_path / "workers" / f"worker{index}", {"document_ids": [0]})
        store.put_json("documents", "doc0", {"doc_id": 0, "prompt_hash": f"hash{index}"})
    with pytest.raises(ValueError, match="Conflicting worker artifacts"):
        build_report(tmp_path)


def test_mechanism_labels_require_directional_evidence_and_do_not_claim_equivalence():
    key = "selector/adaptive/flexible_correct/entropy-minus-cheap"
    for interval, expected in (([1, 2], "supported_in_development"),
                               ([-2, -1], "unsupported_in_development"),
                               ([0, 0], "unresolved")):
        findings = mechanism_findings({key: {"ci95_pp": interval}}, {}, {"companions": 10})
        assert findings["entropy_selection_benefit"]["status"] == expected
        assert findings["unsafe_simultaneous_commitment"]["status"] == "unresolved"
        assert findings["conflict_predicts_stability_beyond_confidence"]["status"] == "unresolved"


def test_threshold_benchmarks_pair_block_and_ranking_controls():
    arms = {
        "threshold_0.90_block64_confidence": False,
        "threshold_0.90_block256_confidence": True,
        "threshold_0.90_block64_incoming": True,
        "threshold_0.95_block64_confidence": True,
    }
    records = {
        arm: {"arm": arm, "doc_id": 0, "flexible_correct": correct, "strict_correct": correct,
              "accounting": {"evaluated_rows": 10, "model_calls": 10}, "wall_seconds": 1}
        for arm, correct in arms.items()
    }
    _, comparisons, _, _ = benchmark_summaries(records, [0])
    for alternative in ("threshold_0.90_block256_confidence", "threshold_0.90_block64_incoming"):
        key = f"{alternative}-minus-threshold_0.90_block64_confidence/flexible_correct"
        assert comparisons[key]["wins"] == 1
    assert not any("minus-threshold_0.95" in key for key in comparisons)
