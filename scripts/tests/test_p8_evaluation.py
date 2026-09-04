"""Test Phase-8 manifests and compact diagnostic retention.

Run from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_evaluation.py -v
"""

import importlib.util
from pathlib import Path

import pytest

from dllm.core.eval.diagnostic_retention import retain_diagnostics


PLAN = Path(
    "/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/"
    "dependency_guided/p8_evaluation_plan.json"
)
MANIFEST_TOOL = Path(
    "/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/"
    "dependency_guided/prepare_p8_manifest.py"
)
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "prepare_p8_manifest",
    MANIFEST_TOOL,
)
if MANIFEST_SPEC is None or MANIFEST_SPEC.loader is None:
    raise RuntimeError(f"Could not load manifest module from {MANIFEST_TOOL}.")
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)
build_manifest = MANIFEST_MODULE.build_manifest
load_plan = MANIFEST_MODULE.load_plan


def test_smoke_and_primary_ranges_are_frozen_and_disjoint_from_tuning():
    plan = load_plan(PLAN)

    smoke, smoke_metadata = build_manifest(
        plan,
        stage="smoke",
        task="gsm8k_cot",
    )
    first_primary, primary_metadata = build_manifest(
        plan,
        stage="primary",
        task="gsm8k_cot",
        shard_index=0,
    )

    assert smoke == {"gsm8k_cot": [80]}
    assert smoke_metadata["shard_count"] == 1
    assert first_primary["gsm8k_cot"] == list(range(80, 96))
    assert primary_metadata["shard_count"] == 78
    assert min(first_primary["gsm8k_cot"]) >= 80


def test_last_primary_shards_clip_to_dataset_boundary():
    plan = load_plan(PLAN)

    gsm, gsm_metadata = build_manifest(
        plan,
        stage="primary",
        task="gsm8k_cot",
        shard_index=77,
    )
    humaneval, humaneval_metadata = build_manifest(
        plan,
        stage="primary",
        task="humaneval_instruct_llada",
        shard_index=5,
    )

    assert gsm["gsm8k_cot"] == list(range(1312, 1319))
    assert gsm_metadata["sample_count"] == 7
    assert humaneval["humaneval_instruct_llada"] == list(range(160, 164))
    assert humaneval_metadata["sample_count"] == 4


def test_invalid_shard_is_rejected():
    plan = load_plan(PLAN)

    with pytest.raises(ValueError, match="shard_index"):
        build_manifest(
            plan,
            stage="primary",
            task="hendrycks_math500",
            shard_index=32,
        )


def test_compact_diagnostics_remove_candidate_pool_but_keep_cost_fields():
    diagnostics = [
        [
            {
                "lookahead_model_calls": 4,
                "captured_base_forward_count": 1,
                "candidates": [{"name": "candidate_0", "positions": [2, 4]}],
                "selected_candidate": {
                    "name": "candidate_0",
                    "positions": [2, 4],
                    "valid": True,
                },
            }
        ]
    ]

    compact = retain_diagnostics(diagnostics, "compact")

    assert "candidates" not in compact[0][0]
    assert "positions" not in compact[0][0]["selected_candidate"]
    assert compact[0][0]["lookahead_model_calls"] == 4
    assert compact[0][0]["captured_base_forward_count"] == 1
    assert retain_diagnostics(diagnostics, "full") is diagnostics
    assert retain_diagnostics(diagnostics, "none") == [[]]


def test_unknown_diagnostic_retention_is_rejected():
    with pytest.raises(ValueError, match="diagnostic_retention"):
        retain_diagnostics([], "summary")
