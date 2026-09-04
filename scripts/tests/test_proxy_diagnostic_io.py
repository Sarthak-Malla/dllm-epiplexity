"""
Test configuration-safe, resumable proxy diagnostic JSONL persistence.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_proxy_diagnostic_io.py" -v
"""

import json

import pytest

from dllm.core.samplers.diagnostic_io import (
    ProxyConfigurationMismatchError,
    ProxyDiagnosticStore,
    build_proxy_failure_id,
    build_proxy_state_id,
    configuration_fingerprint,
)


def _configuration() -> dict[str, object]:
    return {
        "checkpoint": "/absolute/checkpoint/revision",
        "dataset": "gsm8k",
        "response_length": 8,
        "mask_ratios": [1.0, 0.75, 0.5, 0.25],
        "last_n_layers": 4,
        "seed": 42,
    }


def test_fingerprints_and_record_ids_are_stable_and_configuration_bound():
    configuration = _configuration()
    reordered = dict(reversed(tuple(configuration.items())))
    fingerprint = configuration_fingerprint(configuration)

    assert fingerprint == configuration_fingerprint(reordered)
    first = build_proxy_state_id(
        config_fingerprint=fingerprint,
        example_id="gsm8k-test-0",
        prompt="question",
        state_index=0,
        target_mask_ratio=1.0,
    )
    repeated = build_proxy_state_id(
        config_fingerprint=fingerprint,
        example_id="gsm8k-test-0",
        prompt="question",
        state_index=0,
        target_mask_ratio=1.0,
    )
    next_stage = build_proxy_state_id(
        config_fingerprint=fingerprint,
        example_id="gsm8k-test-0",
        prompt="question",
        state_index=1,
        target_mask_ratio=0.75,
    )

    assert first == repeated
    assert first.startswith("proxy-state-")
    assert first != next_stage


def test_store_flushes_states_and_resumes_without_duplicates(tmp_path):
    configuration = _configuration()
    store = ProxyDiagnosticStore(
        tmp_path,
        configuration=configuration,
        environment={"gpu_name": "test-gpu"},
    )
    state_id = build_proxy_state_id(
        config_fingerprint=store.configuration_fingerprint,
        example_id="gsm8k-test-0",
        prompt="question",
        state_index=0,
        target_mask_ratio=1.0,
    )

    assert store.append_state(
        state_id,
        {
            "example_id": "gsm8k-test-0",
            "state_index": 0,
            "masked_count": 8,
        },
    )
    lines_after_append = store.states_path.read_text().splitlines()
    assert len(lines_after_append) == 1
    persisted = json.loads(lines_after_append[0])
    assert persisted["state_id"] == state_id
    assert persisted["status"] == "completed"
    assert persisted["configuration_fingerprint"] == (
        store.configuration_fingerprint
    )
    assert not store.append_state(state_id, {"unexpected": "replacement"})
    assert len(store.states_path.read_text().splitlines()) == 1

    resumed = ProxyDiagnosticStore(
        tmp_path,
        configuration=configuration,
        environment={"gpu_name": "different-resume-gpu"},
    )
    assert resumed.completed_state_ids == frozenset({state_id})
    assert not resumed.append_state(state_id, {"unexpected": "replacement"})
    metadata = json.loads(resumed.metadata_path.read_text())
    assert metadata["configuration"] == configuration
    assert metadata["creation_environment"] == {"gpu_name": "test-gpu"}


def test_store_rejects_configuration_mixing(tmp_path):
    ProxyDiagnosticStore(tmp_path, configuration=_configuration())
    changed = _configuration()
    changed["last_n_layers"] = 2

    with pytest.raises(
        ProxyConfigurationMismatchError,
        match="different configuration",
    ):
        ProxyDiagnosticStore(tmp_path, configuration=changed)


def test_failure_records_include_oom_sizes_and_resume_without_duplicates(tmp_path):
    store = ProxyDiagnosticStore(tmp_path, configuration=_configuration())
    failure_id = build_proxy_failure_id(
        config_fingerprint=store.configuration_fingerprint,
        example_id="gsm8k-test-0",
        prompt="question",
        failure_kind="cuda_out_of_memory",
    )
    failure = {
        "failure_kind": "cuda_out_of_memory",
        "sequence_length": 38,
        "response_length": 8,
        "evaluation_pool_size": 4,
        "top_confidence_pool_size": 2,
        "random_pool_size": 2,
    }

    assert store.append_failure(failure_id, failure)
    assert not store.append_failure(failure_id, failure)
    persisted = json.loads(store.failures_path.read_text().strip())
    assert persisted["failure_id"] == failure_id
    assert persisted["status"] == "failed"
    assert persisted["failure_kind"] == "cuda_out_of_memory"
    assert persisted["sequence_length"] == 38
    assert persisted["evaluation_pool_size"] == 4

    resumed = ProxyDiagnosticStore(tmp_path, configuration=_configuration())
    assert resumed.failure_ids == frozenset({failure_id})


def test_resume_rejects_corrupt_or_mixed_jsonl_records(tmp_path):
    store = ProxyDiagnosticStore(tmp_path, configuration=_configuration())
    store.states_path.write_text(
        json.dumps(
            {
                "state_id": "proxy-state-invalid",
                "status": "completed",
                "configuration_fingerprint": "wrong",
            }
        )
        + "\n"
    )

    with pytest.raises(ProxyConfigurationMismatchError, match="fingerprint"):
        ProxyDiagnosticStore(tmp_path, configuration=_configuration())
