"""Test the balanced P2.5 collection plan without loading datasets or a model.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_dataset_collection.py -v
"""

from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest

from dllm.core.samplers.proxy_manifest import (
    ProxyManifestExample,
    manifest_fingerprint,
)


COLLECTOR_PATH = Path(
    "/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/"
    "dependency_guided/collect_proxy_diagnostic_dataset.py"
)
COLLECTOR_SPEC = importlib.util.spec_from_file_location(
    "p2_5_collect_proxy_diagnostic_dataset",
    COLLECTOR_PATH,
)
if COLLECTOR_SPEC is None or COLLECTOR_SPEC.loader is None:
    raise RuntimeError(f"Could not load collector module from {COLLECTOR_PATH}.")
COLLECTOR_MODULE = importlib.util.module_from_spec(COLLECTOR_SPEC)
COLLECTOR_SPEC.loader.exec_module(COLLECTOR_MODULE)
_run_configuration = COLLECTOR_MODULE._run_configuration
_validate_collect_manifest = COLLECTOR_MODULE._validate_collect_manifest


def _example(dataset_label: str, dataset_index: int) -> ProxyManifestExample:
    """Build one constructed entry for collection-plan validation."""
    return ProxyManifestExample(
        dataset_label=dataset_label,
        dataset_path=f"organization/{dataset_label}",
        dataset_config="default",
        dataset_revision="revision-hash",
        dataset_fingerprint=f"{dataset_label}-fingerprint",
        split="test",
        dataset_index=dataset_index,
        example_id=f"{dataset_label}-test-{dataset_index:05d}",
        source_id=str(dataset_index),
        prompt=f"prompt {dataset_label} {dataset_index}",
        prompt_format=f"{dataset_label}_v1",
        num_fewshot=0,
        selection_seed=42,
    )


def _examples() -> tuple[ProxyManifestExample, ...]:
    """Return two alternating examples per task."""
    return (
        _example("gsm8k", 1),
        _example("humaneval", 2),
        _example("gsm8k", 3),
        _example("humaneval", 4),
    )


def _args(examples) -> Namespace:
    """Build the explicit subset of collection arguments used by helpers."""
    return Namespace(
        expected_manifest_sha256=manifest_fingerprint(examples),
        expected_example_count=4,
        expected_examples_per_task=2,
        seed=42,
        response_length=8,
        mask_ratios=[1.0, 0.75, 0.5, 0.25],
        last_n_layers=4,
        top_confidence_pool_size=2,
        random_pool_size=2,
        oracle_candidate_chunk_size=2,
        sink_filter=True,
        sink_quantile=0.99,
        renormalize_selected_keys=True,
        zero_diagonal=True,
        dtype="bfloat16",
    )


def test_collection_plan_validates_balance_order_and_manifest_hash():
    examples = _examples()
    args = _args(examples)

    assert _validate_collect_manifest(args, examples) == manifest_fingerprint(
        examples
    )

    grouped = (examples[0], examples[2], examples[1], examples[3])
    grouped_args = _args(grouped)
    with pytest.raises(ValueError, match="alternate"):
        _validate_collect_manifest(grouped_args, grouped)

    args.expected_manifest_sha256 = "wrong"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_collect_manifest(args, examples)


def test_run_configuration_records_exact_planned_work():
    examples = _examples()
    args = _args(examples)
    fingerprint = manifest_fingerprint(examples)

    configuration = _run_configuration(
        args,
        checkpoint="/absolute/checkpoint/revision",
        examples=examples,
        manifest_sha256=fingerprint,
        target_mask_counts=(8, 6, 4, 2),
    )

    assert configuration["manifest_sha256"] == fingerprint
    assert configuration["manifest_example_count"] == 4
    assert configuration["planned_state_count"] == 16
    assert configuration["planned_base_forward_count"] == 16
    assert configuration["planned_batched_oracle_forward_count"] == 28
    assert configuration["planned_candidate_sequence_count"] == 56
    assert configuration["oracle_candidate_chunk_size"] == 2
    assert [source["dataset_label"] for source in configuration["datasets"]] == [
        "gsm8k",
        "humaneval",
    ]
