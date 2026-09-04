"""Test frozen, balanced proxy-diagnostic manifests.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_manifest.py -v
"""

from dataclasses import replace
import json

import pytest

from dllm.core.samplers.proxy_manifest import (
    ProxyManifestExample,
    format_gsm8k_cot_prompt,
    format_humaneval_instruct_prompt,
    interleave_balanced_examples,
    manifest_fingerprint,
    read_manifest,
    select_dataset_indices,
    write_or_validate_manifest,
)


def _example(
    *,
    dataset_label: str,
    dataset_index: int,
) -> ProxyManifestExample:
    """Build one complete constructed manifest entry."""
    return ProxyManifestExample(
        dataset_label=dataset_label,
        dataset_path=f"organization/{dataset_label}",
        dataset_config="default",
        dataset_revision="revision-hash",
        dataset_fingerprint="dataset-fingerprint",
        split="test",
        dataset_index=dataset_index,
        example_id=f"{dataset_label}-test-{dataset_index:05d}",
        source_id=str(dataset_index),
        prompt=f"prompt {dataset_label} {dataset_index}",
        prompt_format=f"{dataset_label}_v1",
        num_fewshot=0,
        selection_seed=42,
    )


def test_prompt_formatters_freeze_fewshot_and_instruction_settings():
    gsm8k = format_gsm8k_cot_prompt("Target question?", num_fewshot=2)

    assert gsm8k.count("Q: ") == 3
    assert gsm8k.count("\n\nA:") == 3
    assert gsm8k.endswith("Q: Target question?\n\nA:")
    assert format_humaneval_instruct_prompt("def add(a, b):") == (
        "Complete the following python code:\ndef add(a, b):"
    )
    with pytest.raises(ValueError, match="num_fewshot"):
        format_gsm8k_cot_prompt("Target question?", num_fewshot=6)


def test_dataset_sampling_is_sorted_deterministic_and_namespaced():
    first = select_dataset_indices(100, 32, seed=42, namespace="gsm8k:test")
    repeated = select_dataset_indices(100, 32, seed=42, namespace="gsm8k:test")
    other_task = select_dataset_indices(
        100,
        32,
        seed=42,
        namespace="humaneval:test",
    )

    assert first == repeated
    assert first == tuple(sorted(first))
    assert len(first) == len(set(first)) == 32
    assert first != other_task
    with pytest.raises(ValueError, match="cannot exceed"):
        select_dataset_indices(2, 3, seed=42, namespace="invalid")


def test_balanced_interleave_preserves_each_task_order():
    gsm8k = tuple(_example(dataset_label="gsm8k", dataset_index=i) for i in range(2))
    humaneval = tuple(
        _example(dataset_label="humaneval", dataset_index=i) for i in range(2)
    )

    interleaved = interleave_balanced_examples(gsm8k, humaneval)

    assert [example.dataset_label for example in interleaved] == [
        "gsm8k",
        "humaneval",
        "gsm8k",
        "humaneval",
    ]
    assert [example.dataset_index for example in interleaved] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="equal lengths"):
        interleave_balanced_examples(gsm8k, humaneval[:1])


def test_manifest_round_trip_fingerprint_and_mismatch_protection(tmp_path):
    examples = (
        _example(dataset_label="gsm8k", dataset_index=1),
        _example(dataset_label="humaneval", dataset_index=2),
    )
    path = tmp_path / "manifest.jsonl"

    assert write_or_validate_manifest(path, examples)
    assert not write_or_validate_manifest(path, examples)
    assert read_manifest(path) == examples
    assert manifest_fingerprint(read_manifest(path)) == manifest_fingerprint(examples)
    assert manifest_fingerprint(tuple(reversed(examples))) != manifest_fingerprint(
        examples
    )

    changed = (replace(examples[0], prompt="changed prompt"), examples[1])
    with pytest.raises(ValueError, match="differs"):
        write_or_validate_manifest(path, changed)


def test_manifest_reader_rejects_unknown_fields_and_duplicate_rows(tmp_path):
    first = _example(dataset_label="gsm8k", dataset_index=1)
    duplicate_row = replace(first, example_id="different-id")
    with pytest.raises(ValueError, match="dataset rows"):
        manifest_fingerprint((first, duplicate_row))

    malformed = first.to_dict()
    malformed["unexpected"] = True
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(malformed) + "\n")
    with pytest.raises(ValueError, match="extra"):
        read_manifest(path)
