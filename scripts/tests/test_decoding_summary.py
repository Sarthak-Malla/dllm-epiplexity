"""Test call accounting and distributed-padding deduplication.

Run on a compute node:
    source /home/sarthak.malla/.zshrc
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_summary.py -q
"""

import pytest

from dllm.core.eval.decoding_summary import (
    aggregate_decoding_summaries,
    summarize_decoding_steps,
)


def _step(size, *, index=0, refill=False, lookahead=0):
    return {
        "captured_base_forward_count": 1,
        "lookahead_model_calls": lookahead,
        "selected_candidate": {
            "index": index,
            "action_size": size,
            "fallback_source": "adaptive_singleton_refill" if refill else None,
            "mean_confidence": .9,
            "entropy_sum": .2 * size,
        },
    }


def test_counts_calls_and_singleton_refill_separately():
    summary = summarize_decoding_steps([_step(3), _step(1, index=2, refill=True)])
    assert summary["model_calls"] == summary["base_model_calls"] == 2
    assert summary["lookahead_model_calls"] == 0
    assert summary["committed_tokens"] == 4
    assert summary["mean_action_size"] == 2
    assert summary["action_size_histogram"] == {"3": 1, "1": 1}
    assert summary["singleton_actions"] == summary["singleton_refill_selections"] == 1
    assert summary["candidate_zero_selections"] == 1
    assert summary["mean_selected_entropy_sum"] == pytest.approx(.4)


def test_unknown_calls_are_not_reported_as_zero():
    assert summarize_decoding_steps([]) is None
    summary = aggregate_decoding_summaries([{"summary": None}])
    assert summary["generated_records"] == 1
    assert summary["measured_records"] == 0
    assert summary["mean_model_calls_per_example"] is None


def test_lookahead_calls_are_counted_as_well_as_base_calls():
    assert summarize_decoding_steps([_step(4, lookahead=4)])["model_calls"] == 5


def test_distributed_padding_is_excluded_from_per_example_mean():
    first = {
        "rank": 0, "example_index": 0, "task_name": "gsm8k_cot",
        "doc_id": 0, "request_index": 0, "prompt_sha256": "first",
        "summary": summarize_decoding_steps([_step(4)]),
    }
    second = {
        **first, "rank": 1, "doc_id": 1, "prompt_sha256": "second",
        "summary": summarize_decoding_steps([_step(3), _step(1)]),
    }
    padding = {**second, "example_index": 1}
    result = aggregate_decoding_summaries([first, second, padding])
    assert result["unique_measured_examples"] == 2
    assert result["duplicate_measured_records"] == 1
    assert result["model_calls_including_repeated_examples"] == 5
    assert result["model_calls_unique_examples"] == 3
    assert result["mean_model_calls_per_example"] == 1.5
    assert result["action_size_histogram"] == {"4": 1, "3": 1, "1": 1}
