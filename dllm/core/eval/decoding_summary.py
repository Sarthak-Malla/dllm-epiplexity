"""Summarize saved decoding steps and deduplicate distributed evaluation padding.

Run the focused checks on a compute node:
    source /home/sarthak.malla/.zshrc
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:15:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_summary.py -q
"""

from collections import Counter


def summarize_decoding_steps(steps):
    """Count forwards over the full generated canvas, before answer trimming."""
    if not steps:
        return None
    selected = [
        step["selected_candidate"] for step in steps
        if step.get("selected_candidate") is not None
    ]
    sizes = [item["action_size"] for item in selected]
    base_calls = sum(step["captured_base_forward_count"] for step in steps)
    lookahead_calls = sum(step["lookahead_model_calls"] for step in steps)
    return {
        "model_calls": base_calls + lookahead_calls,
        "base_model_calls": base_calls,
        "lookahead_model_calls": lookahead_calls,
        "recorded_steps": len(steps),
        "decoding_actions": len(selected),
        "committed_tokens": sum(sizes),
        "mean_action_size": sum(sizes) / len(sizes) if sizes else None,
        "action_size_histogram": dict(Counter(str(size) for size in sizes)),
        "singleton_actions": sizes.count(1),
        "singleton_refill_selections": sum(
            item.get("fallback_source") == "adaptive_singleton_refill"
            for item in selected
        ),
        "candidate_zero_selections": sum(item["index"] == 0 for item in selected),
        "selected_candidate_histogram": dict(Counter(str(item["index"]) for item in selected)),
        "mean_selected_confidence": (
            sum(item["mean_confidence"] for item in selected) / len(selected)
            if selected and all("mean_confidence" in item for item in selected) else None
        ),
        "mean_selected_entropy_sum": (
            sum(item["entropy_sum"] for item in selected) / len(selected)
            if selected and all("entropy_sum" in item for item in selected) else None
        ),
    }


def aggregate_decoding_summaries(records):
    """Report calls per unique request separately from repeated/padded work.

    Calls count each example's participation in model forwards. With batch size
    one and CFG off (the diagnostic N4 launcher), these equal model invocations.
    Task, document, request index and prompt hash identify duplicated requests.
    """
    measured = [record for record in records if record.get("summary") is not None]
    unique = {}
    for record in measured:
        if record.get("task_name") is not None and record.get("doc_id") is not None:
            key = (
                record["task_name"], record["doc_id"], record.get("request_index"),
                record.get("prompt_sha256"),
            )
        else:
            key = (record.get("rank"), record["example_index"])
        unique.setdefault(key, record)
    summaries = [record["summary"] for record in unique.values()]
    count = len(summaries)
    histogram = Counter()
    for summary in summaries:
        histogram.update(summary["action_size_histogram"])
    return {
        "counting_scope": "full generation before stop-string trimming",
        "call_semantics": "per-example forward participation; equals invocations at batch_size=1",
        "generated_records": len(records),
        "measured_records": len(measured),
        "unique_measured_examples": count,
        "duplicate_measured_records": len(measured) - count,
        "model_calls_including_repeated_examples": sum(
            record["summary"]["model_calls"] for record in measured
        ),
        "model_calls_unique_examples": sum(item["model_calls"] for item in summaries),
        "mean_model_calls_per_example": (
            sum(item["model_calls"] for item in summaries) / count if count else None
        ),
        "mean_base_model_calls_per_example": (
            sum(item["base_model_calls"] for item in summaries) / count if count else None
        ),
        "mean_lookahead_model_calls_per_example": (
            sum(item["lookahead_model_calls"] for item in summaries) / count if count else None
        ),
        "singleton_actions": sum(item["singleton_actions"] for item in summaries),
        "singleton_refill_selections": sum(
            item["singleton_refill_selections"] for item in summaries
        ),
        "action_size_histogram": dict(histogram),
    }
