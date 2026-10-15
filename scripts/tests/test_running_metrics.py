"""Run with: pytest scripts/tests/test_running_metrics.py -v"""

from dllm.core.eval.running_metrics import batch_running_accuracy


def test_gsm8k_running_accuracy_matches_flexible_last_number_extraction():
    docs = [
        {"answer": "work\n#### 1,234"},
        {"answer": "work\n#### -7"},
    ]
    outputs = [
        "First 12, then the answer is $1,234.",
        "The answer is 8.",
    ]

    assert batch_running_accuracy("gsm8k_cot", docs, outputs) == 0.5


def test_running_accuracy_returns_none_for_unknown_tasks():
    assert batch_running_accuracy("unknown", [{}], ["answer"]) is None
