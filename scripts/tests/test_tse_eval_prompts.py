"""Run with: pytest scripts/tests/test_tse_eval_prompts.py -v"""

import pytest

from dllm.pipelines.tse.eval import _select_fewshot_prefix


def gsm8k_history() -> list[dict[str, str]]:
    """Return a compact two-shot prompt in lm-eval's rendered format."""
    return [
        {
            "role": "user",
            "content": (
                "Q: demo zero?\nA: zero\n\n"
                "Q: demo one?\nA: one\n\n"
                "Q: target?\nA:"
            ),
        }
    ]


def test_select_fewshot_prefix_keeps_target_and_requested_prefix():
    selected = _select_fewshot_prefix(gsm8k_history(), 1)

    assert selected[0]["content"] == "Q: demo zero?\nA: zero\n\nQ: target?\nA:"


def test_select_fewshot_prefix_supports_zero_shot_without_mutating_input():
    original = gsm8k_history()
    selected = _select_fewshot_prefix(original, 0)

    assert selected[0]["content"] == "Q: target?\nA:"
    assert "demo zero" in original[0]["content"]


def test_model_prompt_counts_are_independent_of_master_canvas_count():
    source_history = gsm8k_history()

    zero_shot_master = _select_fewshot_prefix(source_history, 0)
    two_shot_auxiliary = _select_fewshot_prefix(source_history, 2)

    assert "demo zero" not in zero_shot_master[0]["content"]
    assert "demo zero" in two_shot_auxiliary[0]["content"]
    assert "demo one" in two_shot_auxiliary[0]["content"]


def test_select_fewshot_prefix_rejects_unavailable_examples():
    with pytest.raises(ValueError, match="contains 2"):
        _select_fewshot_prefix(gsm8k_history(), 3)
