"""Compute batch-level preview accuracy during generation.

Run the focused tests with:
    pytest scripts/tests/test_running_metrics.py -v
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence


_GSM8K_NUMBER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_LOGGER = logging.getLogger(__name__)
_HUMANEVAL_RUNNING_DISABLED = False


def _last_gsm8k_number(text: str) -> str:
    matches = _GSM8K_NUMBER.findall(text)
    if not matches:
        return ""
    groups = matches[-1]
    value = next((group for group in groups if group), "")
    return value.replace(",", "").replace("$", "").removesuffix(".")


def batch_running_accuracy(
    task_name: str | None,
    docs: Sequence[dict],
    outputs: Sequence[str],
) -> float | None:
    """Return the task's batch accuracy, or ``None`` when unsupported."""
    global _HUMANEVAL_RUNNING_DISABLED

    if not task_name or len(docs) != len(outputs) or not docs:
        return None
    if task_name.startswith("gsm8k"):
        correct = 0
        for doc, output in zip(docs, outputs):
            target = str(doc["answer"]).split("####")[-1].strip()
            correct += _last_gsm8k_number(output) == _last_gsm8k_number(target)
        return correct / len(outputs)
    if task_name == "humaneval_instruct":
        if _HUMANEVAL_RUNNING_DISABLED:
            return None
        try:
            from lm_eval.tasks.humaneval.utils import (
                build_predictions_instruct,
                pass_at_k,
            )

            predictions = build_predictions_instruct(
                [[output] for output in outputs],
                list(docs),
            )
            references = [
                f"{doc['test']}\ncheck({doc['entry_point']})" for doc in docs
            ]
            result = pass_at_k(references, predictions, k=[1])
            return float(result["pass@1"])
        except Exception:
            _HUMANEVAL_RUNNING_DISABLED = True
            _LOGGER.warning(
                "Disabling HumanEval running pass@1 after code evaluation failed; "
                "the lm-eval final metric is unaffected.",
                exc_info=True,
            )
            return None
    return None
