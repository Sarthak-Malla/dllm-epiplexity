"""Run: pytest scripts/tests/test_native_span_rescore.py -q."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tse" / "ctca"))
from native_span_rescore import conditional_span_log_probability, score_native


class TinyTokenizer:
    def get_vocab(self):
        return {"[MASK]": 0, "a": 1, "b": 2}


class ContextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 3))
        logits[:, 0, 1] = 5
        logits[:, 1, 2] = torch.where(input_ids[:, 0] == 1, 5.0, -5.0)
        return SimpleNamespace(logits=logits)


def test_conditional_score_uses_filled_prefix_and_changes_span_likelihood():
    model = ContextModel()
    tokenizer = TinyTokenizer()
    canvas = torch.tensor([0, 0])
    parallel, _ = score_native(model, tokenizer, canvas, [0, 1], "a")
    conditional, _ = conditional_span_log_probability(
        model, tokenizer, canvas, 0, [1, 2], "a", parallel[0, 1])
    parallel_sum = float(parallel[0, 1] + parallel[1, 2])
    assert conditional > parallel_sum + 5
    assert torch.equal(canvas, torch.tensor([0, 0]))
