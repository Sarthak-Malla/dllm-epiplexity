"""Count actual model evaluations and synchronized model time.

Import ForwardAccounting in the experiment runner. Test on a compute node:
    python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_artifacts.py
Source the shell configuration and activate dllm before the user's srun command.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import time

import torch


_LABEL: ContextVar[str] = ContextVar("experiment_forward_label", default="unclassified")
_FIELDS = ("model_calls", "evaluated_rows", "input_tokens", "model_seconds")


class ForwardAccounting:
    """Count each outer model invocation once, including candidate/padded rows.

    Scopes replace rather than add labels, so nested scopes cannot double-count.
    The hooks do not retain model outputs, perform inference, or change RNG state.
    """

    def __init__(self, model):
        self.model = model
        self.by_label = defaultdict(lambda: {key: 0 for key in _FIELDS})
        self._pending = []
        self._handles = []

    @staticmethod
    def _sync(device):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _before(self, module, args, kwargs):
        inputs = kwargs.get("input_ids")
        if inputs is None and args:
            inputs = args[0]
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2:
            raise ValueError("Forward accounting requires batched input_ids with shape [B,T].")
        self._sync(inputs.device)
        self._pending.append((_LABEL.get(), int(inputs.shape[0]), int(inputs.shape[1]),
                              inputs.device, time.perf_counter()))

    def _after(self, module, args, kwargs, output):
        if not self._pending:
            return
        label, rows, width, device, started = self._pending.pop()
        self._sync(device)
        values = self.by_label[label]
        values["model_calls"] += 1
        values["evaluated_rows"] += rows
        values["input_tokens"] += rows * width
        values["model_seconds"] += time.perf_counter() - started

    def __enter__(self):
        if self._handles:
            raise RuntimeError("The same ForwardAccounting context cannot be entered twice.")
        self._handles = [
            self.model.register_forward_pre_hook(self._before, with_kwargs=True),
            self.model.register_forward_hook(self._after, with_kwargs=True, always_call=True),
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._pending.clear()

    @contextmanager
    def scope(self, label: str):
        if not label:
            raise ValueError("An accounting scope needs a nonempty label.")
        token = _LABEL.set(label)
        try:
            yield self
        finally:
            _LABEL.reset(token)

    def snapshot(self) -> dict:
        by_label = copy.deepcopy(dict(self.by_label))
        totals = {field: sum(record[field] for record in by_label.values()) for field in _FIELDS}
        return {**totals, "by_label": by_label}

    def delta(self, before: dict) -> dict:
        after = self.snapshot()
        labels = set(before.get("by_label", {})) | set(after["by_label"])
        differences = {
            label: {field: after["by_label"].get(label, {}).get(field, 0)
                    - before.get("by_label", {}).get(label, {}).get(field, 0)
                    for field in _FIELDS}
            for label in sorted(labels)
        }
        return {**{field: after[field] - before.get(field, 0) for field in _FIELDS},
                "by_label": differences}
