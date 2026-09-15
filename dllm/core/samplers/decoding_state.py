"""Replayable decoding state and frozen decisions for training-free experiments.

Run the focused tests on a compute node after sourcing ~/.zshrc and activating
the dllm environment:
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_replay.py
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import random
from typing import Any

import numpy as np
import torch

from dllm.core.samplers.parallel_candidates import CommittedAnchorState


@dataclass
class RNGState:
    """Python, NumPy, and initialized PyTorch generators without CUDA startup."""

    python: tuple
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]

    @classmethod
    def capture(cls) -> RNGState:
        numpy_state = np.random.get_state()
        return cls(
            python=random.getstate(),
            numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=(
                [value.clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else []
            ),
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state((
            self.numpy[0], np.asarray(self.numpy[1], dtype=np.uint32),
            *self.numpy[2:],
        ))
        torch.set_rng_state(self.torch_cpu.cpu())
        if self.torch_cuda:
            if len(self.torch_cuda) != torch.cuda.device_count():
                raise ValueError("Replay requires the same visible CUDA generator count.")
            torch.cuda.set_rng_state_all([value.cpu() for value in self.torch_cuda])


@contextmanager
def isolated_rng(rng: RNGState | None = None):
    """Run a branch without consuming its caller's random generators."""
    previous = RNGState.capture()
    try:
        if rng is not None:
            rng.restore()
        yield
    finally:
        previous.restore()


def _copy_tree(value: Any, device: torch.device | str | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device) if device is not None else value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item, device) for item in value)
    return deepcopy(value)


@dataclass
class DecodeState:
    """All non-model state needed to continue at a decision boundary."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    unmasked_index: torch.Tensor
    prompt_lens: list[int]
    max_new_tokens: int
    num_blocks: int
    steps_per_block: int
    config: Any
    rng: RNGState
    block_index: int = 0
    step_index: int = 0
    global_step_index: int = 0
    block_span_mask: torch.Tensor | None = None
    num_transfer_tokens: torch.Tensor | None = None
    effective_steps: int = 0
    anchor_state: CommittedAnchorState | None = None
    histories: list[torch.Tensor] | None = None
    selected_candidates: list[list[str]] = field(default_factory=list)
    diagnostics: list[list[dict[str, object]]] | None = None

    @property
    def done(self) -> bool:
        return self.block_index >= self.num_blocks

    def clone(self, *, include_history: bool = True) -> DecodeState:
        # Avoid copying an entire decoding trajectory for every prepared step.
        copied = {
            name: deepcopy(value)
            for name, value in vars(self).items()
            if name not in {"histories", "diagnostics", "selected_candidates"}
        }
        copied["histories"] = (
            _copy_tree(self.histories) if include_history else
            ([self.input_ids.clone()] if self.histories is not None else None)
        )
        copied["diagnostics"] = (
            deepcopy(self.diagnostics) if include_history else
            ([[] for _ in self.prompt_lens] if self.diagnostics is not None else None)
        )
        copied["selected_candidates"] = (
            deepcopy(self.selected_candidates) if include_history else
            [[] for _ in self.prompt_lens]
        )
        return type(self)(**copied)

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU tensor/container payload suitable for torch.save."""
        payload = {
            name: _copy_tree(value, "cpu")
            for name, value in vars(self).items()
            if name not in {"config", "rng", "anchor_state"}
        }
        payload["config"] = asdict(self.config)
        payload["rng"] = _copy_tree(asdict(self.rng), "cpu")
        payload["anchor_state"] = (
            _copy_tree(asdict(self.anchor_state), "cpu")
            if self.anchor_state is not None else None
        )
        return {"version": 1, "state": payload}

    @classmethod
    def from_state_dict(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> DecodeState:
        from dllm.core.samplers.entropy_drop import EntropyDropSamplerConfig

        if payload.get("version") != 1:
            raise ValueError("Unsupported decoding snapshot version.")
        raw = payload["state"]
        values = {
            name: _copy_tree(value, device)
            for name, value in raw.items()
            if name not in {"config", "rng", "anchor_state"}
        }
        values["config"] = EntropyDropSamplerConfig(**raw["config"])
        values["rng"] = RNGState(**_copy_tree(raw["rng"], "cpu"))
        values["anchor_state"] = (
            CommittedAnchorState(**_copy_tree(raw["anchor_state"], device))
            if raw["anchor_state"] is not None else None
        )
        return cls(**values)


@dataclass
class PreparedStep:
    """One shared pre-reveal prediction and candidate pool; no committed action."""

    state: DecodeState
    config: Any
    base_forward: Any
    x0: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    top2_margin: torch.Tensor | None
    probabilities: torch.Tensor
    candidates: Any
    dependency: Any
    active_mask: torch.Tensor
    masked_active_mask: torch.Tensor
    requested_k: torch.Tensor
    step_seed: int
    reconstruction_seconds: float
    proposal_seconds: float
    rng_after: RNGState
    selections: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSelection:
    selector: str
    best_mask: torch.Tensor
    best_index: torch.Tensor
    best_names: tuple[str | None, ...]
    scores: torch.Tensor
    detail: Any


@dataclass
class CommitValues:
    token_ids: torch.Tensor
    confidence: torch.Tensor
    first_positions: torch.Tensor
    companion_mask: torch.Tensor
    flipped_mask: torch.Tensor
    original_value_probability: torch.Tensor
    refreshed_probabilities: torch.Tensor
    # Preserve logit-based decisions: low-precision probabilities can introduce
    # ties that did not exist in the logits used by the deployed decoder.
    refreshed_token_ids: torch.Tensor | None = None
