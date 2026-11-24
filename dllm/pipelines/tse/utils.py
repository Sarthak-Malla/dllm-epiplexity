"""Utils for the TSE pipeline.

Run focused TSE tests with:
    pytest scripts/tests/test_tse_components.py -v
"""

from contextlib import contextmanager
from functools import wraps
import os
import time

import torch


def initialize_canvas(
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    mask_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad prompts followed by the same number of generation slots."""
    prompt_lens = [prompt.numel() for prompt in prompts]
    canvas = prompts[0].new_full((len(prompts), max(prompt_lens) + max_new_tokens), pad_id)
    attention_mask = torch.zeros_like(canvas)
    for index, (prompt, length) in enumerate(zip(prompts, prompt_lens)):
        end = length + max_new_tokens
        canvas[index, :length] = prompt
        canvas[index, length:end] = mask_id
        attention_mask[index, :end] = 1
    return canvas, attention_mask, prompt_lens


def forward_token_logits(
    model: torch.nn.Module, canvas: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Return logits indexed by the token position they predict."""
    logits = model(canvas, attention_mask=attention_mask).logits
    if getattr(getattr(model, "config", None), "model_type", "") in {"Dream", "dream"}:
        # Dream's native sampler uses the preceding position's logits.
        logits = torch.cat((logits[:, :1], logits[:, :-1]), dim=1)
    return logits


def _timers_enabled() -> bool:
    return os.environ.get("TSE_TIMERS", "").casefold() in {"1", "true", "yes", "on"}


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _timer_context(label: str):
    if not _timers_enabled():
        yield
        return
    _synchronize_cuda()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        _synchronize_cuda()
        elapsed = time.perf_counter() - start_time
        print(f"[timer] {label}: {elapsed:.4f}s", flush=True)


def timer(func=None, *, label: str | None = None):
    """Measure elapsed wall time as a decorator or labeled context manager."""
    if isinstance(func, str):
        return _timer_context(func)
    if func is None:
        if label is not None:
            return _timer_context(label)

        def decorator(wrapped):
            return timer(wrapped)

        return decorator

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _timer_context(label or func.__name__):
            return func(*args, **kwargs)

    return wrapper
