"""Utils for the TSE pipeline.

Run focused TSE tests with:
    pytest scripts/tests/test_tse_components.py -v
"""

from contextlib import contextmanager
from functools import wraps
import os
import time

import torch


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
