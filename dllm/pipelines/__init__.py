"""Import a pipeline by name, for example `from dllm.pipelines import tie`."""

from importlib import import_module

# RL requires TRL and is imported only when requested explicitly.
from . import a2d, bert, dream, editflow, fastdllm, llada, llada2, llada21

__all__ = [
    "a2d", "bert", "dream", "editflow", "fastdllm", "llada", "llada2", "llada21",
    "rl", "tie", "tse",
]


def __getattr__(name: str):
    if name in {"tie", "tse"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
