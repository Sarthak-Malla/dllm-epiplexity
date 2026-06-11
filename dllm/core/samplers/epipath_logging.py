"""
JSONL logging helpers for the EpiPath sampler.

Run tests from the repository root with:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_epipath_sampler.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EpiPathJSONLLogger:
    """Append JSON-serializable EpiPath records to a JSONL file."""

    def __init__(self, log_path: str | None, log_level: str = "candidate"):
        self.log_path = log_path
        self.log_level = log_level
        if self.enabled:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.log_path is not None and self.log_level != "none"

    def write(self, record: dict[str, Any]) -> None:
        """Append a record when logging is enabled."""
        if not self.enabled:
            return
        with Path(self.log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
