"""Deterministic in-memory or on-disk matrix caching for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch


class CTCACacheManager:
    """Store Procrustes rotations by a deterministic model-pair signature."""

    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        force_rebuild: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.force_rebuild = force_rebuild
        self._memory: dict[str, torch.Tensor] = {}

    @staticmethod
    def signature(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except TypeError as error:
            raise ValueError("cache metadata must be JSON serializable") from error
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, metadata: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"procrustes_R_{self.signature(metadata)}.pt"

    def get_or_create(
        self,
        metadata: dict,
        builder: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Return a cached CPU tensor or build and persist it."""
        key = self.signature(metadata)
        if not self.force_rebuild and key in self._memory:
            return self._memory[key]

        path = self.path_for(metadata)
        if not self.force_rebuild and path is not None and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("metadata") != metadata:
                raise ValueError(f"Malformed CTCA cache metadata in {path}")
            rotation = payload.get("rotation")
            if not isinstance(rotation, torch.Tensor) or rotation.ndim != 2:
                raise ValueError(f"Malformed CTCA rotation in {path}")
            self._memory[key] = rotation
            return rotation

        rotation = builder().detach().float().cpu()
        if rotation.ndim != 2 or not torch.isfinite(rotation).all():
            raise ValueError("CTCA cache builder must return a finite matrix")
        self._memory[key] = rotation
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"metadata": metadata, "rotation": rotation}, path)
        return rotation


def tokenizer_vocab_signature(tokenizer) -> str:
    """Hash a tokenizer vocabulary independently of dictionary ordering."""
    vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in tokenizer.get_vocab().items()
    )
    payload = json.dumps(vocabulary, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
