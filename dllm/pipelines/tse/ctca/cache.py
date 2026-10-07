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
    """Store deterministic CTCA tensors by model-pair signatures."""

    def __init__(
        self,
        cache_dir: str | Path | None,
        *,
        force_rebuild: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.force_rebuild = force_rebuild
        self._memory: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    @staticmethod
    def signature(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except TypeError as error:
            raise ValueError("cache metadata must be JSON serializable") from error
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, metadata: dict, artifact: str = "procrustes_R") -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{artifact}_{self.signature(metadata)}.pt"

    def get_or_create(
        self,
        metadata: dict,
        builder: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Return a cached CPU tensor or build and persist it."""
        key = self.signature(metadata)
        if not self.force_rebuild and key in self._memory:
            cached = self._memory[key]
            if not isinstance(cached, torch.Tensor):
                raise ValueError("CTCA cache key collision between tensor payload types")
            return cached

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

    def get_or_create_tensors(
        self,
        metadata: dict,
        *,
        artifact: str,
        builder: Callable[[], dict[str, torch.Tensor]],
        required_keys: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Return a cached CPU tensor payload and whether it was rebuilt."""
        key = f"{artifact}:{self.signature(metadata)}"
        if not self.force_rebuild and key in self._memory:
            cached = self._memory[key]
            if not isinstance(cached, dict):
                raise ValueError("CTCA cache key collision between tensor payload types")
            return cached, False

        path = self.path_for(metadata, artifact=artifact)
        if not self.force_rebuild and path is not None and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("metadata") != metadata:
                raise ValueError(f"Malformed CTCA cache metadata in {path}")
            tensors = payload.get("tensors")
            if not isinstance(tensors, dict):
                raise ValueError(f"Malformed CTCA tensor payload in {path}")
            self._validate_tensor_payload(tensors, required_keys, path)
            self._memory[key] = tensors
            return tensors, False

        tensors = builder()
        self._validate_tensor_payload(tensors, required_keys, None)
        tensors = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
        self._memory[key] = tensors
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"metadata": metadata, "tensors": tensors}, path)
        return tensors, True

    @staticmethod
    def _validate_tensor_payload(
        tensors: dict[str, torch.Tensor],
        required_keys: tuple[str, ...],
        path: Path | None,
    ) -> None:
        label = f" in {path}" if path is not None else ""
        for name in required_keys:
            tensor = tensors.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Malformed CTCA tensor {name!r}{label}")
            if not torch.isfinite(tensor.float()).all():
                raise ValueError(f"CTCA tensor {name!r} must be finite{label}")


def tokenizer_vocab_signature(tokenizer) -> str:
    """Hash a tokenizer vocabulary independently of dictionary ordering."""
    vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in tokenizer.get_vocab().items()
    )
    payload = json.dumps(vocabulary, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
