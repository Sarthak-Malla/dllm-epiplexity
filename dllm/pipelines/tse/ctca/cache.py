"""Deterministic in-memory or on-disk matrix caching for CTCA.

Run the focused tests with:
    pytest scripts/tests/test_ctca.py -v
"""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
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
        self._memory: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    @staticmethod
    def signature(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        except TypeError as error:
            raise ValueError("cache metadata must be JSON serializable") from error
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, metadata: dict, artifact: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{artifact}_{self.signature(metadata)}.pt"

    def get_or_create_tensors(
        self,
        metadata: dict,
        *,
        artifact: str,
        builder: Callable[[], dict[str, torch.Tensor]],
        required_keys: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Return a current-run CPU payload, optionally loading it from disk."""
        key = (artifact, self.signature(metadata))
        if key in self._memory:
            cached = self._memory[key]
            if not all(name in cached for name in required_keys):
                raise ValueError("CTCA cache payload is missing required tensors")
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
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary_path = Path(handle.name)
            try:
                torch.save({"metadata": metadata, "tensors": tensors}, temporary_path)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        self._memory[key] = tensors
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
    """Hash a tokenizer vocabulary and its special IDs."""
    vocabulary = sorted(
        (str(token), int(token_id))
        for token, token_id in tokenizer.get_vocab().items()
    )
    return CTCACacheManager.signature(
        {
            "vocabulary": vocabulary,
            "special_ids": sorted(set(getattr(tokenizer, "all_special_ids", ()))),
            "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        }
    )
