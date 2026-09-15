"""Atomic artifacts and identities for the training-free decoding experiments.

Import from the experiment runner. Run its tests on a compute node with:
    python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_artifacts.py
Prepare the dllm environment and use the user-provided srun workflow first.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def _canonical(value: Any) -> Any:
    """Represent tensors by exact bytes without storing their full values in JSON."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor_dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"array_dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_hash(value: Any) -> str:
    """Hash complete state/configuration identities, including tensor values."""
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def jsonable(value: Any) -> Any:
    """Convert small diagnostic values to JSON; snapshots use torch serialization."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after its complete contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid artifact component: {value!r}")
    return value


class RunStore:
    """An immutable manifest and individually resumable state/branch units.

    One process writes a run directory at a time. Different stages reuse the same
    manifest; stage/arm choices belong in their records, not in that identity.
    Snapshot files are trusted local outputs, never arbitrary downloaded pickles.
    """

    def __init__(self, root: Path | str, manifest: dict, resume: bool = False):
        self.root = Path(root).resolve()
        self.on_write = None
        self.manifest = jsonable(manifest)
        identity = {key: value for key, value in self.manifest.items() if key != "created_at"}
        self.manifest_hash = stable_hash(identity)
        path = self.root / "manifest.json"
        if path.exists():
            if not resume:
                raise FileExistsError(f"Run exists; pass --resume explicitly: {self.root}")
            old = json.loads(path.read_text())
            old_identity = {key: value for key, value in old.items() if key != "created_at"}
            if stable_hash(old_identity) != self.manifest_hash:
                raise ValueError("Resume rejected: model, documents, configuration, or source hashes changed.")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError(f"Nonempty run directory has no manifest: {self.root}")
            self.manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            atomic_json(path, self.manifest)

    def _path(self, kind: str, key: str) -> Path:
        return self.root / _component(kind) / f"{_component(key)}.json"

    def has(self, kind: str, key: str) -> bool:
        return self._path(kind, key).is_file()

    def get_json(self, kind: str, key: str) -> dict:
        path = self._path(kind, key)
        record = json.loads(path.read_text())
        if record.get("_manifest_hash") != self.manifest_hash:
            raise ValueError(f"Artifact belongs to another manifest: {path}")
        content = {name: value for name, value in record.items() if name != "_manifest_hash"}
        return content

    def put_json(self, kind: str, key: str, data: dict) -> None:
        value = jsonable(data)
        if "_manifest_hash" in value:
            raise ValueError("_manifest_hash is reserved for artifact provenance.")
        if self.has(kind, key):
            if stable_hash(self.get_json(kind, key)) != stable_hash(value):
                raise ValueError(f"Refusing to overwrite a different completed {kind} unit: {key}")
            return
        atomic_json(self._path(kind, key), {**value, "_manifest_hash": self.manifest_hash})
        if self.on_write is not None:
            self.on_write(kind, value, unit_id=key)

    def put_snapshot(self, key: str, state_dict: dict) -> None:
        import torch

        key = _component(key)
        identity = stable_hash(state_dict)
        if self.has("snapshot_index", key):
            if self.get_json("snapshot_index", key)["state_hash"] != identity:
                raise ValueError(f"Snapshot collision: {key}")
            self.get_snapshot(key)  # Verify the existing payload before reusing it.
            return
        directory = self.root / "snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.pt"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(state_dict, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.put_json("snapshot_index", key, {"state_hash": identity, "path": str(path)})

    def get_snapshot(self, key: str) -> dict:
        import torch

        metadata = self.get_json("snapshot_index", key)
        path = self.root / "snapshots" / f"{_component(key)}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if stable_hash(state) != metadata["state_hash"]:
            raise ValueError(f"Snapshot payload does not match its recorded identity: {path}")
        return state

    def record_reuse(self, kind: str, key: str, source_key: str) -> None:
        event = {"kind": kind, "key": key, "source_key": source_key}
        self.put_json("reuse", stable_hash(event), event)

    def mark_complete(self, stage: str, summary: dict) -> None:
        self.put_json("completed", stage, {"stage": stage, **summary})
