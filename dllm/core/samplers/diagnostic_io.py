"""
Persist proxy diagnostic states as resumable, configuration-safe JSONL.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    TEST_ROOT=/home/sarthak.malla/dllm-selection-ensemble/scripts/tests
    pytest "${TEST_ROOT}/test_proxy_diagnostic_io.py" -v
"""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = 1


class ProxyConfigurationMismatchError(ValueError):
    """Raised when an output directory contains a different configuration."""


def _canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value deterministically."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("Diagnostic metadata must be JSON-compatible.") from error


def _json_copy(value: object) -> object:
    """Return a detached JSON-compatible copy of a value."""
    return json.loads(_canonical_json(value))


def configuration_fingerprint(configuration: Mapping[str, object]) -> str:
    """Return a full SHA-256 fingerprint for a run configuration."""
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping.")
    canonical = _canonical_json(dict(configuration)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _stable_record_id(prefix: str, payload: Mapping[str, object]) -> str:
    """Build a compact stable identifier from canonical record identity."""
    digest = hashlib.sha256(
        _canonical_json(dict(payload)).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def build_proxy_state_id(
    *,
    config_fingerprint: str,
    example_id: str,
    prompt: str,
    state_index: int,
    target_mask_ratio: float,
) -> str:
    """Build a stable state ID before expensive model collection begins."""
    return _stable_record_id(
        "proxy-state",
        {
            "config_fingerprint": config_fingerprint,
            "example_id": example_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "state_index": state_index,
            "target_mask_ratio": float(target_mask_ratio),
        },
    )


def build_proxy_failure_id(
    *,
    config_fingerprint: str,
    example_id: str,
    prompt: str,
    failure_kind: str,
) -> str:
    """Build a stable failure ID so repeated OOMs do not duplicate records."""
    return _stable_record_id(
        "proxy-failure",
        {
            "config_fingerprint": config_fingerprint,
            "example_id": example_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "failure_kind": failure_kind,
        },
    )


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one strict JSON object from disk."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON metadata from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Metadata at {path} must contain a JSON object.")
    return value


def _read_jsonl_ids(
    path: Path,
    *,
    id_field: str,
    expected_fingerprint: str,
    expected_status: str,
) -> set[str]:
    """Read and validate IDs from one append-only diagnostic JSONL file."""
    if not path.exists():
        return set()
    identifiers = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Record at {path}:{line_number} is not an object.")
            identifier = record.get(id_field)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    f"Record at {path}:{line_number} lacks {id_field}."
                )
            fingerprint = record.get("configuration_fingerprint")
            if fingerprint != expected_fingerprint:
                raise ProxyConfigurationMismatchError(
                    f"Record at {path}:{line_number} has configuration "
                    f"fingerprint {fingerprint!r}, expected "
                    f"{expected_fingerprint!r}."
                )
            if record.get("status") != expected_status:
                raise ValueError(
                    f"Record at {path}:{line_number} has status "
                    f"{record.get('status')!r}, expected {expected_status!r}."
                )
            if identifier in identifiers:
                raise ValueError(f"Duplicate {id_field} {identifier!r} in {path}.")
            identifiers.add(identifier)
    return identifiers


class ProxyDiagnosticStore:
    """Append and resume fingerprinted proxy state and failure records."""

    def __init__(
        self,
        output_directory: Path,
        *,
        configuration: Mapping[str, object],
        environment: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(output_directory, Path):
            raise TypeError("output_directory must be a pathlib.Path.")
        self.output_directory = output_directory
        self.metadata_path = output_directory / "configuration.json"
        self.states_path = output_directory / "states.jsonl"
        self.failures_path = output_directory / "failures.jsonl"
        self.configuration = _json_copy(dict(configuration))
        self.environment = _json_copy(dict(environment or {}))
        self.configuration_fingerprint = configuration_fingerprint(configuration)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._initialize_or_validate_metadata()
        self._completed_state_ids = _read_jsonl_ids(
            self.states_path,
            id_field="state_id",
            expected_fingerprint=self.configuration_fingerprint,
            expected_status="completed",
        )
        self._failure_ids = _read_jsonl_ids(
            self.failures_path,
            id_field="failure_id",
            expected_fingerprint=self.configuration_fingerprint,
            expected_status="failed",
        )

    @property
    def completed_state_ids(self) -> frozenset[str]:
        """Return state IDs already durably present in states.jsonl."""
        return frozenset(self._completed_state_ids)

    @property
    def failure_ids(self) -> frozenset[str]:
        """Return failure IDs already durably present in failures.jsonl."""
        return frozenset(self._failure_ids)

    def _initialize_or_validate_metadata(self) -> None:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": self.configuration_fingerprint,
            "configuration": self.configuration,
            "creation_environment": self.environment,
        }
        if self.metadata_path.exists():
            existing = _read_json_object(self.metadata_path)
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise ProxyConfigurationMismatchError(
                    "Diagnostic schema version does not match the current writer."
                )
            if existing.get("configuration") != self.configuration:
                raise ProxyConfigurationMismatchError(
                    "Output directory already contains a different configuration."
                )
            if (
                existing.get("configuration_fingerprint")
                != self.configuration_fingerprint
            ):
                raise ProxyConfigurationMismatchError(
                    "Stored configuration fingerprint is inconsistent."
                )
            return

        temporary_path = self.metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.metadata_path)

    def _append_jsonl(self, path: Path, record: Mapping[str, object]) -> None:
        serialized = _canonical_json(dict(record))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_state(
        self,
        state_id: str,
        state: Mapping[str, object],
    ) -> bool:
        """Append and flush one completed state, or skip an existing ID."""
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a nonempty string.")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        if state_id in self._completed_state_ids:
            return False
        record = dict(_json_copy(dict(state)))
        record.update(
            {
                "state_id": state_id,
                "status": "completed",
                "configuration_fingerprint": self.configuration_fingerprint,
            }
        )
        self._append_jsonl(self.states_path, record)
        self._completed_state_ids.add(state_id)
        return True

    def append_failure(
        self,
        failure_id: str,
        failure: Mapping[str, object],
    ) -> bool:
        """Append and flush one failure record, or skip an existing ID."""
        if not isinstance(failure_id, str) or not failure_id:
            raise ValueError("failure_id must be a nonempty string.")
        if not isinstance(failure, Mapping):
            raise TypeError("failure must be a mapping.")
        if failure_id in self._failure_ids:
            return False
        record = dict(_json_copy(dict(failure)))
        record.update(
            {
                "failure_id": failure_id,
                "status": "failed",
                "configuration_fingerprint": self.configuration_fingerprint,
            }
        )
        self._append_jsonl(self.failures_path, record)
        self._failure_ids.add(failure_id)
        return True
