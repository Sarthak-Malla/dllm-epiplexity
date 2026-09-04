"""Build and validate frozen proxy-diagnostic example manifests.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_proxy_manifest.py -v
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProxyManifestExample:
    """One exact dataset prompt selected for proxy-state collection."""

    dataset_label: str
    dataset_path: str
    dataset_config: str | None
    dataset_revision: str
    dataset_fingerprint: str
    split: str
    dataset_index: int
    example_id: str
    source_id: str
    prompt: str
    prompt_format: str
    num_fewshot: int
    selection_seed: int

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous manifest records."""
        for name in (
            "dataset_label",
            "dataset_path",
            "dataset_revision",
            "dataset_fingerprint",
            "split",
            "example_id",
            "source_id",
            "prompt",
            "prompt_format",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string.")
        if self.dataset_config is not None and (
            not isinstance(self.dataset_config, str) or not self.dataset_config
        ):
            raise ValueError("dataset_config must be a nonempty string or None.")
        for name in ("dataset_index", "num_fewshot"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed,
            int,
        ):
            raise ValueError("selection_seed must be an integer.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record with an explicit schema version."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> "ProxyManifestExample":
        """Parse one strict manifest record."""
        if not isinstance(record, Mapping):
            raise TypeError("Manifest record must be a mapping.")
        expected_keys = {"schema_version", *cls.__dataclass_fields__}
        if set(record) != expected_keys:
            missing = sorted(expected_keys - set(record))
            extra = sorted(set(record) - expected_keys)
            raise ValueError(
                f"Manifest record keys differ; missing={missing}, extra={extra}."
            )
        if record["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Manifest schema version does not match the current reader."
            )
        values = {name: record[name] for name in cls.__dataclass_fields__}
        return cls(**values)


GSM8K_COT_FEWSHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer "
        "is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer "
        "is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom "
        "and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
        "then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
)


def format_gsm8k_cot_prompt(question: str, *, num_fewshot: int) -> str:
    """Format a fixed first-N GSM8K chain-of-thought diagnostic prompt."""
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a nonempty string.")
    if isinstance(num_fewshot, bool) or not isinstance(num_fewshot, int):
        raise ValueError("num_fewshot must be an integer.")
    if not 0 <= num_fewshot <= len(GSM8K_COT_FEWSHOT):
        raise ValueError(
            f"num_fewshot must be between 0 and {len(GSM8K_COT_FEWSHOT)}."
        )
    sections = [
        f"Q: {fewshot_question}\n\nA: {fewshot_answer}"
        for fewshot_question, fewshot_answer in GSM8K_COT_FEWSHOT[:num_fewshot]
    ]
    sections.append(f"Q: {question}\n\nA:")
    return "\n\n".join(sections)


def format_humaneval_instruct_prompt(prompt: str) -> str:
    """Apply the repository's HumanEval instruction wrapper."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string.")
    return f"Complete the following python code:\n{prompt}"


def select_dataset_indices(
    dataset_size: int,
    sample_count: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[int, ...]:
    """Select a deterministic sorted sample using a namespace-derived seed."""
    for name, value in (
        ("dataset_size", dataset_size),
        ("sample_count", sample_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if sample_count > dataset_size:
        raise ValueError("sample_count cannot exceed dataset_size.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a nonempty string.")
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = random.Random(derived_seed)
    return tuple(sorted(generator.sample(range(dataset_size), sample_count)))


def interleave_balanced_examples(
    first: Sequence[ProxyManifestExample],
    second: Sequence[ProxyManifestExample],
) -> tuple[ProxyManifestExample, ...]:
    """Interleave two equal task lists so partial runs remain task-balanced."""
    if len(first) != len(second):
        raise ValueError("Balanced task lists must have equal lengths.")
    return tuple(
        example
        for pair in zip(first, second)
        for example in pair
    )


def _canonical_json(value: object) -> str:
    """Serialize manifest data deterministically and reject NaN values."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("Manifest data must be JSON-compatible.") from error


def validate_manifest_examples(
    examples: Sequence[ProxyManifestExample],
) -> None:
    """Require a nonempty manifest with unique IDs and dataset rows."""
    if not examples:
        raise ValueError("Manifest must contain at least one example.")
    if not all(isinstance(example, ProxyManifestExample) for example in examples):
        raise TypeError("Manifest entries must be ProxyManifestExample values.")
    example_ids = [example.example_id for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Manifest example IDs must be unique.")
    rows = [
        (
            example.dataset_path,
            example.dataset_config,
            example.dataset_revision,
            example.split,
            example.dataset_index,
        )
        for example in examples
    ]
    if len(rows) != len(set(rows)):
        raise ValueError("Manifest dataset rows must be unique.")


def manifest_fingerprint(examples: Sequence[ProxyManifestExample]) -> str:
    """Hash the ordered, canonical manifest contents with SHA-256."""
    validate_manifest_examples(examples)
    payload = [example.to_dict() for example in examples]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def write_or_validate_manifest(
    path: Path,
    examples: Sequence[ProxyManifestExample],
) -> bool:
    """Atomically create a manifest, or validate an identical existing one."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    validate_manifest_examples(examples)
    serialized = "".join(
        _canonical_json(example.to_dict()) + "\n" for example in examples
    )
    if path.exists():
        existing = read_manifest(path)
        if tuple(existing) != tuple(examples):
            raise ValueError(f"Existing manifest differs from requested data: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)
    return True


def read_manifest(path: Path) -> tuple[ProxyManifestExample, ...]:
    """Read and strictly validate a frozen JSONL manifest."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid manifest JSONL at {path}:{line_number}."
                ) from error
            try:
                examples.append(ProxyManifestExample.from_dict(record))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid manifest record at {path}:{line_number}: {error}"
                ) from error
    validate_manifest_examples(examples)
    return tuple(examples)
