"""Prepare and collect the balanced P2.5 proxy diagnostic dataset.

Prepare the frozen manifest on a login node with the dllm environment active:

    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/collect_proxy_diagnostic_dataset.py prepare \
        --manifest /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_5_balanced/manifest.jsonl \
        --examples-per-task 32 --seed 42 \
        --gsm8k-dataset-path gsm8k --gsm8k-dataset-config main \
        --gsm8k-revision 740312add88f781978c0658806c59bc2815b9866 \
        --gsm8k-split test --gsm8k-num-fewshot 5 \
        --humaneval-dataset-path openai/openai_humaneval \
        --humaneval-dataset-config openai_humaneval \
        --humaneval-revision 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544 \
        --humaneval-split test --humaneval-num-fewshot 0

The collect subcommand requires the SHA-256 printed by the prepare command. It
loads the checkpoint once, alternates tasks, and safely resumes JSONL states.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForMaskedLM

import dllm.pipelines.llada.models  # noqa: F401
from dllm.core.samplers.diagnostic_io import (
    ProxyDiagnosticStore,
    build_proxy_failure_id,
    build_proxy_state_id,
)
from dllm.core.samplers.proxy_diagnostics import (
    ProxyStateCollectionConfig,
    collect_llada_proxy_states,
    validate_proxy_state_collection_config,
)
from dllm.core.samplers.proxy_manifest import (
    ProxyManifestExample,
    format_gsm8k_cot_prompt,
    format_humaneval_instruct_prompt,
    interleave_balanced_examples,
    manifest_fingerprint,
    read_manifest,
    select_dataset_indices,
    write_or_validate_manifest,
)
from dllm.utils import get_tokenizer


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
EXPECTED_DATASET_LABELS = frozenset({"gsm8k", "humaneval"})


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit dataset and sampling arguments for manifest preparation."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--examples-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gsm8k-dataset-path", required=True)
    parser.add_argument("--gsm8k-dataset-config", required=True)
    parser.add_argument("--gsm8k-revision", required=True)
    parser.add_argument("--gsm8k-split", required=True)
    parser.add_argument("--gsm8k-num-fewshot", type=int, required=True)
    parser.add_argument("--humaneval-dataset-path", required=True)
    parser.add_argument("--humaneval-dataset-config", required=True)
    parser.add_argument("--humaneval-revision", required=True)
    parser.add_argument("--humaneval-split", required=True)
    parser.add_argument("--humaneval-num-fewshot", type=int, required=True)


def _add_collect_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit model, manifest, and collection arguments."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-example-count", type=int, required=True)
    parser.add_argument("--expected-examples-per-task", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--mask-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--last-n-layers", type=int, required=True)
    parser.add_argument("--top-confidence-pool-size", type=int, required=True)
    parser.add_argument("--random-pool-size", type=int, required=True)
    parser.add_argument("--oracle-candidate-chunk-size", type=int, required=True)
    parser.add_argument(
        "--sink-filter",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--sink-quantile", type=float, required=True)
    parser.add_argument(
        "--renormalize-selected-keys",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--progress-every", type=int, required=True)


def parse_args() -> argparse.Namespace:
    """Parse the P2.5 manifest-preparation or collection command."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_arguments(subparsers.add_parser("prepare"))
    _add_collect_arguments(subparsers.add_parser("collect"))
    return parser.parse_args()


def _print_summary(**values: object) -> None:
    """Print one compact machine-readable progress or completion record."""
    print(json.dumps(values, indent=2, sort_keys=True), flush=True)


def _load_dataset(
    *,
    dataset_path: str,
    dataset_config: str,
    revision: str,
    split: str,
):
    """Load one explicitly revision-pinned Hugging Face dataset split."""
    from datasets import load_dataset

    return load_dataset(
        dataset_path,
        dataset_config,
        revision=revision,
        split=split,
    )


def _dataset_namespace(
    *,
    dataset_path: str,
    dataset_config: str,
    revision: str,
    split: str,
) -> str:
    """Build the deterministic namespace used to derive sampling seeds."""
    return f"{dataset_path}:{dataset_config}:{revision}:{split}"


def _prepare_gsm8k_examples(
    args: argparse.Namespace,
    dataset,
) -> tuple[ProxyManifestExample, ...]:
    """Select and format the fixed GSM8K half of the manifest."""
    namespace = _dataset_namespace(
        dataset_path=args.gsm8k_dataset_path,
        dataset_config=args.gsm8k_dataset_config,
        revision=args.gsm8k_revision,
        split=args.gsm8k_split,
    )
    indices = select_dataset_indices(
        len(dataset),
        args.examples_per_task,
        seed=args.seed,
        namespace=namespace,
    )
    fingerprint = str(dataset._fingerprint)
    return tuple(
        ProxyManifestExample(
            dataset_label="gsm8k",
            dataset_path=args.gsm8k_dataset_path,
            dataset_config=args.gsm8k_dataset_config,
            dataset_revision=args.gsm8k_revision,
            dataset_fingerprint=fingerprint,
            split=args.gsm8k_split,
            dataset_index=index,
            example_id=f"gsm8k-{args.gsm8k_split}-{index:05d}",
            source_id=str(index),
            prompt=format_gsm8k_cot_prompt(
                dataset[index]["question"],
                num_fewshot=args.gsm8k_num_fewshot,
            ),
            prompt_format="gsm8k_cot_first_n_v1",
            num_fewshot=args.gsm8k_num_fewshot,
            selection_seed=args.seed,
        )
        for index in indices
    )


def _prepare_humaneval_examples(
    args: argparse.Namespace,
    dataset,
) -> tuple[ProxyManifestExample, ...]:
    """Select and format the fixed HumanEval half of the manifest."""
    if args.humaneval_num_fewshot != 0:
        raise ValueError("The HumanEval diagnostic requires zero few-shot examples.")
    namespace = _dataset_namespace(
        dataset_path=args.humaneval_dataset_path,
        dataset_config=args.humaneval_dataset_config,
        revision=args.humaneval_revision,
        split=args.humaneval_split,
    )
    indices = select_dataset_indices(
        len(dataset),
        args.examples_per_task,
        seed=args.seed,
        namespace=namespace,
    )
    fingerprint = str(dataset._fingerprint)
    return tuple(
        ProxyManifestExample(
            dataset_label="humaneval",
            dataset_path=args.humaneval_dataset_path,
            dataset_config=args.humaneval_dataset_config,
            dataset_revision=args.humaneval_revision,
            dataset_fingerprint=fingerprint,
            split=args.humaneval_split,
            dataset_index=index,
            example_id=f"humaneval-{args.humaneval_split}-{index:05d}",
            source_id=str(dataset[index]["task_id"]),
            prompt=format_humaneval_instruct_prompt(dataset[index]["prompt"]),
            prompt_format="humaneval_instruct_llada_v1",
            num_fewshot=args.humaneval_num_fewshot,
            selection_seed=args.seed,
        )
        for index in indices
    )


def prepare_manifest(args: argparse.Namespace) -> int:
    """Load pinned datasets and create or validate the frozen manifest."""
    if args.examples_per_task <= 0:
        raise ValueError("examples-per-task must be positive.")
    gsm8k = _load_dataset(
        dataset_path=args.gsm8k_dataset_path,
        dataset_config=args.gsm8k_dataset_config,
        revision=args.gsm8k_revision,
        split=args.gsm8k_split,
    )
    humaneval = _load_dataset(
        dataset_path=args.humaneval_dataset_path,
        dataset_config=args.humaneval_dataset_config,
        revision=args.humaneval_revision,
        split=args.humaneval_split,
    )
    examples = interleave_balanced_examples(
        _prepare_gsm8k_examples(args, gsm8k),
        _prepare_humaneval_examples(args, humaneval),
    )
    created = write_or_validate_manifest(args.manifest, examples)
    _print_summary(
        status="created" if created else "already_valid",
        manifest_path=str(args.manifest.resolve()),
        manifest_sha256=manifest_fingerprint(examples),
        example_count=len(examples),
        examples_per_task=args.examples_per_task,
        task_counts=dict(sorted(Counter(x.dataset_label for x in examples).items())),
        gsm8k_dataset_fingerprint=str(gsm8k._fingerprint),
        humaneval_dataset_fingerprint=str(humaneval._fingerprint),
    )
    return 0


def _target_mask_counts(
    response_length: int,
    mask_ratios: tuple[float, ...],
) -> tuple[int, ...]:
    """Calculate the exact active-position count at every saved stage."""
    return tuple(
        max(1, int(math.floor(response_length * ratio + 0.5)))
        for ratio in mask_ratios
    )


def _validate_collect_manifest(
    args: argparse.Namespace,
    examples: tuple[ProxyManifestExample, ...],
) -> str:
    """Reject a changed, unbalanced, or unexpectedly sized manifest."""
    fingerprint = manifest_fingerprint(examples)
    if fingerprint != args.expected_manifest_sha256:
        raise ValueError(
            "Manifest SHA-256 differs from --expected-manifest-sha256: "
            f"{fingerprint} versus {args.expected_manifest_sha256}."
        )
    if len(examples) != args.expected_example_count:
        raise ValueError(
            f"Manifest has {len(examples)} examples, expected "
            f"{args.expected_example_count}."
        )
    counts = Counter(example.dataset_label for example in examples)
    if frozenset(counts) != EXPECTED_DATASET_LABELS:
        raise ValueError(
            f"Manifest dataset labels must be {sorted(EXPECTED_DATASET_LABELS)}."
        )
    if any(count != args.expected_examples_per_task for count in counts.values()):
        raise ValueError(
            "Manifest is not balanced at the expected examples-per-task count: "
            f"{dict(sorted(counts.items()))}."
        )
    if args.expected_example_count != (
        len(EXPECTED_DATASET_LABELS) * args.expected_examples_per_task
    ):
        raise ValueError(
            "Expected total example count is inconsistent with task balance."
        )
    if len(examples) % len(EXPECTED_DATASET_LABELS) != 0:
        raise ValueError("Manifest length is incompatible with alternating tasks.")
    selection_seeds = {example.selection_seed for example in examples}
    if selection_seeds != {args.seed}:
        raise ValueError(
            "Manifest selection seed differs from the collection seed: "
            f"{sorted(selection_seeds)} versus {args.seed}."
        )
    for pair_index in range(0, len(examples), 2):
        labels = (
            examples[pair_index].dataset_label,
            examples[pair_index + 1].dataset_label,
        )
        if labels != ("gsm8k", "humaneval"):
            raise ValueError("Manifest execution order must alternate the two tasks.")
    return fingerprint


def _dataset_sources(
    examples: tuple[ProxyManifestExample, ...],
) -> list[dict[str, object]]:
    """Return the distinct pinned dataset/prompt configurations."""
    records = {
        (
            example.dataset_label,
            example.dataset_path,
            example.dataset_config,
            example.dataset_revision,
            example.dataset_fingerprint,
            example.split,
            example.prompt_format,
            example.num_fewshot,
        )
        for example in examples
    }
    return [
        {
            "dataset_label": values[0],
            "dataset_path": values[1],
            "dataset_config": values[2],
            "dataset_revision": values[3],
            "dataset_fingerprint": values[4],
            "split": values[5],
            "prompt_format": values[6],
            "num_fewshot": values[7],
        }
        for values in sorted(records)
    ]


def _run_configuration(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    examples: tuple[ProxyManifestExample, ...],
    manifest_sha256: str,
    target_mask_counts: tuple[int, ...],
) -> dict[str, object]:
    """Build the semantic configuration protected against mixed resumes."""
    pool_capacity = args.top_confidence_pool_size + args.random_pool_size
    candidates_per_example = sum(
        min(masked_count, pool_capacity) for masked_count in target_mask_counts
    )
    state_count = len(examples) * len(args.mask_ratios)
    oracle_forwards_per_example = sum(
        math.ceil(min(masked_count, pool_capacity) / args.oracle_candidate_chunk_size)
        for masked_count in target_mask_counts
    )
    return {
        "checkpoint": checkpoint,
        "datasets": _dataset_sources(examples),
        "manifest_sha256": manifest_sha256,
        "manifest_example_count": len(examples),
        "examples_per_task": args.expected_examples_per_task,
        "manifest_order_policy": "alternating_gsm8k_humaneval",
        "selection_seed": args.seed,
        "response_length": args.response_length,
        "mask_ratios": list(args.mask_ratios),
        "target_mask_counts": list(target_mask_counts),
        "last_n_layers": args.last_n_layers,
        "top_confidence_pool_size": args.top_confidence_pool_size,
        "random_pool_size": args.random_pool_size,
        "oracle_candidate_chunk_size": args.oracle_candidate_chunk_size,
        "sink_filter_enabled": args.sink_filter,
        "sink_quantile": args.sink_quantile,
        "sink_threshold": None,
        "renormalize_selected_keys": args.renormalize_selected_keys,
        "zero_diagonal": args.zero_diagonal,
        "seed": args.seed,
        "dtype": args.dtype,
        "token_value_policy": "base_argmax",
        "trajectory_policy": "stagewise_highest_confidence",
        "evaluation_pool_policy": "top_confidence_plus_seeded_random",
        "chat_template_applied": True,
        "planned_state_count": state_count,
        "planned_base_forward_count": state_count,
        "planned_batched_oracle_forward_count": (
            len(examples) * oracle_forwards_per_example
        ),
        "planned_candidate_sequence_count": (
            len(examples) * candidates_per_example
        ),
    }


def _expected_state_ids(
    store: ProxyDiagnosticStore,
    examples: tuple[ProxyManifestExample, ...],
    mask_ratios: tuple[float, ...],
) -> dict[str, tuple[str, ...]]:
    """Precompute every durable state ID before model collection."""
    return {
        example.example_id: tuple(
            build_proxy_state_id(
                config_fingerprint=store.configuration_fingerprint,
                example_id=example.example_id,
                prompt=example.prompt,
                state_index=state_index,
                target_mask_ratio=target_ratio,
            )
            for state_index, target_ratio in enumerate(mask_ratios)
        )
        for example in examples
    }


def _build_model_inputs(tokenizer, example: ProxyManifestExample, response_length: int):
    """Apply the chat template and append an all-mask response region."""
    prompt_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": example.prompt}]],
        add_generation_prompt=True,
        tokenize=True,
    )[0]
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
    response_tokens = torch.full(
        (response_length,),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    input_ids = torch.cat([prompt_tensor, response_tokens]).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, prompt_tensor.numel() :] = True
    return input_ids, attention_mask, response_mask, prompt_tensor.numel()


def _record_oom(
    *,
    store: ProxyDiagnosticStore,
    example: ProxyManifestExample,
    input_ids: torch.Tensor,
    prompt_length: int,
    args: argparse.Namespace,
    target_mask_counts: tuple[int, ...],
    expected_ids: tuple[str, ...],
    error: torch.cuda.OutOfMemoryError,
) -> str:
    """Persist one configuration-bound CUDA OOM record."""
    completed = set(expected_ids) & store.completed_state_ids
    failed_state_index = min(len(completed), len(args.mask_ratios) - 1)
    pool_size = min(
        target_mask_counts[failed_state_index],
        args.top_confidence_pool_size + args.random_pool_size,
    )
    failure_id = build_proxy_failure_id(
        config_fingerprint=store.configuration_fingerprint,
        example_id=example.example_id,
        prompt=example.prompt,
        failure_kind="cuda_out_of_memory",
    )
    store.append_failure(
        failure_id,
        {
            "failure_kind": "cuda_out_of_memory",
            "example_id": example.example_id,
            "source": example.to_dict(),
            "error": str(error),
            "sequence_length": int(input_ids.shape[1]),
            "prompt_length": prompt_length,
            "response_length": args.response_length,
            "state_index": failed_state_index,
            "target_mask_ratio": args.mask_ratios[failed_state_index],
            "planned_masked_count": target_mask_counts[failed_state_index],
            "evaluation_pool_size": pool_size,
            "top_confidence_pool_size": args.top_confidence_pool_size,
            "random_pool_size": args.random_pool_size,
            "completed_state_ids": sorted(completed),
        },
    )
    return failure_id


def collect_manifest(args: argparse.Namespace) -> int:
    """Load the checkpoint once and collect every incomplete manifest example."""
    examples = read_manifest(args.manifest)
    manifest_sha256 = _validate_collect_manifest(args, examples)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive.")
    checkpoint = str(args.checkpoint.resolve())
    mask_ratios = tuple(args.mask_ratios)
    config = ProxyStateCollectionConfig(
        mask_ratios=mask_ratios,
        last_n_layers=args.last_n_layers,
        top_confidence_pool_size=args.top_confidence_pool_size,
        random_pool_size=args.random_pool_size,
        oracle_candidate_chunk_size=args.oracle_candidate_chunk_size,
        seed=args.seed,
        renormalize_selected_keys=args.renormalize_selected_keys,
        zero_diagonal=args.zero_diagonal,
        sink_filter_enabled=args.sink_filter,
        sink_quantile=args.sink_quantile,
    )
    validate_proxy_state_collection_config(
        config,
        response_length=args.response_length,
    )
    target_mask_counts = _target_mask_counts(args.response_length, mask_ratios)
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The P2.5 real-checkpoint collector requires a GPU.")
    device_index = requested_device.index if requested_device.index is not None else 0
    torch.cuda.set_device(device_index)
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_device": args.device,
        "gpu_name": torch.cuda.get_device_name(device_index),
    }
    configuration = _run_configuration(
        args,
        checkpoint=checkpoint,
        examples=examples,
        manifest_sha256=manifest_sha256,
        target_mask_counts=target_mask_counts,
    )
    store = ProxyDiagnosticStore(
        args.output_directory,
        configuration=configuration,
        environment=environment,
    )
    expected_ids_by_example = _expected_state_ids(store, examples, mask_ratios)
    all_expected_ids = {
        state_id
        for state_ids in expected_ids_by_example.values()
        for state_id in state_ids
    }
    if all_expected_ids.issubset(store.completed_state_ids):
        _print_summary(
            status="already_completed",
            completed_state_count=len(all_expected_ids),
            manifest_sha256=manifest_sha256,
            states_path=str(store.states_path.resolve()),
        )
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()

    tokenizer = get_tokenizer(model_name_or_path=checkpoint)
    if tokenizer.mask_token_id is None:
        raise ValueError("The loaded tokenizer does not define mask_token_id.")
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            dtype=DTYPES[args.dtype],
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        failure_id = build_proxy_failure_id(
            config_fingerprint=store.configuration_fingerprint,
            example_id="__model_load__",
            prompt=checkpoint,
            failure_kind="cuda_out_of_memory",
        )
        store.append_failure(
            failure_id,
            {
                "failure_kind": "cuda_out_of_memory",
                "example_id": "__model_load__",
                "error": str(error),
                "checkpoint": checkpoint,
                "response_length": args.response_length,
                "evaluation_pool_capacity": (
                    args.top_confidence_pool_size + args.random_pool_size
                ),
            },
        )
        _print_summary(
            status="failed",
            failure_kind="cuda_out_of_memory",
            failure_id=failure_id,
            failures_path=str(store.failures_path.resolve()),
        )
        return 2
    model.eval()
    configured_mask_token_id = getattr(model.config, "mask_token_id", None)
    if (
        configured_mask_token_id is not None
        and configured_mask_token_id != tokenizer.mask_token_id
    ):
        raise ValueError(
            "Tokenizer and model mask token IDs differ: "
            f"{tokenizer.mask_token_id} versus {configured_mask_token_id}."
        )
    maximum_length = int(model.config.max_sequence_length)
    model_device = next(model.parameters()).device
    total_written = 0
    total_skipped = 0
    completed_examples = 0
    invocation_state_count = 0
    invocation_candidate_sequence_count = 0
    invocation_oracle_forward_count = 0

    for example_number, example in enumerate(examples, start=1):
        expected_ids = expected_ids_by_example[example.example_id]
        if set(expected_ids).issubset(store.completed_state_ids):
            total_skipped += len(expected_ids)
            completed_examples += 1
            continue
        input_ids, attention_mask, response_mask, prompt_length = _build_model_inputs(
            tokenizer,
            example,
            args.response_length,
        )
        if input_ids.shape[1] > maximum_length:
            raise ValueError(
                f"Example {example.example_id} sequence length "
                f"{input_ids.shape[1]} exceeds model maximum {maximum_length}."
            )
        example_written = 0
        example_skipped = 0

        def persist_snapshot(snapshot) -> None:
            nonlocal example_written, example_skipped
            nonlocal invocation_candidate_sequence_count, invocation_state_count
            nonlocal invocation_oracle_forward_count
            invocation_state_count += 1
            evaluated_candidates = int(snapshot.evaluation_positions.numel())
            invocation_candidate_sequence_count += evaluated_candidates
            invocation_oracle_forward_count += math.ceil(
                evaluated_candidates / args.oracle_candidate_chunk_size
            )
            state_record = snapshot.to_dict()
            state_record["source"] = {
                **example.to_dict(),
                "chat_template_applied": True,
                "manifest_sha256": manifest_sha256,
            }
            if store.append_state(
                expected_ids[snapshot.state_index],
                state_record,
            ):
                example_written += 1
            else:
                example_skipped += 1

        try:
            collect_llada_proxy_states(
                model,
                input_ids.to(model_device),
                attention_mask=attention_mask.to(model_device),
                response_mask=response_mask.to(model_device),
                mask_token_id=tokenizer.mask_token_id,
                example_id=example.example_id,
                checkpoint=checkpoint,
                config=config,
                snapshot_callback=persist_snapshot,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            failure_id = _record_oom(
                store=store,
                example=example,
                input_ids=input_ids,
                prompt_length=prompt_length,
                args=args,
                target_mask_counts=target_mask_counts,
                expected_ids=expected_ids,
                error=error,
            )
            _print_summary(
                status="partial_failed",
                failure_kind="cuda_out_of_memory",
                failure_id=failure_id,
                example_id=example.example_id,
                error=str(error),
                completed_state_count=len(
                    all_expected_ids & store.completed_state_ids
                ),
                failures_path=str(store.failures_path.resolve()),
            )
            return 2
        missing = set(expected_ids) - store.completed_state_ids
        if missing:
            raise RuntimeError(
                f"Example {example.example_id} did not persist states: "
                f"{sorted(missing)}"
            )
        total_written += example_written
        total_skipped += example_skipped
        completed_examples += 1
        if (
            example_number % args.progress_every == 0
            or example_number == len(examples)
        ):
            _print_summary(
                event="progress",
                example_number=example_number,
                example_count=len(examples),
                completed_examples=completed_examples,
                completed_state_count=len(
                    all_expected_ids & store.completed_state_ids
                ),
                last_example_id=example.example_id,
                last_dataset=example.dataset_label,
            )

    missing = all_expected_ids - store.completed_state_ids
    if missing:
        raise RuntimeError(
            f"Collection ended with {len(missing)} missing expected states."
        )
    elapsed = time.perf_counter() - started
    _print_summary(
        status="completed",
        manifest_sha256=manifest_sha256,
        example_count=len(examples),
        state_count=len(all_expected_ids),
        written_state_count=total_written,
        skipped_state_count=total_skipped,
        durable_candidate_sequence_count=(
            len(examples)
            * sum(
                min(
                    masked_count,
                    args.top_confidence_pool_size + args.random_pool_size,
                )
                for masked_count in target_mask_counts
            )
        ),
        invocation_candidate_sequence_count=invocation_candidate_sequence_count,
        invocation_base_forward_count=invocation_state_count,
        invocation_batched_oracle_forward_count=invocation_oracle_forward_count,
        runtime_seconds=elapsed,
        peak_allocated_gib=(
            torch.cuda.max_memory_allocated(device_index) / (1024**3)
        ),
        peak_reserved_gib=(
            torch.cuda.max_memory_reserved(device_index) / (1024**3)
        ),
        configuration_path=str(store.metadata_path.resolve()),
        states_path=str(store.states_path.resolve()),
        failures_path=str(store.failures_path.resolve()),
    )
    return 0


def main() -> int:
    """Dispatch manifest preparation or real-checkpoint collection."""
    args = parse_args()
    if args.command == "prepare":
        return prepare_manifest(args)
    return collect_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
