"""Frozen experiment configurations; inspect with runner.py --dry-run collect.

Run only through the documented Slurm commands in this experiment directory.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
CHECKPOINT = Path(
    "/home/sarthak.malla/.cache/huggingface/hub/"
    "models--GSAI-ML--LLaDA-8B-Instruct/snapshots/"
    "08b83a6feb34df1a6011b80c3c00c7563e963b07"
)
DEFAULT_OUTPUT = ROOT / "eval_results/path_selection/experiments/training_free_v1"
DOCUMENT_IDS = tuple(range(100))
SNAPSHOT_THRESHOLDS = (0, 85, 170)
THRESHOLDS = (0.80, 0.90, 0.95)
EXPERIMENTS = ("selectors", "precedence", "proposals", "attention_mass")


def reference_config():
    """Return the original adaptive reference with every relevant knob fixed."""
    from dllm.core.samplers.entropy_drop import EntropyDropSamplerConfig

    return EntropyDropSamplerConfig(
        max_length=4096, max_new_tokens=256, steps=64, block_size=64, temperature=0.0,
        cfg_scale=0.0, stochastic_transfer=False, return_dict=True,
        proposal_strategy="dependency", candidate_budget=4, candidate_chunk_size=1,
        dependency_commit_k=1, dependency_parallel_variant="soft_full",
        dependency_cardinality_strategy="entropy_budget",
        dependency_max_action_size=4, dependency_action_sizes="1|2|4",
        dependency_size_scoring="per_token", dependency_utility_threshold=0.0,
        dependency_entropy_budget=2.0, dependency_immediate_cost_weight=1.0,
        dependency_size_penalty=0.0, dependency_last_n_layers=4,
        dependency_direction="outgoing", dependency_target_weighting="entropy",
        dependency_position_temperature=1.0, dependency_confidence_exponent=0.0,
        dependency_seed_strategy="legacy", dependency_seed_entropy_weight=0.0,
        dependency_generation_seed=42, dependency_sink_filter_enabled=True,
        dependency_sink_quantile=0.99, dependency_sink_threshold=None,
        dependency_zero_diagonal=True, dependency_renormalize_selected_keys=True,
        dependency_fallback_strategy="dependency_only",
        dependency_conflict_normalization="max", dependency_conflict_penalty=1.0,
        dependency_hard_conflict_threshold=0.25, dependency_anchor_support_weight=1.0,
        dependency_anchor_confidence_threshold=0.8, diagnostic_metadata=True,
        dependency_candidate_selector="entropy_drop",
        commit_mode="simultaneous", dependency_preserve_attention_mass=False,
        dependency_budget_search="first_unaffordable",
    )


def continuation_config():
    """Use the same cheap adaptive policy after every diagnostic intervention."""
    return replace(reference_config(), dependency_candidate_selector="max_confidence")


def fixed_config(*, confidence_proposals=False):
    """Fix action size while retaining legacy seed and companion mechanics."""
    return replace(
        reference_config(), dependency_cardinality_strategy="fixed", dependency_commit_k=4,
        dependency_parallel_variant="top_confidence" if confidence_proposals else "soft_full",
    )


def benchmark_configs():
    """Enumerate isolated deployable arms; callers must select one explicitly."""
    cheap = continuation_config()
    arms = {
        "reference_entropy": reference_config(),
        "reference_cheap": cheap,
        "first_action_entropy": cheap,
        "fixed4_entropy": fixed_config(),
        "fixed4_cheap": replace(fixed_config(), dependency_candidate_selector="max_confidence"),
        "attention_mass": replace(cheap, dependency_preserve_attention_mass=True,
                                  dependency_renormalize_selected_keys=False),
        "affordable_companions": replace(cheap, dependency_budget_search="best_affordable"),
        "seed_first": replace(cheap, commit_mode="seed_first"),
    }
    for threshold in THRESHOLDS:
        for block in (64, 256):
            for ranking in ("confidence", "incoming"):
                name = f"threshold_{threshold:.2f}_block{block}_{ranking}"
                arms[name] = replace(
                    cheap, proposal_strategy="confidence_threshold", block_size=block,
                    confidence_threshold=threshold, confidence_ranking=ranking,
                    dependency_cardinality_strategy="fixed", candidate_budget=1,
                    dependency_preserve_attention_mass=True,
                    dependency_renormalize_selected_keys=False,
                )
    return {name: replace(config, diagnostic_metadata=False, return_dict=False)
            for name, config in arms.items()}


def benchmark_selector_schedule(arm):
    """Describe selection by response-global action index, never by block index."""
    selector = benchmark_configs()[arm].dependency_candidate_selector
    return {
        "scope": "response",
        "first_action": "entropy_drop" if arm == "first_action_entropy" else selector,
        "remaining_actions": selector,
    }


def suite_configuration():
    """Return immutable settings shared by collection, diagnostics, and benchmarks."""
    return {
        "schema_version": 1, "suite": "training_free_v1", "task": "gsm8k_cot",
        "document_ids": list(DOCUMENT_IDS), "num_fewshot": 5,
        "evaluation_seeds": [0, 1234, 1234, 1234], "generation_seed": 42,
        "checkpoint": str(CHECKPOINT), "dtype": "bfloat16", "batch_size": 1,
        "snapshot_thresholds": list(SNAPSHOT_THRESHOLDS), "response_cache": False,
        "request_cache": False,
        "external_logging": {"provider": "wandb", "default_mode": "online"},
        "reference": asdict(reference_config()),
        "continuation": asdict(continuation_config()),
        "fixed4": asdict(fixed_config()),
        "benchmarks": {name: asdict(config) for name, config in benchmark_configs().items()},
        "benchmark_selector_schedules": {
            name: benchmark_selector_schedule(name) for name in benchmark_configs()
        },
    }
