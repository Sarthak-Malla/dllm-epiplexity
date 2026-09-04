"""Analyze the frozen P2.5 proxy diagnostic on a CPU login node.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_proxy_diagnostics.py \
        --states /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_5_balanced_chunk8/states.jsonl \
        --source-configuration /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_5_balanced_chunk8/configuration.json \
        --output-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_6_analysis \
        --expected-state-count 256 \
        --expected-manifest-sha256 879d69edbbd7998792d9d340f43907698aca9fb5d63bd611f11679641f025651 \
        --dependency-field after_sink --confidence-exponent 1.0 \
        --random-seed 42 --recall-ks 2 4 8 \
        --bootstrap-samples 2000 --bootstrap-seed 42
"""

import argparse
import json
from pathlib import Path

from dllm.core.samplers.proxy_analysis import run_proxy_analysis


def parse_args() -> argparse.Namespace:
    """Parse an explicit and independently reproducible analysis command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--dependency-field",
        choices=("before_sink", "after_sink"),
        required=True,
    )
    parser.add_argument("--confidence-exponent", type=float, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--recall-ks", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the read-only analysis and print its machine-readable summary."""
    args = parse_args()
    summary = run_proxy_analysis(
        states_path=args.states,
        source_configuration_path=args.source_configuration,
        output_directory=args.output_directory,
        expected_state_count=args.expected_state_count,
        expected_manifest_sha256=args.expected_manifest_sha256,
        dependency_field=args.dependency_field,
        confidence_exponent=args.confidence_exponent,
        random_seed=args.random_seed,
        recall_ks=args.recall_ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
