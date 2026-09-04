"""Run the predeclared P2.7 frozen-state dependency variant analysis.

Run on a CPU login node with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/analyze_proxy_variants.py \
        --states /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_5_balanced_chunk8/states.jsonl \
        --source-configuration /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_5_balanced_chunk8/configuration.json \
        --output-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/proxy_diagnostics/p2_7_variants \
        --expected-state-count 256 \
        --expected-manifest-sha256 879d69edbbd7998792d9d340f43907698aca9fb5d63bd611f11679641f025651 \
        --dependency-fields before_sink after_sink \
        --directions incoming outgoing symmetric \
        --confidence-exponents 0.0 0.5 1.0 2.0 \
        --target-weightings uniform entropy \
        --random-seed 42 --recall-ks 2 4 8 \
        --bootstrap-samples 2000 --bootstrap-seed 42
"""

import argparse
import json
from pathlib import Path

from dllm.core.samplers.proxy_variants import run_proxy_variant_analysis


def parse_args() -> argparse.Namespace:
    """Parse an explicit, fingerprinted P2.7 analysis command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--dependency-fields",
        choices=("before_sink", "after_sink"),
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--directions",
        choices=("incoming", "outgoing", "symmetric"),
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--confidence-exponents", type=float, nargs="+", required=True
    )
    parser.add_argument(
        "--target-weightings",
        choices=("uniform", "entropy"),
        nargs="+",
        required=True,
    )
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--recall-ks", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Run P2.7 and print its machine-readable summary."""
    args = parse_args()
    summary = run_proxy_variant_analysis(
        states_path=args.states,
        source_configuration_path=args.source_configuration,
        output_directory=args.output_directory,
        expected_state_count=args.expected_state_count,
        expected_manifest_sha256=args.expected_manifest_sha256,
        dependency_fields=args.dependency_fields,
        directions=args.directions,
        confidence_exponents=args.confidence_exponents,
        target_weightings=args.target_weightings,
        random_seed=args.random_seed,
        recall_ks=args.recall_ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
