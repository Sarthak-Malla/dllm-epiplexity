"""Compare two Phase-5 traces until their decoding paths diverge.

Run after the chunk-size-1 smoke completes:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/compare_p5_chunking.py \
        --reference-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p5_6/entropy_drop/gsm8k_cot/dependency/n4 \
        --comparison-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p5_6/smoke_chunk1/entropy_drop/gsm8k_cot/dependency/n4 \
        --output-path /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p5_6/smoke_chunk_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_diagnostics(directory: Path) -> list[dict[str, object]]:
    """Load the unique diagnostics sidecar in a run directory."""
    matches = list(directory.glob("results.json_*_diagnostics.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one diagnostics sidecar in {directory}, found {len(matches)}."
        )
    return json.loads(matches[0].read_text())


def _candidate_signature(step: dict[str, object]) -> tuple[tuple[object, ...], ...]:
    """Represent the proposal pool without verifier-dependent values."""
    return tuple(
        (
            candidate["index"],
            candidate["name"],
            candidate["valid"],
            tuple(candidate["positions"]),
            candidate["source"],
            candidate["fallback_source"],
            candidate["proposal_score"],
        )
        for candidate in step["candidates"]
    )


def _selected_signature(step: dict[str, object]) -> tuple[object, ...] | None:
    """Represent the committed action independently of its verifier score."""
    candidate = step.get("selected_candidate")
    if candidate is None:
        return None
    return candidate["index"], tuple(candidate["positions"]), candidate["source"]


def compare(
    reference: list[dict[str, object]],
    comparison: list[dict[str, object]],
) -> dict[str, object]:
    """Compare aligned examples and stop score comparison after first divergence."""
    if len(reference) != len(comparison):
        raise ValueError("Trace files contain different example counts.")
    example_reports = []
    overall_max_score_difference = 0.0
    overall_max_per_heldout_difference = 0.0
    for reference_example, comparison_example in zip(reference, comparison):
        reference_steps = reference_example.get("steps", [])
        comparison_steps = comparison_example.get("steps", [])
        if len(reference_steps) != len(comparison_steps):
            raise ValueError("Aligned examples contain different step counts.")
        first_divergence = None
        pool_mismatch_before_divergence = None
        max_score_difference = 0.0
        max_per_heldout_difference = 0.0
        compared_steps = 0
        for step_index, (reference_step, comparison_step) in enumerate(
            zip(reference_steps, comparison_steps)
        ):
            if _candidate_signature(reference_step) != _candidate_signature(
                comparison_step
            ):
                pool_mismatch_before_divergence = step_index
                break
            reference_candidates = reference_step["candidates"]
            comparison_candidates = comparison_step["candidates"]
            for left, right in zip(reference_candidates, comparison_candidates):
                if not left["valid"]:
                    continue
                difference = abs(left["verifier_score"] - right["verifier_score"])
                max_score_difference = max(max_score_difference, difference)
                heldout = max(int(left["heldout_count"]), 1)
                max_per_heldout_difference = max(
                    max_per_heldout_difference,
                    difference / heldout,
                )
            compared_steps += 1
            if _selected_signature(reference_step) != _selected_signature(
                comparison_step
            ):
                first_divergence = {
                    "step_index": step_index,
                    "remaining_response_masks": reference_step[
                        "remaining_response_masks"
                    ],
                    "reference_selected": reference_step["selected_candidate"],
                    "comparison_selected": comparison_step["selected_candidate"],
                }
                break
        overall_max_score_difference = max(
            overall_max_score_difference, max_score_difference
        )
        overall_max_per_heldout_difference = max(
            overall_max_per_heldout_difference, max_per_heldout_difference
        )
        example_reports.append(
            {
                "example_index": reference_example.get("example_index"),
                "step_count": len(reference_steps),
                "comparable_prefix_steps": compared_steps,
                "candidate_pool_mismatch_before_divergence": (
                    pool_mismatch_before_divergence
                ),
                "first_selection_divergence": first_divergence,
                "all_selections_equal": (
                    first_divergence is None
                    and pool_mismatch_before_divergence is None
                ),
                "max_verifier_score_abs_difference": max_score_difference,
                "max_verifier_score_abs_difference_per_heldout": (
                    max_per_heldout_difference
                ),
            }
        )
    return {
        "schema_version": 1,
        "example_count": len(reference),
        "candidate_pools_equal_through_comparable_prefix": all(
            report["candidate_pool_mismatch_before_divergence"] is None
            for report in example_reports
        ),
        "full_path_candidate_pool_equality": (
            True
            if all(report["all_selections_equal"] for report in example_reports)
            else None
        ),
        "all_selections_equal": all(
            report["all_selections_equal"] for report in example_reports
        ),
        "max_verifier_score_abs_difference": overall_max_score_difference,
        "max_verifier_score_abs_difference_per_heldout": (
            overall_max_per_heldout_difference
        ),
        "examples": example_reports,
    }


def main() -> None:
    """Parse paths, compare traces, and save the audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--comparison-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        _load_diagnostics(args.reference_directory.resolve()),
        _load_diagnostics(args.comparison_directory.resolve()),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
