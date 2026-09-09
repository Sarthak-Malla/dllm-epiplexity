"""Plot action-size density and evaluate large-action GSM8K generations.

Run from any directory with:
    source /home/sarthak.malla/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/analyze_large_actions.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path

MPLCONFIG_DIRECTORY = Path("/scratch/sarthak.malla/tmp/matplotlib")
MPLCONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIRECTORY))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
DEFAULT_RUN_DIRECTORY = (
    ROOT
    / "eval_results/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42"
)
DEFAULT_REFERENCE_DIRECTORY = (
    ROOT
    / "eval_results/path_selection/dependency_guided/p8/reduced_full_v3/full"
    / "gsm8k_cot/dependency_entropy_budget_n4/seed42"
)
DEFAULT_OUTPUT_DIRECTORY = (
    ROOT
    / "eval_results/path_selection/ablation/2_entropy_budget_without_k_limit"
    / "vectorized_soft_full_v1/large_action_analysis"
)
SAMPLE_COUNT = 1319
RESPONSE_TOKENS = 256
MAXIMUM_ACTION_BINS = (
    ("<48", 1, 47),
    ("48-55", 48, 55),
    ("56-62", 56, 62),
    ("63", 63, 63),
    ("64", 64, 64),
)


def _only_path(directory: Path, pattern: str) -> Path:
    """Return the only path matching a required artifact pattern."""
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern!r} below {directory}, found {len(matches)}."
        )
    return matches[0]


def _load_outcomes(directory: Path) -> dict[int, dict[str, int]]:
    """Load strict and flexible correctness for every scored document."""
    samples_path = _only_path(directory, "samples_gsm8k_cot_*.jsonl")
    outcomes: dict[int, dict[str, int]] = {}
    for line in samples_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        filter_name = str(record["filter"])
        if filter_name not in {"strict-match", "flexible-extract"}:
            continue
        doc_id = int(record["doc_id"])
        outcomes.setdefault(doc_id, {})[filter_name] = int(record["exact_match"])
    expected_ids = set(range(SAMPLE_COUNT))
    if set(outcomes) != expected_ids:
        raise ValueError("Sample artifacts do not contain all GSM8K document IDs.")
    if any(len(filters) != 2 for filters in outcomes.values()):
        raise ValueError("Each GSM8K document must have both evaluation filters.")
    return outcomes


def _load_traces(directory: Path) -> dict[int, list[dict[str, object]]]:
    """Map distributed local trace indices back to GSM8K document IDs."""
    paths = sorted(directory.glob("results.json_entropy_drop_diagnostics_rank*.json"))
    if not paths:
        raise FileNotFoundError(f"No distributed diagnostics below {directory}.")
    world_size = len(paths)
    traces: dict[int, list[dict[str, object]]] = {}
    for rank, path in enumerate(paths):
        examples = json.loads(path.read_text())
        doc_ids = list(range(rank, SAMPLE_COUNT, world_size))
        if len(examples) < len(doc_ids):
            raise ValueError(f"Missing diagnostic examples in {path}.")
        # The shorter rank is padded to equal length for distributed generation.
        # Only the leading real examples correspond to scored dataset documents.
        for doc_id, example in zip(doc_ids, examples[: len(doc_ids)]):
            steps = example.get("steps", [])
            if not steps:
                raise ValueError(f"Document {doc_id} has no retained steps in {path}.")
            traces[doc_id] = steps
    if set(traces) != set(range(SAMPLE_COUNT)):
        raise ValueError("Diagnostics could not be aligned to every GSM8K document.")
    return traces


def _wilson_interval(correct: int, count: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson binomial interval."""
    z = 1.959963984540054
    probability = correct / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    """Return the exact two-sided paired-binomial McNemar p-value."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * math.ldexp(float(tail), -discordant))


def _generation_rows(
    traces: dict[int, list[dict[str, object]]],
    outcomes: dict[int, dict[str, int]],
    reference_outcomes: dict[int, dict[str, int]],
) -> list[dict[str, float | int]]:
    """Create one action-profile and quality row per scored generation."""
    rows = []
    for doc_id in range(SAMPLE_COUNT):
        action_sizes = np.asarray(
            [int(step["commit_k"]) for step in traces[doc_id]], dtype=np.int64
        )
        committed_tokens = int(action_sizes.sum())
        if committed_tokens != RESPONSE_TOKENS:
            raise ValueError(
                f"Document {doc_id} committed {committed_tokens} tokens, "
                f"expected {RESPONSE_TOKENS}."
            )
        row: dict[str, float | int] = {
            "doc_id": doc_id,
            "action_count": len(action_sizes),
            "mean_action_size": float(action_sizes.mean()),
            "max_action_size": int(action_sizes.max()),
        }
        for threshold in (4, 8, 16, 32):
            row[f"token_share_above_{threshold}"] = float(
                action_sizes[action_sizes > threshold].sum() / RESPONSE_TOKENS
            )
        row.update(
            {
                "uncapped_flexible_correct": outcomes[doc_id]["flexible-extract"],
                "uncapped_strict_correct": outcomes[doc_id]["strict-match"],
                "capped_flexible_correct": reference_outcomes[doc_id][
                    "flexible-extract"
                ],
                "capped_strict_correct": reference_outcomes[doc_id]["strict-match"],
            }
        )
        rows.append(row)
    return rows


def _evaluate_group(
    label: str,
    rows: list[dict[str, float | int]],
) -> dict[str, float | int | str]:
    """Summarize conditional quality and the paired capped comparison."""
    count = len(rows)
    flexible = sum(int(row["uncapped_flexible_correct"]) for row in rows)
    strict = sum(int(row["uncapped_strict_correct"]) for row in rows)
    reference_flexible = sum(int(row["capped_flexible_correct"]) for row in rows)
    reference_strict = sum(int(row["capped_strict_correct"]) for row in rows)
    wins = sum(
        int(row["uncapped_flexible_correct"])
        and not int(row["capped_flexible_correct"])
        for row in rows
    )
    losses = sum(
        int(row["capped_flexible_correct"])
        and not int(row["uncapped_flexible_correct"])
        for row in rows
    )
    flexible_low, flexible_high = _wilson_interval(flexible, count)
    strict_low, strict_high = _wilson_interval(strict, count)
    return {
        "maximum_action_size_bin": label,
        "generation_count": count,
        "mean_of_generation_mean_action_size": float(
            np.mean([float(row["mean_action_size"]) for row in rows])
        ),
        "uncapped_flexible_correct": flexible,
        "uncapped_flexible_accuracy": flexible / count,
        "uncapped_flexible_wilson95_low": flexible_low,
        "uncapped_flexible_wilson95_high": flexible_high,
        "uncapped_strict_correct": strict,
        "uncapped_strict_accuracy": strict / count,
        "uncapped_strict_wilson95_low": strict_low,
        "uncapped_strict_wilson95_high": strict_high,
        "capped_flexible_correct": reference_flexible,
        "capped_flexible_accuracy": reference_flexible / count,
        "capped_strict_correct": reference_strict,
        "capped_strict_accuracy": reference_strict / count,
        "uncapped_minus_capped_flexible_pp": (
            100.0 * (flexible - reference_flexible) / count
        ),
        "uncapped_flexible_paired_wins": wins,
        "uncapped_flexible_paired_losses": losses,
        "uncapped_flexible_mcnemar_exact_p": _exact_mcnemar_p(wins, losses),
    }


def _maximum_action_evaluations(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int | str]]:
    """Evaluate fixed, interpretable bins of maximum generation action size."""
    evaluations = []
    for label, lower, upper in MAXIMUM_ACTION_BINS:
        selected = [
            row
            for row in rows
            if lower <= int(row["max_action_size"]) <= upper
        ]
        if not selected:
            raise ValueError(f"Maximum-action bin {label} is empty.")
        evaluations.append(_evaluate_group(label, selected))
    if sum(int(row["generation_count"]) for row in evaluations) != SAMPLE_COUNT:
        raise ValueError("Maximum-action bins do not partition the evaluation set.")
    return evaluations


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty list of dictionaries as CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    action_sizes: np.ndarray,
    evaluations: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Plot discrete action density and quality by maximum-action bin."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 160,
        }
    )
    figure, (density_axis, quality_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        gridspec_kw={"height_ratios": (1.08, 1.0)},
        constrained_layout=True,
    )

    sizes = np.arange(1, 65)
    counts = np.bincount(action_sizes, minlength=65)[1:]
    action_density = counts / counts.sum()
    token_density = sizes * counts / np.dot(sizes, counts)
    density_axis.bar(
        sizes,
        action_density,
        width=0.86,
        color="#4472C4",
        alpha=0.72,
        label="Share of decoding actions",
    )
    density_axis.plot(
        sizes,
        token_density,
        color="#D95F02",
        linewidth=2.0,
        label="Share of committed tokens",
    )
    density_axis.set_yscale("log")
    density_axis.set_xlim(0.2, 64.8)
    density_axis.set_xlabel("Simultaneously unmasked tokens (action size)")
    density_axis.set_ylabel("Probability mass (log scale)")
    density_axis.set_title(
        f"Action-size density across {len(action_sizes):,} scored decoding actions"
    )
    density_axis.grid(axis="y", alpha=0.22, which="both")
    for percentile, location in (("p90", 25), ("p95", 47), ("p99", 63)):
        density_axis.axvline(location, color="#555555", linewidth=0.9, alpha=0.55)
        density_axis.text(
            location + 0.6,
            0.13,
            f"{percentile}={location}",
            rotation=90,
            va="top",
            color="#444444",
            fontsize=8,
        )
    density_axis.legend(frameon=False, ncol=2, loc="upper right")

    labels = [str(row["maximum_action_size_bin"]) for row in evaluations]
    x_values = np.arange(len(labels))
    series = (
        (
            "Uncapped flexible",
            "uncapped_flexible_accuracy",
            "uncapped_flexible_wilson95_low",
            "uncapped_flexible_wilson95_high",
            "#4472C4",
            "o",
        ),
        (
            "Uncapped strict",
            "uncapped_strict_accuracy",
            "uncapped_strict_wilson95_low",
            "uncapped_strict_wilson95_high",
            "#D95F02",
            "s",
        ),
    )
    for name, value_key, low_key, high_key, color, marker in series:
        values = np.asarray([float(row[value_key]) for row in evaluations])
        lower = values - np.asarray([float(row[low_key]) for row in evaluations])
        upper = np.asarray([float(row[high_key]) for row in evaluations]) - values
        quality_axis.errorbar(
            x_values,
            values,
            yerr=np.vstack((lower, upper)),
            color=color,
            marker=marker,
            linewidth=2,
            capsize=4,
            label=f"{name} (95% Wilson CI)",
        )
    capped_flexible = [float(row["capped_flexible_accuracy"]) for row in evaluations]
    quality_axis.plot(
        x_values,
        capped_flexible,
        color="#666666",
        marker="^",
        linestyle="--",
        linewidth=1.6,
        label="Capped-k=4 flexible on same questions",
    )
    for index, row in enumerate(evaluations):
        quality_axis.text(
            index,
            0.985,
            f"n={int(row['generation_count'])}",
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
    quality_axis.set_xticks(x_values, labels)
    quality_axis.set_ylim(0.25, 1.0)
    quality_axis.set_xlabel("Generation maximum action size")
    quality_axis.set_ylabel("GSM8K accuracy")
    quality_axis.set_title("Generation quality conditioned on maximum action size")
    quality_axis.grid(axis="y", alpha=0.22)
    quality_axis.legend(frameon=False, loc="lower right")

    figure.suptitle(
        "Uncapped entropy-budget 2.0: large-action behavior",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def analyze(
    run_directory: Path,
    reference_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Run the complete large-action analysis and write reusable artifacts."""
    outcomes = _load_outcomes(run_directory)
    reference_outcomes = _load_outcomes(reference_directory)
    traces = _load_traces(run_directory)
    rows = _generation_rows(traces, outcomes, reference_outcomes)
    evaluations = _maximum_action_evaluations(rows)
    action_sizes = np.asarray(
        [int(step["commit_k"]) for steps in traces.values() for step in steps],
        dtype=np.int64,
    )
    action_counts = Counter(action_sizes.tolist())
    total_tokens = int(action_sizes.sum())
    output_directory.mkdir(parents=True, exist_ok=True)
    generation_path = output_directory / "large_action_generation_metrics.csv"
    evaluation_path = output_directory / "large_action_bin_evaluation.csv"
    plot_path = output_directory / "action_size_density_and_quality.png"
    summary_path = output_directory / "summary.json"
    _write_csv(generation_path, rows)
    _write_csv(evaluation_path, evaluations)
    _plot(action_sizes, evaluations, plot_path)

    summary: dict[str, object] = {
        "schema_version": 1,
        "interpretation": (
            "Conditional accuracy is descriptive, not causal: the entropy-budget "
            "policy selects larger actions in lower-entropy, typically easier states."
        ),
        "scored_generation_count": SAMPLE_COUNT,
        "distributed_padding_traces_excluded": 1,
        "action_count": len(action_sizes),
        "committed_token_count": total_tokens,
        "action_size_counts": {
            str(size): count for size, count in sorted(action_counts.items())
        },
        "action_size_percentiles": {
            "p50": float(np.percentile(action_sizes, 50, method="lower")),
            "p90": float(np.percentile(action_sizes, 90, method="lower")),
            "p95": float(np.percentile(action_sizes, 95, method="lower")),
            "p99": float(np.percentile(action_sizes, 99, method="lower")),
        },
        "large_action_shares": {
            f"above_{threshold}": {
                "action_fraction": float(np.mean(action_sizes > threshold)),
                "committed_token_fraction": float(
                    action_sizes[action_sizes > threshold].sum() / total_tokens
                ),
            }
            for threshold in (4, 8, 16, 32)
        },
        "maximum_action_bin_evaluations": evaluations,
        "artifacts": {
            "generation_metrics_csv": str(generation_path.resolve()),
            "bin_evaluation_csv": str(evaluation_path.resolve()),
            "plot": str(plot_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    """Parse paths, run the analysis, and print its compact result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument(
        "--reference-directory", type=Path, default=DEFAULT_REFERENCE_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    arguments = parser.parse_args()
    summary = analyze(
        arguments.run_directory.resolve(),
        arguments.reference_directory.resolve(),
        arguments.output_directory.resolve(),
    )
    print(
        json.dumps(
            {
                "scored_generation_count": summary["scored_generation_count"],
                "action_count": summary["action_count"],
                "action_size_percentiles": summary["action_size_percentiles"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
