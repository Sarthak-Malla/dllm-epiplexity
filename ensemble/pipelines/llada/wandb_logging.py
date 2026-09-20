"""Minimal decoding metrics for the LLaDA evaluation entry point.

Run via /home/sarthak.malla/dllm-learning-decoding-path/ensemble/pipelines/llada/eval.py
with --wandb_args project=dllm-ensemble. WANDB_LOG_EVERY sets the logging interval.
"""

import os
import sys
from collections import Counter
from dataclasses import dataclass, field

from lm_eval.loggers import WandbLogger


class MinimalWandbLogger(WandbLogger):
    """Keep native final-result logging while leaving generated samples local."""

    def log_eval_samples(self, samples):
        # The harness still writes --log_samples to disk. Avoid uploading prompt
        # and response tables when only scores and decoding metrics were requested.
        pass


@dataclass
class _PositionCountStats:
    """Exact means and frequencies of position counts per active sequence-step."""

    total: int = 0
    count: int = 0
    frequencies: Counter = field(default_factory=Counter)
    window_total: int = 0
    window_count: int = 0

    def add(self, positions):
        self.total += positions
        self.count += 1
        self.window_total += positions
        self.window_count += 1
        self.frequencies[positions] += 1

    def histogram(self):
        from wandb import Histogram

        # Integer-centered bins preserve individual counts for small blocks.
        # Larger blocks use wider bins, without clipping counts at 100 or
        # retaining a history proportional to the number of decoding steps.
        maximum = max(self.frequencies)
        width = max(1, (maximum + 64) // 64)
        bins = [0] * (maximum // width + 1)
        for positions, observations in self.frequencies.items():
            bins[positions // width] += observations
        edges = [i * width - 0.5 for i in range(len(bins) + 1)]
        return Histogram(np_histogram=(bins, edges))


class DecodingMetricsLogger:
    """Aggregate metrics from rank zero's uncached decoding steps.

    tokens_per_sequence_step is revealed tokens divided by active sequence-steps.
    expansion_rate is expanded sequence-steps divided by active sequence-steps.
    remaining_masks_in_block is the latest batch total for the current block.
    Overlap and proposal counts weight active sequence-steps equally. Overlap
    histograms accumulate every observed count, including between W&B uploads.
    """

    def __init__(self, run, log_every=25):
        if log_every < 1:
            raise ValueError("WANDB_LOG_EVERY must be a positive integer")
        self.run = run
        self.log_every = log_every
        self.steps = 0
        self.tokens = 0
        self.active = 0
        self.expanded = 0
        self.has_expansion_metrics = False
        self.pending_steps = 0
        self.pending_tokens = 0
        self.pending_active = 0
        self.pending_expanded = 0
        self.remaining_masks = 0
        self.overlap = {}
        self.proposals = {}
        run.config.update(
            {
                "decoding_metrics_scope": "rank_0_uncached_requests",
                "decoding_log_every": log_every,
                "decoding_overlap_unit": "positions",
                "decoding_proposals_unit": "positions",
                "decoding_position_count_aggregation": "active_sequence_step_mean",
                "decoding_overlap_histogram_max_bins": 64,
            }
        )
        run.define_metric("decoding/step")
        run.define_metric("decoding/*", step_metric="decoding/step")

    def __call__(self, metrics):
        self.steps += 1
        self.tokens += metrics["tokens_committed"]
        self.active += metrics["active_sequences"]
        self.expanded += metrics.get("expanded_sequences", 0)
        self.has_expansion_metrics |= "expanded_sequences" in metrics
        self.pending_steps += 1
        self.pending_tokens += metrics["tokens_committed"]
        self.pending_active += metrics["active_sequences"]
        self.pending_expanded += metrics.get("expanded_sequences", 0)
        self.remaining_masks = metrics["remaining_masks"]
        for metric, destination in (
            ("position_overlap", self.overlap),
            ("position_proposals", self.proposals),
        ):
            for name, counts in metrics.get(metric, {}).items():
                if name not in destination:
                    destination[name] = _PositionCountStats()
                for positions in counts:
                    destination[name].add(positions)
        if self.pending_steps >= self.log_every:
            self.flush()

    def flush(self):
        if not self.pending_steps:
            return
        values = {
            "decoding/step": self.steps,
            "decoding/tokens_per_sequence_step": (
                self.pending_tokens / max(1, self.pending_active)
            ),
            "decoding/remaining_masks_in_block": self.remaining_masks,
        }
        summary = {
            "decoding/observed_steps": self.steps,
            "decoding/mean_tokens_per_sequence_step": self.tokens / max(1, self.active),
        }
        if self.has_expansion_metrics:
            values["decoding/expansion_rate"] = (
                self.pending_expanded / max(1, self.pending_active)
            )
            summary["decoding/mean_expansion_rate"] = self.expanded / max(1, self.active)
        for group, observations in (
            ("overlap", self.overlap), ("proposals", self.proposals)
        ):
            for name, stats in observations.items():
                if not stats.window_count:
                    continue
                prefix = f"decoding/{group}/{name}"
                values[f"{prefix}_count"] = stats.window_total / stats.window_count
                if group == "overlap":
                    histogram = stats.histogram()
                    values[f"{prefix}_count_distribution"] = histogram
                    summary[f"{prefix}_count_distribution"] = histogram
                summary[f"{prefix}_mean_count"] = stats.total / stats.count
                summary[f"{prefix}_observations"] = stats.count
                stats.window_total = 0
                stats.window_count = 0
        # Let W&B assign its history step so native final-result logging can
        # append normally. decoding/step supplies the separate decoding axis.
        self.run.log(values)
        self.run.summary.update(summary)
        self.pending_steps = 0
        self.pending_tokens = 0
        self.pending_active = 0
        self.pending_expanded = 0


def get_decoding_logger(rank):
    """Attach only to an already initialized main-process W&B run."""
    if rank != 0:
        return None
    # Native W&B initialization already imports the optional SDK. Leave it
    # optional when evaluating without --wandb_args.
    run = getattr(sys.modules.get("wandb"), "run", None)
    if run is None or getattr(run, "disabled", False):
        return None
    return DecodingMetricsLogger(run, log_every=int(os.environ.get("WANDB_LOG_EVERY", 25)))
