"""Evaluate LLaDA baselines and ensembles through the existing lm-eval harness.

Preview a configured run with:
    bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh candidate_expansion --dry-run

After activating the dllm environment, inspect the harness CLI with:
    python /home/sarthak.malla/dllm-learning-decoding-path/ensemble/pipelines/llada/eval.py --help

Use --model llada_ensemble and --model_args sampler_type=greedy,... for the
scheduled baseline. Other sampler types are min_entropy, max_top2_prob,
candidate_expansion, majority_voting, and all. Ensemble strategy lists use semicolons,
e.g. strategies=[low_confidence;min_entropy;max_top2_prob], inside quoted model_args.
"""

import os
import sys
from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.registry import register_model

from dllm.core.eval.mdlm import _parse_token_list
from dllm.core.samplers import EnsembleSampler, EnsembleSamplerConfig, MDLMSampler
from dllm.pipelines.llada.eval import (
    LLaDAEvalConfig,
    LLaDAEvalHarness,
    LLaDAEvalSamplerConfig,
)
from ensemble.pipelines.llada.wandb_logging import (
    MinimalWandbLogger,
    get_decoding_logger,
)


BASELINE_STRATEGIES = {
    "greedy": "low_confidence",
    "min_entropy": "min_entropy",
    "max_top2_prob": "max_top2_prob",
}
ENSEMBLE_POLICIES = {"candidate_expansion", "majority_voting", "all"}


@dataclass
class LLaDAEnsembleEvalSamplerConfig(EnsembleSamplerConfig):
    """Match the native LLaDA evaluation generation defaults."""

    max_new_tokens: int = 1024
    block_size: int = 1024


class LLaDAScheduledStrategySampler(EnsembleSampler):
    """Use the native MDLM decoding schedule with a single scoring strategy."""

    def __post_init__(self):
        MDLMSampler.__post_init__(self)

    # Reuse MDLM's loops directly; EnsembleSampler only supplies position scores.
    sample = MDLMSampler.sample
    infill = MDLMSampler.infill


def _parse_strategies(value):
    """Normalize lm-eval's semicolon-delimited string to a strategy tuple."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        value = tuple(part.strip() for part in value.split(";") if part.strip())
    elif isinstance(value, (tuple, list)):
        value = tuple(value)
    else:
        raise ValueError(
            "strategies must be a list, tuple, or semicolon-delimited string"
        )

    if not value or len(set(value)) != len(value):
        raise ValueError("strategies must contain distinct strategy names")
    supported = {"low_confidence", "min_entropy", "max_top2_prob", "random"}
    if any(strategy not in supported for strategy in value):
        raise ValueError(f"Unknown strategy in {value}")
    return value


@register_model("llada_ensemble")
class LLaDAEnsembleEvalHarness(LLaDAEvalHarness):
    """Select the decoder while reusing LLaDA's model and evaluation integration.

    The greedy and single-strategy baselines use MDLM's scheduled transfers.
    Agreement modes run until blocks finish without a schedule. The all policy
    uses scheduled proposal counts and commits their union.
    """

    def __init__(self, sampler_type: str = "greedy", **kwargs):
        if "candidate_growth" in kwargs:
            raise ValueError(
                "This sampler does not use candidate_growth; candidate_fraction "
                "applies to the remaining masks at each step"
            )
        if sampler_type in BASELINE_STRATEGIES:
            invalid = {
                "strategies", "candidate_fraction", "ensemble_policy"
            } & kwargs.keys()
            if invalid:
                raise ValueError(
                    f"Baseline {sampler_type!r} does not use {sorted(invalid)}"
                )
            strategy = BASELINE_STRATEGIES[sampler_type]
            if kwargs.get("remasking", strategy) != strategy:
                raise ValueError(
                    f"sampler_type={sampler_type} requires remasking={strategy}"
                )
            sampler_config = LLaDAEvalSamplerConfig(remasking=strategy)
            sampler_cls = (
                MDLMSampler
                if sampler_type == "greedy"
                else LLaDAScheduledStrategySampler
            )
        elif sampler_type in ENSEMBLE_POLICIES:
            unused = {"remasking"}
            if sampler_type != "all":
                unused |= {"steps", "stochastic_transfer"}
            invalid = unused & kwargs.keys()
            if invalid:
                raise ValueError(
                    f"Sampler {sampler_type!r} "
                    f"does not use {sorted(invalid)}"
                )
            if kwargs.get("ensemble_policy", sampler_type) != sampler_type:
                raise ValueError("ensemble_policy must agree with sampler_type")
            sampler_config = LLaDAEnsembleEvalSamplerConfig(
                ensemble_policy=sampler_type
            )
            kwargs["strategies"] = _parse_strategies(
                kwargs.get("strategies", sampler_config.strategies)
            )
            if sampler_type != "all":
                candidate_fraction = float(
                    kwargs.get("candidate_fraction", sampler_config.candidate_fraction)
                )
                if not 0 < candidate_fraction <= 1:
                    raise ValueError("candidate_fraction must be in (0, 1]")
                kwargs["candidate_fraction"] = candidate_fraction
            elif int(kwargs.get("steps", sampler_config.steps)) < 1:
                raise ValueError("steps must be positive")
            sampler_cls = EnsembleSampler
        else:
            available = sorted(set(BASELINE_STRATEGIES) | ENSEMBLE_POLICIES)
            raise ValueError(
                f"Unknown sampler_type: {sampler_type!r}; choose from {available}"
            )

        # MDLMEvalHarness already parses the two suppression lists. CFG's keep
        # list needs the same treatment before the base harness builds configs.
        if "cfg_keep_tokens" in kwargs:
            kwargs["cfg_keep_tokens"] = _parse_token_list(kwargs["cfg_keep_tokens"])

        super().__init__(
            eval_config=LLaDAEvalConfig(),
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )
        self.sampler_type = sampler_type
        self.decoding_logger = get_decoding_logger(self.rank)
        if self.decoding_logger is not None:
            self.sampler.step_callback = self.decoding_logger

    def generate_until(self, requests):
        try:
            return super().generate_until(requests)
        finally:
            if self.decoding_logger is not None:
                self.decoding_logger.flush()


def main(args=None):
    """Run the native CLI with one W&B run and local-only sample logging."""
    args = args if args is not None else parse_eval_args(setup_parser())
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        # Native lm-eval initializes its W&B logger before constructing the
        # distributed model, so guard here to prevent a run per GPU worker.
        args.wandb_args = ""
        args.wandb_config_args = ""

    if not args.wandb_args:
        return cli_evaluate(args)

    import lm_eval.loggers
    import wandb

    original_logger = lm_eval.loggers.WandbLogger
    # The CLI has no separate flag to keep local samples without uploading them.
    # Adapt only its logger for this invocation; native evaluation and final
    # result serialization remain unchanged.
    lm_eval.loggers.WandbLogger = MinimalWandbLogger
    try:
        return cli_evaluate(args)
    finally:
        lm_eval.loggers.WandbLogger = original_logger
        # The CLI finishes successful runs itself. Close any run left by an error.
        if wandb.run is not None:
            error = sys.exc_info()[1]
            succeeded = error is None or (
                isinstance(error, SystemExit) and error.code in (None, 0)
            )
            wandb.run.finish(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
