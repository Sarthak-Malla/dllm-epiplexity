"""Run CPU checks after activating the dllm environment:

python -m pytest /home/sarthak.malla/dllm-learning-decoding-path/scripts/tests/test_ensemble_llada_eval.py

Model loading and Accelerate are replaced with CPU fixtures; no downloads occur.
"""

from types import SimpleNamespace
import sys

import pytest
import torch
import wandb
from lm_eval.loggers import WandbLogger
from lm_eval.api.registry import get_model

from dllm.core.samplers import EnsembleSampler, EnsembleSamplerConfig, MDLMSamplerConfig
from ensemble.pipelines.llada.eval import (
    LLaDAEnsembleEvalHarness,
    LLaDAScheduledStrategySampler,
)
from ensemble.pipelines.llada import eval as ensemble_eval
from ensemble.pipelines.llada.wandb_logging import (
    DecodingMetricsLogger,
    MinimalWandbLogger,
    get_decoding_logger,
)


@pytest.fixture
def cpu_loader(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("device_anchor", torch.empty(0))

        @property
        def device(self):
            return self.device_anchor.device

        def forward(self, input_ids, attention_mask=None):
            logits = torch.full((*input_ids.shape, 5), -10.0, device=self.device)
            logits[..., 0] = 0.0
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        name_or_path = "test/llada"
        mask_token_id = 4
        eos_token_id = 3
        bos_token_id = 2
        pad_token_id = None

        def __call__(self, context):
            return {"input_ids": [1] * len(context)}

        def decode(self, ids, skip_special_tokens=True):
            return "a" * len(ids)

        def apply_chat_template(self, messages, **kwargs):
            return "chat:" + messages[0]["content"]

    loaded = []

    def load_model(args, config):
        loaded.append(args)
        return Model()

    monkeypatch.setattr("dllm.utils.get_model", load_model)
    monkeypatch.setattr("dllm.utils.get_tokenizer", lambda args: Tokenizer())
    monkeypatch.setattr(
        "dllm.core.eval.base.accelerate.Accelerator",
        lambda: SimpleNamespace(num_processes=1),
    )
    return loaded


@pytest.mark.parametrize(
    "sampler_type",
    [
        "greedy", "min_entropy", "max_top2_prob",
        "candidate_expansion", "majority_voting",
    ],
)
def test_harness_parses_model_args_and_generates(cpu_loader, sampler_type):
    model_args = (
        f"sampler_type={sampler_type},pretrained=test/llada,max_new_tokens=3,"
        "block_size=2,temperature=0.0,cfg_scale=0.0,"
        "suppress_tokens=[3],begin_suppress_tokens=[],cfg_keep_tokens=[1;2]"
    )
    is_ensemble = sampler_type in {"candidate_expansion", "majority_voting"}
    if is_ensemble:
        model_args += (
            ",candidate_fraction=0.10,"
            "strategies=[low_confidence;min_entropy;max_top2_prob]"
        )
    else:
        model_args += ",steps=2"

    harness = LLaDAEnsembleEvalHarness.create_from_arg_string(
        model_args, {"batch_size": 1, "device": "cpu"}
    )
    assert get_model("llada_ensemble") is LLaDAEnsembleEvalHarness
    assert cpu_loader[0].model_name_or_path == "test/llada"
    assert harness.sampler_config.suppress_tokens == [3]
    assert harness.sampler_config.begin_suppress_tokens == []
    assert harness.sampler_config.cfg_keep_tokens == [1, 2]
    assert harness.sampler_config.max_new_tokens == 3
    assert harness.sampler_type == sampler_type
    if is_ensemble:
        assert type(harness.sampler) is EnsembleSampler
        assert isinstance(harness.sampler_config, EnsembleSamplerConfig)
        assert not hasattr(harness.sampler_config, "steps")
        assert harness.sampler.scheduler is None
        assert harness.sampler_config.ensemble_policy == sampler_type
        assert harness.sampler_config.candidate_fraction == 0.10
        assert not hasattr(harness.sampler_config, "candidate_growth")
        assert harness.sampler_config.strategies == (
            "low_confidence", "min_entropy", "max_top2_prob"
        )
    else:
        assert isinstance(harness.sampler_config, MDLMSamplerConfig)
        assert harness.sampler.scheduler is not None
        assert harness.sampler_config.steps == 2
        assert harness.sampler_config.remasking == (
            "low_confidence" if sampler_type == "greedy" else sampler_type
        )
        if sampler_type != "greedy":
            assert isinstance(harness.sampler, LLaDAScheduledStrategySampler)

    requests = [
        SimpleNamespace(args=("xx", {"until": ["aa"]})),
        SimpleNamespace(args=("x", {"until": ["stop"]})),
    ]
    assert harness.generate_until(requests) == ["", "aaa"]
    assert harness.apply_chat_template([{"role": "user", "content": "hello"}]) == (
        "chat:hello"
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sampler_type": "missing"}, "Unknown sampler_type"),
        ({"sampler_type": "greedy", "remasking": "random"}, "requires remasking"),
        ({"sampler_type": "greedy", "candidate_fraction": 0.5}, "does not use"),
        ({"sampler_type": "candidate_expansion", "steps": 64}, "does not use"),
        (
            {"sampler_type": "majority_voting", "remasking": "min_entropy"},
            "does not use",
        ),
        ({"sampler_type": "candidate_expansion", "strategies": "[]"}, "distinct"),
        (
            {"sampler_type": "candidate_expansion", "strategies": "[bad]"},
            "Unknown strategy",
        ),
        (
            {"sampler_type": "candidate_expansion", "candidate_fraction": 0.0},
            "candidate_fraction",
        ),
        (
            {"sampler_type": "candidate_expansion", "ensemble_policy": "majority_voting"},
            "must agree",
        ),
    ],
)
def test_invalid_selection_fails_before_loading_model(cpu_loader, kwargs, message):
    with pytest.raises(ValueError, match=message):
        LLaDAEnsembleEvalHarness(pretrained="test/llada", device="cpu", **kwargs)
    assert not cpu_loader


@pytest.mark.parametrize("sampler_type", ["candidate_expansion", "majority_voting"])
def test_ensemble_candidate_fraction_default(cpu_loader, sampler_type):
    harness = LLaDAEnsembleEvalHarness(
        sampler_type=sampler_type, pretrained="test/llada", device="cpu",
    )
    assert harness.sampler_config.candidate_fraction == 0.10
    assert not hasattr(harness.sampler_config, "candidate_growth")


@pytest.mark.parametrize("sampler_type", [
    "greedy", "min_entropy", "max_top2_prob",
    "candidate_expansion", "majority_voting",
])
def test_legacy_candidate_growth_fails_before_loading_model(cpu_loader, sampler_type):
    with pytest.raises(ValueError, match="does not use candidate_growth"):
        LLaDAEnsembleEvalHarness.create_from_arg_string(
            f"sampler_type={sampler_type},pretrained=test/llada,candidate_growth=0.25",
            {"device": "cpu"},
        )
    assert not cpu_loader


class FakeRun:
    """Collect scalar logs without starting W&B or accessing the network."""

    disabled = False

    def __init__(self):
        class Config(dict):
            def update(self, values, **kwargs):
                super().update(values)

        self.config = Config()
        self.summary = {}
        self.records = []
        self.axes = []
        self.exit_codes = []

    def define_metric(self, name, **kwargs):
        self.axes.append((name, kwargs))

    def log(self, values, **kwargs):
        self.records.append(dict(values))

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


def test_decoding_metrics_are_weighted_and_flush_partial_window():
    run = FakeRun()
    logger = DecodingMetricsLogger(run, log_every=2)
    logger.flush()
    assert run.records == []
    logger({"tokens_committed": 4, "active_sequences": 2,
            "remaining_masks": 5, "expanded_sequences": 1})
    assert run.records == []
    logger({"tokens_committed": 3, "active_sequences": 1,
            "remaining_masks": 2, "expanded_sequences": 1})
    assert run.records == [{
        "decoding/step": 2,
        "decoding/tokens_per_sequence_step": pytest.approx(7 / 3),
        "decoding/remaining_masks_in_block": 2,
        "decoding/expansion_rate": pytest.approx(2 / 3),
    }]
    logger({"tokens_committed": 2, "active_sequences": 1,
            "remaining_masks": 0, "expanded_sequences": 0})
    logger.flush()
    logger.flush()
    assert len(run.records) == 2
    assert run.records[-1]["decoding/tokens_per_sequence_step"] == 2
    assert run.summary == {
        "decoding/observed_steps": 3,
        "decoding/mean_tokens_per_sequence_step": 9 / 4,
        "decoding/mean_expansion_rate": 1 / 2,
    }
    assert run.config["decoding_metrics_scope"] == "rank_0_uncached_requests"
    assert ("decoding/*", {"step_metric": "decoding/step"}) in run.axes


def test_position_counts_include_every_sequence_step_across_flushes():
    run = FakeRun()
    logger = DecodingMetricsLogger(run, log_every=2)
    before = "before_expansion/all"
    pair = "before_expansion/low_confidence__min_entropy"
    after = "after_expansion/all"
    proposed_before = "before_expansion/min_entropy"
    proposed_after = "after_expansion/min_entropy"
    logger({
        "tokens_committed": 2, "active_sequences": 2, "remaining_masks": 8,
        "position_overlap": {before: [0, 2], pair: [2, 4], after: [1, 2]},
        "position_proposals": {proposed_before: [2, 4], proposed_after: [3, 4]},
    })
    assert not run.records
    logger({
        "tokens_committed": 1, "active_sequences": 1, "remaining_masks": 7,
        "position_overlap": {before: [4], pair: [4], after: [4]},
        "position_proposals": {proposed_before: [4], proposed_after: [4]},
    })
    prefix = f"decoding/overlap/{before}"
    first = run.records[0]
    assert first[f"{prefix}_count"] == 2  # Three equally weighted row-steps.
    histogram = first[f"{prefix}_count_distribution"]
    assert isinstance(histogram, wandb.Histogram)
    assert histogram.bins == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert sum(histogram.histogram) == 3
    assert [histogram.histogram[i] for i in (0, 2, 4)] == [1, 1, 1]
    assert first[f"decoding/overlap/{pair}_count"] == pytest.approx(10 / 3)
    assert first[f"decoding/overlap/{after}_count"] == pytest.approx(7 / 3)
    assert first[f"decoding/proposals/{proposed_before}_count"] == pytest.approx(10 / 3)
    assert first[f"decoding/proposals/{proposed_after}_count"] == pytest.approx(11 / 3)
    logger({
        "tokens_committed": 1, "active_sequences": 1, "remaining_masks": 0,
        "position_overlap": {before: [1], pair: [1], after: [1]},
        "position_proposals": {proposed_before: [1], proposed_after: [1]},
    })
    logger.flush()
    logger.flush()
    assert len(run.records) == 2
    assert run.records[-1][f"{prefix}_count"] == 1
    assert run.summary[f"{prefix}_mean_count"] == 7 / 4
    assert run.summary[f"{prefix}_observations"] == 4
    final_histogram = run.summary[f"{prefix}_count_distribution"]
    assert sum(final_histogram.histogram) == 4
    assert final_histogram.histogram[1] == 1
    assert histogram.histogram[1] == 0  # Earlier uploads are immutable snapshots.
    assert run.records[-1][f"decoding/proposals/{proposed_before}_count"] == 1
    assert run.summary[f"decoding/proposals/{proposed_before}_mean_count"] == 11 / 4
    assert run.summary[f"decoding/proposals/{proposed_after}_mean_count"] == 3
    assert run.summary[f"decoding/proposals/{proposed_after}_observations"] == 4
    assert run.config["decoding_overlap_unit"] == "positions"
    assert run.config["decoding_proposals_unit"] == "positions"
    assert not any("pct" in key for record in run.records for key in record)


def test_overlap_histogram_handles_zero_and_large_position_counts():
    run = FakeRun()
    logger = DecodingMetricsLogger(run, log_every=1)
    prefix = "decoding/overlap/before_expansion/all"
    for count in (0, 64, 128, 1024):
        logger({
            "tokens_committed": count, "active_sequences": 1, "remaining_masks": 0,
            "position_overlap": {"before_expansion/all": [count]},
        })
        assert run.records[-1][f"{prefix}_count"] == count
    histogram = run.summary[f"{prefix}_count_distribution"]
    assert sum(histogram.histogram) == 4
    assert len(histogram.histogram) <= 64
    for count in (0, 64, 128, 1024):
        assert sum(
            observations
            for start, end, observations in zip(
                histogram.bins, histogram.bins[1:], histogram.histogram
            )
            if start <= count < end
        ) == 1
    assert run.summary[f"{prefix}_mean_count"] == (64 + 128 + 1024) / 4


@pytest.mark.parametrize("sampler_type", [
    "greedy", "min_entropy", "max_top2_prob",
    "candidate_expansion", "majority_voting",
])
def test_harness_logs_actual_decoding_and_flushes(cpu_loader, monkeypatch, sampler_type):
    run = FakeRun()
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setenv("WANDB_LOG_EVERY", "100")
    options = {} if sampler_type in ensemble_eval.ENSEMBLE_POLICIES else {"steps": 2}
    harness = LLaDAEnsembleEvalHarness(
        sampler_type=sampler_type, pretrained="test/llada", device="cpu",
        max_new_tokens=3, block_size=2, batch_size=1, **options,
    )
    assert harness.sampler.step_callback is harness.decoding_logger
    assert harness.generate_until([
        SimpleNamespace(args=("x", {"until": []})),
    ]) == ["aaa"]
    assert len(run.records) == 1  # The partial interval flushes at generation end.
    assert run.records[0]["decoding/remaining_masks_in_block"] == 0
    steps = run.summary["decoding/observed_steps"]
    assert run.summary["decoding/mean_tokens_per_sequence_step"] == 3 / steps
    assert ("decoding/expansion_rate" in run.records[0]) == (
        sampler_type in ensemble_eval.ENSEMBLE_POLICIES
    )
    overlap_keys = [key for key in run.records[0] if key.endswith("_count_distribution")]
    proposal_keys = [key for key in run.records[0] if key.startswith("decoding/proposals/")]
    if sampler_type in ensemble_eval.ENSEMBLE_POLICIES:
        # All strategies propose and agree on one position per step.
        assert len(overlap_keys) == 8  # All-way + three pairs, before and after.
        for key in overlap_keys:
            histogram = run.records[0][key]
            assert sum(histogram.histogram) == steps
            assert histogram.bins == [-0.5, 0.5, 1.5]
            assert histogram.histogram == [0, steps]
            assert run.records[0][key.removesuffix("_distribution")] == 1
        assert len(proposal_keys) == 6  # Three strategies, before and after.
        assert all(run.records[0][key] == 1 for key in proposal_keys)
    else:
        assert overlap_keys == []
        assert proposal_keys == []


def test_decoding_logging_requires_enabled_rank_zero_run(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    assert get_decoding_logger(0) is None
    run = FakeRun()
    monkeypatch.setattr(wandb, "run", run)
    assert get_decoding_logger(1) is None
    run.disabled = True
    assert get_decoding_logger(0) is None
    assert not run.axes
    monkeypatch.delitem(sys.modules, "wandb")
    assert get_decoding_logger(0) is None


def test_minimal_logger_keeps_native_final_metrics_without_sample_uploads(monkeypatch):
    logger = MinimalWandbLogger.__new__(MinimalWandbLogger)
    logger.run = FakeRun()
    logger.step = None
    # Native table/artifact transport is outside this CPU scalar logging check.
    monkeypatch.setattr(logger, "_log_results_as_table", lambda: None)
    monkeypatch.setattr(logger, "_log_results_as_artifact", lambda: None)
    monkeypatch.setattr(
        WandbLogger, "log_eval_samples",
        lambda *args: pytest.fail("Generated samples should stay local"),
    )
    logger.post_init({
        "results": {"gsm8k_cot": {"exact_match,strict-match": 0.5}},
        "configs": {}, "config": {"limit": 300},
    })
    logger.log_eval_result()
    logger.log_eval_samples({"gsm8k_cot": [{"resps": ["answer"]}]})
    assert logger.run.records == [{"gsm8k_cot/exact_match,strict-match": 0.5}]
    assert logger.run.config["cli_configs"]["limit"] == 300


@pytest.mark.parametrize("rank,enabled", [(0, True), (1, True), (0, False)])
def test_cli_uses_one_wandb_logger_and_preserves_limit(monkeypatch, rank, enabled):
    import lm_eval.loggers

    original_logger = lm_eval.loggers.WandbLogger
    monkeypatch.setenv("RANK", str(rank))
    args = SimpleNamespace(
        wandb_args="project=test" if enabled else "",
        wandb_config_args="", limit=300,
    )

    def fake_cli(received):
        assert received.limit == 300
        should_log = rank == 0 and enabled
        assert bool(received.wandb_args) == should_log
        assert lm_eval.loggers.WandbLogger is (
            MinimalWandbLogger if should_log else original_logger
        )
        return "evaluated"

    monkeypatch.setattr(ensemble_eval, "cli_evaluate", fake_cli)
    monkeypatch.setattr(wandb, "run", None)
    assert ensemble_eval.main(args) == "evaluated"
    assert lm_eval.loggers.WandbLogger is original_logger


@pytest.mark.parametrize("error,exit_code", [(RuntimeError("evaluation failed"), 1),
                                            (SystemExit(0), 0)])
def test_cli_closes_run_and_restores_native_logger(monkeypatch, error, exit_code):
    import lm_eval.loggers

    run = FakeRun()
    original_logger = lm_eval.loggers.WandbLogger
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(wandb, "run", run)

    def fail(args):
        raise error

    monkeypatch.setattr(ensemble_eval, "cli_evaluate", fail)
    with pytest.raises(type(error)):
        ensemble_eval.main(SimpleNamespace(wandb_args="project=test"))
    assert run.exit_codes == [exit_code]
    assert lm_eval.loggers.WandbLogger is original_logger
