"""Check generic task coverage, selector replay, compact artifacts, and W&B.

After sourcing /home/sarthak.malla/.zshrc and activating dllm, run on compute:
    srun -p "$PARTITION" -q "$QUOTATYPE" --cpus-per-task=2 --time=00:20:00 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_first_action_benchmark.py -q
Tests use synthetic tasks and a tiny CPU model; no checkpoint is loaded.
"""

from dataclasses import asdict, replace
import json
from statistics import mean
from types import SimpleNamespace

import pytest
import torch

from examples.path_selection.experiments import runner, launch
from examples.path_selection.experiments import benchmark as benchmark_module
from examples.path_selection.experiments import diagnostics as diagnostic_module
from examples.path_selection.experiments.accounting import ForwardAccounting
from examples.path_selection.experiments.analyze import build_report
from examples.path_selection.experiments.artifacts import RunStore
from examples.path_selection.experiments.configs import benchmark_configs, benchmark_selector_schedule
from examples.path_selection.experiments.telemetry import ExperimentLogger
from dllm.core.samplers.entropy_drop import EntropyDropSampler
# conftest.py exposes sibling tests; lm-eval owns the top-level scripts package.
from test_dependency_guided_decoder import _make_tiny_llada, _tokenizer


@pytest.mark.parametrize("tokens", [[126081], []])
def test_generation_token_suppression_configuration(tmp_path, tokens):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"task": "synthetic", "primary_metric": "score",
                                "primary_filter": "filter",
                                "generation": {"suppress_tokens": tokens, "begin_suppress_tokens": []}}))
    settings = benchmark_module.load_configuration(path)
    config = replace(benchmark_configs()["reference_cheap"], **settings["generation"])
    assert config.suppress_tokens == tokens
    assert config.begin_suppress_tokens == []


@pytest.mark.parametrize("tokens", [[-1], [True], "[126081]"])
def test_invalid_generation_token_suppression(tmp_path, tokens):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"task": "synthetic", "primary_metric": "score",
                                "primary_filter": "filter", "generation": {"suppress_tokens": tokens}}))
    with pytest.raises(ValueError, match="nonnegative token IDs"):
        benchmark_module.load_configuration(path)


@pytest.mark.parametrize("size", [7, 12])
def test_runtime_task_size_and_disjoint_shards(size):
    class Task:
        OUTPUT_TYPE = "generate_until"
        config = SimpleNamespace(repeats=1, num_fewshot=0)
        eval_docs = range(size)

        def set_fewshot_seed(self, **kwargs):
            pass

        def aggregation(self):
            return {"score": mean}

        def build_all_requests(self, **kwargs):
            assert kwargs["limit"] is None
            assert kwargs["cache_requests"] is False
            self.instances = [SimpleNamespace(doc_id=doc) for doc in self.eval_docs]

    harness = SimpleNamespace(apply_chat_template=lambda messages: "prompt", tokenizer_name="synthetic")
    ids, requests = benchmark_module.task_requests(Task(), harness, {})
    assert set(requests) == set(range(size))
    left = set(runner.assigned_document_ids(0, 2, ids))
    right = set(runner.assigned_document_ids(1, 2, ids))
    assert not left & right
    assert left | right == set(range(size))
    assert len(left) == (size + 1) // 2
    assert len(right) == size // 2


def test_dry_run_does_not_load_task_or_create_artifacts(tmp_path, monkeypatch, capsys):
    config = tmp_path / "task.json"
    settings = {"task": "arbitrary_task", "primary_metric": "score", "primary_filter": "filter"}
    config.write_text(json.dumps(settings))
    output = tmp_path / "absent"
    monkeypatch.setattr("sys.argv", ["benchmark.py", "--config", str(config), "--output-root", str(output),
                                     "--arm", "first_action_entropy", "--doc-stop", "3", "--dry-run"])
    monkeypatch.setattr(benchmark_module, "BenchmarkContext", lambda *args: pytest.fail("Loaded a task"))
    benchmark_module.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["settings"] == settings
    assert preview["document_work_range"] == [0, 3]
    assert not output.exists()


def test_launcher_routes_generic_worker_and_logging(tmp_path):
    config = tmp_path / "task.json"
    args = SimpleNamespace(output_root=tmp_path, benchmark_config=config, wandb_mode="online",
                           wandb_project="test", wandb_group="shared",
                           runner_args=["--resume", "--arm", "first_action_entropy"])
    for index, command in enumerate(launch.worker_commands(args)):
        assert str(launch.BENCHMARK_RUNNER) in command
        assert command[command.index("--config") + 1] == str(config)
        assert command[command.index("--device") + 1] == f"cuda:{index}"
        assert command[command.index("--wandb-mode") + 1] == "online"


@pytest.mark.parametrize("filtered", ["answer", ["function body"]])
def test_scoring_preserves_task_filter_shapes_and_native_metrics(filtered, monkeypatch):
    import dllm.utils

    context = benchmark_module.BenchmarkContext.__new__(benchmark_module.BenchmarkContext)
    context.settings = {"primary_metric": "score", "primary_filter": "native"}
    context.prompts = {0: torch.tensor([3, 4])}
    context.harness = SimpleNamespace(tokenizer=object())
    context.requests = {0: SimpleNamespace(doc={}, args=("prompt", {"until": ["STOP"]}))}

    class Filter:
        def apply(self, requests):
            assert requests[0].resps == ["answer"]
            requests[0].filtered_resps["native"] = filtered

    def process(doc, results):
        assert results == [filtered]
        return {"score": 1.0, "secondary": 0.25}

    context.task = SimpleNamespace(_filters=[Filter()], process_results=process)
    monkeypatch.setattr(dllm.utils, "sample_trim", lambda *args: ["answerSTOPsuffix"])
    output = context.score_output(0, torch.tensor([[3, 4, 5]]))
    assert output == {"response": "answer", "primary_correct": True,
                      "metric_scores": {"score,native": 1.0, "secondary,native": 0.25}}


@pytest.mark.parametrize("arm", ["first_action_entropy", "reference_cheap", "reference_entropy"])
def test_real_benchmark_replay_accounting_and_compact_resume(arm, tmp_path, monkeypatch):
    model = _make_tiny_llada()
    sampler = EntropyDropSampler(model=model, tokenizer=_tokenizer())
    configs = {name: replace(config, max_length=16, max_new_tokens=8, block_size=4, steps=4,
                             dependency_last_n_layers=2, dependency_sink_filter_enabled=False,
                             suppress_tokens=[31])
               for name, config in benchmark_configs().items() if name in benchmark_module.ARMS}
    assert asdict(configs["first_action_entropy"]) == asdict(configs["reference_cheap"])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(diagnostic_module, "continuation_config", lambda: configs["reference_cheap"])

    def score(doc_id, sequences):
        return {"generated_token_ids": sequences[0, 2:].tolist(), "primary_correct": True,
                "metric_scores": {"score,native": 1.0}}

    context = SimpleNamespace(harness=SimpleNamespace(sampler=sampler, device=torch.device("cpu")),
                              prompts={0: torch.tensor([3, 4]), 1: torch.tensor([3, 4])}, score_output=score)
    store = RunStore(tmp_path / "benchmark", {"test": "first-action", "arm": arm})
    selected_steps = []
    original_select = sampler.select_step

    def observe_selection(prepared, selector=None):
        selected_steps.append((prepared.state.global_step_index, prepared.state.block_index,
                               selector, prepared.candidates.shape[0]))
        return original_select(prepared, selector=selector)

    monkeypatch.setattr(sampler, "select_step", observe_selection)
    with torch.no_grad(), ForwardAccounting(model) as accounting:
        runner.benchmark(context, store, accounting, [0], arm, expected_document_ids=(0, 1),
                         configuration=configs[arm], save_actions=False)
        assert not store.has("completed", f"benchmark_{arm}")
        first = store.get_json("benchmarks", f"{arm}-doc00000")
        steps = list(selected_steps)
        assert {block for _, block, _, _ in steps} == {0, 1}
        expected = ["entropy_drop" if arm == "reference_entropy" or
                    (arm == "first_action_entropy" and index == 0) else "max_confidence"
                    for index, _, _, _ in steps]
        assert [selector for _, _, selector, _ in steps] == expected
        extra_rows = sum(count for (_, _, selector, count) in steps if selector == "entropy_drop")
        assert first["accounting"]["evaluated_rows"] == len(steps) + extra_rows
        assert first["accounting"]["model_calls"] == len(steps) + extra_rows
        assert first["selector_schedule"] == benchmark_selector_schedule(arm)
        assert first["committed_positions"] == 8
        assert first["action_count"] == len(steps)
        assert "actions" not in first
        if arm == "first_action_entropy":
            assert first["selector_counts"]["entropy_drop"] == 1
        selected_steps.clear()
        runner.benchmark(context, store, accounting, [0, 1], arm, expected_document_ids=(0, 1),
                         configuration=configs[arm], save_actions=False)
        assert selected_steps == steps
        second = store.get_json("benchmarks", f"{arm}-doc00001")
        assert first["generated_token_ids"] == second["generated_token_ids"]
        assert accounting.snapshot()["model_calls"] == 2 * first["accounting"]["model_calls"]
        assert store.has("completed", f"benchmark_{arm}")
        assert not (store.root / "branches").exists()
        assert not (store.root / "diagnostics").exists()
        runner.reset_generation_rng()
        if arm == "first_action_entropy":
            reference = replace(configs["reference_entropy"], diagnostic_metadata=True, return_dict=True)
            state = sampler.initialize_state([context.prompts[0]], reference)
            prepared = sampler.prepare_step(state)
            choice = sampler.select_step(prepared, selector="entropy_drop")
            diagnostic_store = RunStore(tmp_path / "diagnostic", {"test": "initial-replay"})
            diagnostic = diagnostic_module.DiagnosticRunner(sampler, diagnostic_store, accounting, score)
            key = diagnostic.branch(prepared, int(choice.best_index[0]), state_key="initial", doc_id=0)
            replay = diagnostic_store.get_json("branches", key)["generated_token_ids"]
        else:
            state = sampler.initialize_state([context.prompts[0]], configs[arm])
            replay = sampler.continue_from_state(state).sequences[0, 2:].tolist()
        assert first["generated_token_ids"] == replay


def test_generic_report_and_wandb_metrics_preserve_task_labels(tmp_path):
    manifest = {"kind": "policy_benchmark", "task": "synthetic", "primary_metric": "score",
                "primary_filter": "native", "document_ids": [0, 1, 2, 3],
                "settings": {"report_split_at": 2}}
    store = RunStore(tmp_path / "run", manifest)
    logger = ExperimentLogger(store.root, manifest, "benchmark_first_action_entropy", mode="disabled")
    for arm in ("reference_cheap", "first_action_entropy"):
        for doc_id in manifest["document_ids"]:
            correct = (doc_id < 2) == (arm == "first_action_entropy")
            record = {"arm": arm, "doc_id": doc_id, "primary_correct": correct,
                      "metric_scores": {"score,native": float(correct)},
                      "accounting": {"evaluated_rows": 10, "model_calls": 10}, "generation_seconds": 1,
                      "action_count": 3, "committed_positions": 8,
                      "selector_counts": {"entropy_drop": 1, "max_confidence": 2}}
            key = f"{arm}-{doc_id}"
            store.put_json("benchmarks", key, record)
            logger._add("benchmarks", record, key)
    metrics = logger._summary()
    assert metrics["accuracy/primary_rate"] == 0.5
    assert metrics["task_metrics/score,native/mean"] == 0.5
    assert metrics["actions/mean_size"] == 8 / 3
    report = build_report(store.root)
    comparison = "first_action_entropy-minus-reference_cheap"
    assert report["paired_benchmarks"]["all"][comparison]["difference_pp"] == 0
    assert report["paired_benchmarks"]["prefix"][comparison]["difference_pp"] == 100
    assert report["paired_benchmarks"]["remainder"][comparison]["difference_pp"] == -100
    assert report["benchmarks"]["remainder"]["first_action_entropy"]["complete"]
    assert (store.root / "analysis" / "metrics.csv").is_file()


def test_resume_rejects_changed_selector_schedule(tmp_path):
    manifest = {"document_ids": [0], "schedule": benchmark_selector_schedule("first_action_entropy")}
    RunStore(tmp_path / "run", manifest)
    with pytest.raises(ValueError, match="Resume rejected"):
        RunStore(tmp_path / "run", {**manifest, "schedule": benchmark_selector_schedule("reference_cheap")}, resume=True)


def test_compact_wandb_state_rebuilds_native_metrics_on_resume(tmp_path, monkeypatch):
    from test_experiment_telemetry import _client

    calls, runs = _client(monkeypatch)
    configuration = {"task": "synthetic", "document_ids": [0],
                     "primary_metric": "score", "primary_filter": "native"}
    store = RunStore(tmp_path, configuration)
    record = {"arm": "first_action_entropy", "doc_id": 0, "primary_correct": True,
              "metric_scores": {"score,native": 1.0}, "response": "PRIVATE RESPONSE",
              "generation_seconds": 1, "action_count": 2, "committed_positions": 4}
    key = "first_action_entropy-doc0"
    stage = "benchmark_first_action_entropy"
    with ExperimentLogger(tmp_path, configuration, stage, mode="online", compact_state=True) as logger:
        store.put_json("benchmarks", key, record)
        logger.log_unit("benchmarks", record, unit_id=key)
        path = logger.path
    assert "units" not in json.loads(path.read_text())
    with ExperimentLogger(tmp_path, configuration, stage, mode="online", compact_state=True) as logger:
        assert not logger.log_unit("benchmarks", record, unit_id=key)
    assert calls[0]["id"] == calls[1]["id"]
    assert runs[1].summary["accuracy/primary_rate"] == 1
    assert runs[1].summary["task_metrics/score,native/mean"] == 1
    assert runs[1].summary["progress/documents"] == 1
    assert "PRIVATE" not in json.dumps(runs[1].logs)
