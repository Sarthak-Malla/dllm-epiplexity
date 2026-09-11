"""Check fixed-size ablation comparability using only synthetic artifacts.

Run on a Slurm compute node after preparing the environment:
    if [ -f /home/sarthak.malla/.zshrc ]; then
        source /home/sarthak.malla/.zshrc
    else
        source /apps/local/conda_init.sh
    fi
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=1 --time=00:10:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/test_ablation.py

These standard-library tests do not load a model or launch an evaluation.
"""

from __future__ import annotations

import importlib.util
import io
from itertools import combinations, islice
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


DIRECTORY = Path(__file__).resolve().parent


def load_local_module(name: str, filename: str):
    """Import an ablation helper without invoking its command-line entrypoint."""
    specification = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {DIRECTORY / filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


launcher = load_local_module("fixed_search_launcher_tests", "run_ablation.py")
analysis = load_local_module("fixed_search_analysis_tests", "analyze_results.py")


def launcher_arguments(output_root: Path, *, dry_run: bool) -> SimpleNamespace:
    """Provide one small evaluation request to the launcher's public entrypoint."""
    return SimpleNamespace(
        arm="D4", run_tag="synthetic", output_root=output_root,
        generation_seed=42, limit=2, wandb_mode="disabled", dry_run=dry_run,
    )


class LauncherComparabilityTests(unittest.TestCase):
    """Protect the experimental controls and safe preview behavior."""

    def test_only_declared_settings_change_between_arms(self):
        settings = {arm: launcher.model_arguments(arm, 42) for arm in launcher.ARMS}
        reference = settings["D4"]
        varying = {
            key
            for arm_settings in settings.values()
            for key in reference.keys() | arm_settings.keys()
            if reference.get(key) != arm_settings.get(key)
        }
        self.assertEqual(varying, {"dependency_parallel_variant", "candidate_budget", "dependency_confidence_exponent",
                                   "dependency_seed_strategy", "dependency_seed_entropy_weight"})
        self.assertEqual(launcher.ARMS, analysis.ARMS)
        self.assertEqual(launcher.SEED_SETTINGS, analysis.SEED_SETTINGS)
        for arm in launcher.SEED_SETTINGS:
            baseline = settings[f"CD{arm[-1]}"]
            differences = {key for key in baseline if settings[arm][key] != baseline[key]}
            expected = {"dependency_seed_strategy"}
            if arm.startswith("IE"):
                expected.add("dependency_seed_entropy_weight")
            self.assertEqual(differences, expected)
        for dependency, confidence_dependency in (("D4", "CD4"), ("D8", "CD8")):
            changes = {key for key in reference if settings[dependency][key] != settings[confidence_dependency][key]}
            self.assertEqual(changes, {"dependency_confidence_exponent"})
            self.assertEqual(settings[dependency]["dependency_confidence_exponent"], "0.0")
            self.assertEqual(settings[confidence_dependency]["dependency_confidence_exponent"], "1.0")
        self.assertEqual(settings["D4"]["candidate_budget"], settings["C4"]["candidate_budget"])
        self.assertEqual(settings["D8"]["candidate_budget"], settings["C8"]["candidate_budget"])
        commit_k = int(reference["dependency_commit_k"])
        self.assertEqual(commit_k, 4)
        self.assertEqual(int(reference["steps"]) * commit_k, int(reference["max_new_tokens"]))
        self.assertEqual(int(reference["block_size"]) % commit_k, 0)
        self.assertEqual(reference["dependency_cardinality_strategy"], "fixed")
        self.assertEqual(reference["diagnostic_retention"], "full")
        self.assertEqual(reference["temperature"], "0.0")
        self.assertEqual(reference["candidate_chunk_size"], "1")
        # A numeric-looking action-size string is converted to int by lm-eval.
        self.assertIn("|", reference["dependency_action_sizes"])

    def test_dry_run_neither_writes_nor_launches(self):
        arguments = launcher_arguments(Path("/tmp/fixed-search-preview"), dry_run=True)
        with (
            mock.patch.object(launcher, "parse_args", return_value=arguments),
            mock.patch.object(launcher, "source_hashes", return_value={}),
            mock.patch.object(Path, "mkdir", side_effect=AssertionError("preview created a directory")),
            mock.patch.object(Path, "write_text", side_effect=AssertionError("preview wrote a file")),
            mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("preview launched evaluation")),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            launcher.main()
        manifest, _ = json.JSONDecoder().raw_decode(output.getvalue())
        command = manifest["command"]
        self.assertEqual(manifest["world_size"], 2)
        self.assertIn("accelerate.commands.launch", command)
        self.assertEqual(command[command.index("--num_processes") + 1], "2")
        self.assertEqual(command[command.index("--num_machines") + 1], "1")
        self.assertEqual(command[command.index("--main_process_port") + 1], "0")
        self.assertEqual(command.count(str(launcher.ROOT / "examples/path_selection/eval.py")), 1)

    def test_default_subset_splits_evenly_and_odd_limits_are_rejected(self):
        arguments = [str(DIRECTORY / "run_ablation.py"), "--arm", "D4"]
        with mock.patch.object(sys, "argv", arguments):
            self.assertEqual(launcher.parse_args().limit, 300)
        with (
            mock.patch.object(sys, "argv", [*arguments, "--limit", "301"]),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit):
                launcher.parse_args()

    def test_existing_run_is_preserved_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            arguments = launcher_arguments(output_root, dry_run=False)
            run_directory = output_root / "synthetic/gsm8k_cot/limit2/seed42/D4"
            run_directory.mkdir(parents=True)
            sentinel = run_directory / "manifest.json"
            sentinel.write_text("existing experiment\n")
            with (
                mock.patch.object(launcher, "parse_args", return_value=arguments),
                mock.patch.object(launcher, "source_hashes", return_value={}),
                mock.patch.object(launcher, "CHECKPOINT", output_root),
                mock.patch.dict(launcher.os.environ, {"SLURM_JOB_ID": "123", "SLURM_NTASKS": "1", "WORLD_SIZE": "1"}),
                mock.patch.object(launcher.subprocess, "run") as evaluate,
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                with self.assertRaises(FileExistsError):
                    launcher.main()
                evaluate.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing experiment\n")


def write_json(path: Path, value) -> None:
    """Write a synthetic artifact in the isolated temporary fixture."""
    path.write_text(json.dumps(value) + "\n")


def synthetic_trace(arm: str, example_index: int) -> dict:
    """Build a complete four-token trace with known overlap and call counts."""
    variant, budget, confidence_exponent = launcher.ARMS[arm]
    seed_strategy, seed_entropy_weight = launcher.SEED_SETTINGS.get(arm, ("legacy", 0.0))
    response_start = 100 + example_index * 10
    steps = []
    for index in range(64):
        block = index // 16
        first_remaining = response_start + 4 * index
        block_end = response_start + (block + 1) * 64
        eligible = block_end - first_remaining
        count = min(budget, math.comb(eligible, 4))
        candidates = []
        for candidate_index, positions in enumerate(islice(combinations(range(first_remaining, block_end), 4), count)):
            candidates.append({
                "index": candidate_index, "name": f"candidate_{candidate_index}",
                "valid": True, "positions": list(positions), "action_size": 4,
                "immediate_action_cost": 1.0,
                "verifier_score": -float(candidate_index),
                "raw_verifier_score": -4.0 * candidate_index,
                "fallback": False, "fallback_source": None, "stopping_reason": None,
                "mean_within_set_conflict": 0.1, "max_within_set_conflict": 0.2,
            })
        steps.append({
            # Existing single-example telemetry shadows block_index with batch row 0.
            "global_step_index": index, "block_index": 0, "step_index": index % 16,
            "remaining_response_masks": 256 - 4 * index, "response_tokens": 256,
            "proposal_strategy": "dependency", "cardinality_strategy": "fixed", "commit_k": 4,
            "dependency_parallel_variant": variant, "candidate_budget_requested": budget,
            "dependency_confidence_exponent": confidence_exponent,
            "dependency_seed_strategy": seed_strategy, "dependency_seed_entropy_weight": seed_entropy_weight,
            "candidate_action_space_size": eligible, "candidate_action_set_count": math.comb(eligible, 4),
            "candidate_count_realized": count, "candidate_collapse": False,
            "candidates": candidates, "selected_candidate": dict(candidates[0]),
            "captured_base_forward_count": 1, "lookahead_model_calls": count,
            "timing_seconds": {"base_forward": 0.001, "candidate_lookahead": 0.002},
        })
    return {"example_index": example_index, "steps": steps}


def diagnostic_path(directory: Path, rank: int = 0, world_size: int = 2) -> Path:
    """Resolve a synthetic rank's diagnostics using the evaluator naming scheme."""
    suffix = "" if world_size == 1 else f"_rank{rank:05d}-of-{world_size:05d}"
    return directory / f"{analysis.PREFIX}_diagnostics{suffix}.json"


def write_synthetic_runs(run_root: Path, world_size: int = 2, arms=analysis.DEFAULT_ARMS) -> None:
    """Create comparable runs with two paired documents and full rank telemetry."""
    outcomes = {"D4": (1, 0), "D8": (1, 1), "C4": (0, 0), "C8": (1, 0),
                "CD4": (1, 1), "CD8": (1, 1), "I4": (1, 0), "I8": (1, 1),
                "IE4": (1, 1), "IE8": (1, 0), "CS4": (0, 0), "CS8": (1, 0)}
    for arm in arms:
        directory = run_root / arm
        directory.mkdir()
        settings = launcher.model_arguments(arm, 42)
        normalized = analysis.model_arguments(settings)
        write_json(directory / "manifest.json", {
            "arm": arm, "task": "gsm8k_cot", "num_fewshot": 5,
            "evaluation_seed": launcher.EVALUATION_SEED, "generation_seed": 42,
            "limit": 2, "checkpoint": settings["pretrained"], "model_args": settings,
            "world_size": world_size, "response_cache": False,
            "source_hashes": {analysis.LAUNCHER_SOURCE: "synthetic-identical-launcher",
                              str(launcher.ROOT / "dllm/core/samplers/dependency.py"): "synthetic-identical-sampler",
                              str(launcher.ROOT / "examples/path_selection/eval.py"): "synthetic-identical-evaluator"},
        })
        write_json(directory / "completed.json", {"arm": arm, "returncode": 0, "wall_seconds": 3.0})
        accuracy = sum(outcomes[arm]) / 2
        result = {
            "config": {
                "model_args": ",".join(f"{key}={value}" for key, value in settings.items()),
                "batch_size": 1, "use_cache": None, "limit": 2,
                "random_seed": 0, "numpy_seed": 1234, "torch_seed": 1234, "fewshot_seed": 1234,
            },
            "configs": {"gsm8k_cot": {"metadata": {**normalized, "version": "synthetic"}}},
            "n-shot": {"gsm8k_cot": 5},
            "n-samples": {"gsm8k_cot": {"original": 1319, "effective": 2}},
            "results": {"gsm8k_cot": {f"exact_match,{name}": accuracy for name in analysis.FILTERS}},
        }
        write_json(directory / "results_synthetic.json", result)
        samples = []
        for filter_name in analysis.FILTERS:
            for doc_id, correct in enumerate(outcomes[arm]):
                samples.append({
                    "doc_id": doc_id, "filter": filter_name, "exact_match": correct,
                    "doc": {"question": f"Question {doc_id}", "answer": str(doc_id)},
                    "arguments": {"gen_args_0": {"arg_0": f"Prompt {doc_id}"}},
                    "target": str(doc_id), "doc_hash": f"document-{doc_id}",
                    "prompt_hash": f"prompt-{doc_id}", "target_hash": f"target-{doc_id}",
                    "resps": [[f"Arm {arm} response {doc_id}"]],
                })
        (directory / "samples_gsm8k_cot_synthetic.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples)
        )
        if world_size == 1:
            write_json(directory / f"{analysis.PREFIX}_runtime.json", {
                "sampler_type": "entropy_drop", "rank": 0, "world_size": 1,
                "generation_total_seconds": 2.0, "generation_batch_seconds": [1.0, 1.0],
                "generation_batch_count": 2,
            })
            write_json(diagnostic_path(directory, world_size=1), [
                synthetic_trace(arm, example_index) for example_index in range(2)
            ])
        else:
            rank_runtime_paths = []
            for rank in range(world_size):
                rank_runtime_path = directory / f"{analysis.PREFIX}_runtime_rank{rank:05d}-of-{world_size:05d}.json"
                rank_runtime_paths.append(str(rank_runtime_path))
                seconds = float(rank + 1)
                write_json(rank_runtime_path, {
                    "sampler_type": "entropy_drop", "rank": rank, "world_size": world_size,
                    "generation_total_seconds": seconds, "generation_batch_seconds": [seconds],
                    "generation_batch_count": 1,
                })
                trace = synthetic_trace(arm, rank)
                trace["example_index"] = 0
                write_json(diagnostic_path(directory, rank, world_size), [trace])
            write_json(directory / f"{analysis.PREFIX}_diagnostics_manifest.json", {
                "schema_version": 1, "distributed": True, "world_size": world_size,
                "shards": [str(diagnostic_path(directory, rank, world_size)) for rank in range(world_size)],
            })
            write_json(directory / f"{analysis.PREFIX}_runtime.json", {
                "sampler_type": "entropy_drop", "distributed": True, "world_size": world_size,
                "generation_total_seconds": 2.0, "generation_work_seconds": 3.0,
                "generation_rank_seconds": [1.0, 2.0], "generation_batch_count": 1,
                "rank_runtime_paths": rank_runtime_paths,
            })


class AnalysisComparabilityTests(unittest.TestCase):
    """Exercise paired conclusions against complete and corrupted experiment traces."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_root = Path(self.temporary.name)
        write_synthetic_runs(self.run_root)

    def test_complete_paired_runs_reproduce_known_effects_and_search_geometry(self):
        summary = analysis.analyze(self.run_root)
        self.assertEqual(tuple(summary["runs"]), analysis.DEFAULT_ARMS)
        d4 = summary["runs"]["D4"]["diagnostics"]
        d8 = summary["runs"]["D8"]["diagnostics"]
        self.assertEqual(d4["model_calls_per_example"]["mean"], 308)
        self.assertEqual(d8["model_calls_per_example"]["mean"], 548)
        self.assertEqual(d4["document_base_model_calls"], 128)
        self.assertEqual(d4["document_lookahead_model_calls"], 488)
        self.assertEqual(d4["actual_model_calls"], 616)
        self.assertEqual(d8["actual_model_calls"], 1096)
        self.assertEqual(d4["world_size"], 2)
        self.assertEqual([row["document_ids"] for row in d4["rank_metrics"]], [[0], [1]])
        self.assertEqual([row["model_calls"] for row in d4["rank_metrics"]], [308, 308])
        self.assertEqual(summary["runs"]["D4"]["generation_seconds"], 2.0)
        self.assertEqual(summary["runs"]["D4"]["generation_work_seconds"], 3.0)
        self.assertEqual(summary["runs"]["D4"]["generation_rank_seconds"], [1.0, 2.0])
        # All nonfinal D4 pools share three positions; forced final pools have no pairs.
        self.assertEqual(d4["pairwise_jaccard"]["count"], 720)
        self.assertAlmostEqual(d4["pairwise_jaccard"]["mean"], 0.6)
        self.assertEqual(d4["pairwise_replaced_positions"]["mean"], 1)
        self.assertEqual(d4["pairwise_symmetric_difference"]["mean"], 2)
        pair = next(row for row in summary["paired_comparisons"]
                    if row["challenger"] == "D8" and row["reference"] == "D4"
                    and row["filter"] == "strict-match")
        self.assertEqual((pair["wins"], pair["losses"], pair["ties"]), (1, 0, 1))
        self.assertEqual(pair["difference_pp"], 50.0)
        self.assertEqual(summary["descriptive_difference_in_candidate_count_gains_pp"]["strict-match"], 0.0)

    def test_cd_comparison_with_historical_arm_requires_explicit_launcher_exception(self):
        extension = self.run_root / "new_run_tag"
        extension.mkdir()
        write_synthetic_runs(extension, arms=("CD4", "CD8"))
        for arm in ("CD4", "CD8"):
            path = extension / arm / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["source_hashes"][analysis.LAUNCHER_SOURCE] = "new-launcher-with-cd-arms"
            write_json(path, manifest)
        arms = ("D4", "CD4", "CD8")
        arm_dirs = (f"D4={self.run_root / 'D4'}",)
        with self.assertRaisesRegex(ValueError, "mismatch in manifest"):
            analysis.analyze(extension, arms, arm_dirs=arm_dirs)
        summary = analysis.analyze(extension, arms, arm_dirs=arm_dirs, allow_launcher_hash_change=True)
        self.assertEqual(tuple(summary["runs"]), arms)
        self.assertEqual(summary["runs"]["D4"]["directory"], str(self.run_root / "D4"))
        self.assertTrue(summary["allow_launcher_hash_change"])
        self.assertEqual(summary["runs"]["CD4"]["source_hashes"][analysis.LAUNCHER_SOURCE], "new-launcher-with-cd-arms")
        pairs = {(pair["challenger"], pair["reference"]) for pair in summary["paired_comparisons"]}
        self.assertEqual(pairs, {("CD4", "D4"), ("CD8", "CD4")})
        self.assertEqual(summary["descriptive_difference_in_candidate_count_gains_pp"], {})
        output = extension / "analysis"
        analysis.write_outputs(summary, output)
        report = (output / "report.md").read_text()
        self.assertIn("across D4, CD4, CD8.", report)
        self.assertIn("launcher hash was explicitly exempted", report)

    def test_launcher_exception_never_permits_sampler_or_evaluator_changes(self):
        path = self.run_root / "D8/manifest.json"
        original = json.loads(path.read_text())
        for source in ("dllm/core/samplers/dependency.py", "examples/path_selection/eval.py"):
            with self.subTest(source=source):
                manifest = json.loads(json.dumps(original))
                manifest["source_hashes"][str(launcher.ROOT / source)] = "changed-implementation"
                write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, "mismatch in manifest"):
                    analysis.analyze(self.run_root, allow_launcher_hash_change=True)

    def test_declared_cd_exponent_is_required_even_when_artifacts_agree(self):
        write_synthetic_runs(self.run_root, arms=("CD4",))
        directory = self.run_root / "CD4"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["model_args"]["dependency_confidence_exponent"] = "0.5"
        write_json(manifest_path, manifest)
        result_path = directory / "results_synthetic.json"
        result = json.loads(result_path.read_text())
        result["config"]["model_args"] = manifest["model_args"]
        result["configs"]["gsm8k_cot"]["metadata"]["dependency_confidence_exponent"] = 0.5
        write_json(result_path, result)
        with self.assertRaisesRegex(ValueError, "Unexpected dependency_confidence_exponent"):
            analysis.analyze(self.run_root, ("D4", "CD4"))

    def test_historical_directory_overrides_are_explicit_and_unambiguous(self):
        arms = ("D4", "CD4")
        invalid = (("D4=relative/path",), ("C4=/tmp/c4",), ("D4=/tmp/d4", "D4=/tmp/another"))
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                analysis.resolve_arm_directories(self.run_root, arms, overrides)

    def test_diagnostic_exponent_must_match_the_cd_configuration(self):
        write_synthetic_runs(self.run_root, arms=("CD4",))
        path = diagnostic_path(self.run_root / "CD4")
        rows = json.loads(path.read_text())
        rows[0]["steps"][0]["dependency_confidence_exponent"] = 0.0
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "Diagnostic confidence exponent mismatch"):
            analysis.analyze(self.run_root, ("D4", "CD4"))

    def test_changed_prompt_rejects_document_pairing(self):
        path = self.run_root / "D8/samples_gsm8k_cot_synthetic.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["doc_id"] == 0:
                row["arguments"] = {"gen_args_0": {"arg_0": "A different prompt"}}
                row["prompt_hash"] = "changed-prompt"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "arguments mismatch"):
            analysis.analyze(self.run_root)

    def test_seed_arms_have_paired_comparisons_and_fixed_companion_settings(self):
        arms = tuple(launcher.SEED_SETTINGS)
        write_synthetic_runs(self.run_root, arms=arms)
        summary = analysis.analyze(self.run_root, arms)
        pairs = {(row["challenger"], row["reference"]) for row in summary["paired_comparisons"]}
        self.assertEqual(pairs, {("I4", "CS4"), ("I8", "CS8"), ("IE4", "I4"), ("IE8", "I8"),
                                 ("IE4", "CS4"), ("IE8", "CS8"), ("I8", "I4"), ("IE8", "IE4"), ("CS8", "CS4")})
        self.assertEqual(summary["descriptive_difference_in_candidate_count_gains_pp"], {})

    def test_seed_trace_cannot_silently_use_legacy_policy(self):
        write_synthetic_runs(self.run_root, arms=("I4", "IE4"))
        path = diagnostic_path(self.run_root / "IE4")
        rows = json.loads(path.read_text())
        rows[0]["steps"][0]["dependency_seed_strategy"] = "legacy"
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "Diagnostic dependency_seed_strategy mismatch"):
            analysis.analyze(self.run_root, ("I4", "IE4"))

    def test_token_comparison_distinguishes_positions_values_and_commit_order(self):
        first = {0: {"first_proposed": ((10, 5), (11, 6)),
                     "commits": [((10, 5),), ((11, 6),)]}}
        changed_value = {0: {"first_proposed": ((10, 5), (11, 7)),
                             "commits": [((10, 8),), ((11, 6),)]}}
        result = analysis.compare_token_commits(first, changed_value)
        self.assertTrue(result["available"])
        self.assertEqual(result["first_proposed_positions_equal"], 1)
        self.assertEqual(result["first_proposed_tokens_equal"], 0)
        self.assertEqual(result["first_committed_positions_equal"], 1)
        self.assertEqual(result["first_committed_tokens_equal"], 0)
        self.assertEqual(result["final_response_token_ids_equal"], 0)
        self.assertEqual(result["matching_final_token_positions"], 1)
        reordered = {0: {"first_proposed": first[0]["first_proposed"],
                         "commits": list(reversed(first[0]["commits"]))}}
        result = analysis.compare_token_commits(first, reordered)
        self.assertEqual(result["entire_commit_trace_equal"], 0)
        self.assertEqual(result["final_response_token_ids_equal"], 1)

    def test_missing_token_ids_are_unknown_not_evidence_of_equality(self):
        record = {0: {"first_proposed": None, "commits": [None]}}
        self.assertFalse(analysis.compare_token_commits(record, record)["available"])
        summary = analysis.analyze(self.run_root)
        self.assertTrue(all(not pair["available"] for pair in summary["token_comparisons"]))

    def test_full_token_artifacts_reproduce_exact_commit_equality(self):
        for arm in analysis.DEFAULT_ARMS:
            for rank in range(2):
                path = diagnostic_path(self.run_root / arm, rank)
                rows = json.loads(path.read_text())
                for row in rows:
                    for step in row["steps"]:
                        for candidate in step["candidates"]:
                            candidate["token_ids"] = [position % 32 for position in candidate["positions"]]
                        step["selected_candidate"] = dict(step["candidates"][0])
                write_json(path, rows)
        summary = analysis.analyze(self.run_root)
        for pair in summary["token_comparisons"]:
            self.assertTrue(pair["available"])
            self.assertEqual(pair["entire_commit_trace_equal"], 2)
            self.assertEqual(pair["final_response_token_ids_equal"], 2)
        analysis.write_outputs(summary, self.run_root / "token_analysis")
        self.assertTrue((self.run_root / "token_analysis/token_comparisons.csv").is_file())
        path = diagnostic_path(self.run_root / "D8")
        rows = json.loads(path.read_text())
        rows[0]["steps"][0]["candidates"][1]["token_ids"][0] += 1
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "Candidates disagree on a shared base token prediction"):
            analysis.analyze(self.run_root)

    def test_partial_diagnostics_cannot_support_accuracy_comparison(self):
        path = diagnostic_path(self.run_root / "D4")
        rows = json.loads(path.read_text())
        rows[0]["steps"].pop()
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "Missing/partial fixed-k trace"):
            analysis.analyze(self.run_root)

    def test_internally_consistent_extra_setting_change_is_rejected(self):
        directory = self.run_root / "D8"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["model_args"]["dependency_direction"] = "incoming"
        write_json(manifest_path, manifest)
        result_path = directory / "results_synthetic.json"
        result = json.loads(result_path.read_text())
        result["config"]["model_args"] = manifest["model_args"]
        result["configs"]["gsm8k_cot"]["metadata"]["dependency_direction"] = "incoming"
        write_json(result_path, result)
        with self.assertRaisesRegex(ValueError, "mismatch in model_args"):
            analysis.analyze(self.run_root)

    def test_duplicate_sets_are_rejected_even_when_budget_count_matches(self):
        path = diagnostic_path(self.run_root / "D4")
        rows = json.loads(path.read_text())
        pool = rows[0]["steps"][0]["candidates"]
        pool[1]["positions"] = list(pool[0]["positions"])
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate position sets"):
            analysis.analyze(self.run_root)

    def test_missing_rank_diagnostics_cannot_be_treated_as_complete(self):
        diagnostic_path(self.run_root / "D4", rank=1).unlink()
        with self.assertRaises(ValueError):
            analysis.analyze(self.run_root)

    def test_repeated_rank_in_manifest_is_rejected(self):
        directory = self.run_root / "D4"
        path = directory / f"{analysis.PREFIX}_diagnostics_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["shards"][1] = manifest["shards"][0]
        write_json(path, manifest)
        with self.assertRaises(ValueError):
            analysis.analyze(self.run_root)

    def test_rank_identity_must_match_runtime_filename(self):
        path = self.run_root / "D4" / f"{analysis.PREFIX}_runtime_rank00001-of-00002.json"
        runtime = json.loads(path.read_text())
        runtime["rank"] = 0
        write_json(path, runtime)
        with self.assertRaises(ValueError):
            analysis.analyze(self.run_root)

    def test_runtime_world_size_must_match_manifest(self):
        path = self.run_root / "D4" / f"{analysis.PREFIX}_runtime.json"
        runtime = json.loads(path.read_text())
        runtime["world_size"] = 1
        write_json(path, runtime)
        with self.assertRaises(ValueError):
            analysis.analyze(self.run_root)

    def test_parallel_elapsed_time_cannot_be_reported_as_sum_of_rank_times(self):
        path = self.run_root / "D4" / f"{analysis.PREFIX}_runtime.json"
        runtime = json.loads(path.read_text())
        runtime["generation_total_seconds"] = 3.0
        write_json(path, runtime)
        with self.assertRaises(ValueError):
            analysis.analyze(self.run_root)

    def test_existing_single_gpu_artifacts_remain_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            write_synthetic_runs(run_root, world_size=1)
            summary = analysis.analyze(run_root)
        run = summary["runs"]["D4"]
        self.assertEqual(run["diagnostics"]["world_size"], 1)
        self.assertEqual(run["diagnostics"]["actual_model_calls"], 616)
        self.assertEqual(run["generation_seconds"], 2.0)
        self.assertEqual(run["generation_work_seconds"], 2.0)

    def test_exact_paired_test_handles_no_discordance_and_one_sided_wins(self):
        self.assertEqual(analysis.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(analysis.exact_mcnemar(6, 0), 0.03125)
        self.assertEqual(analysis.exact_mcnemar(6, 0), analysis.exact_mcnemar(0, 6))


if __name__ == "__main__":
    unittest.main()
