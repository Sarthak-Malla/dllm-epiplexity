"""Validate and summarize completed fixed-size candidate-search arms.

Run on a Slurm compute node after preparing the environment (never on login):
    if [ -f /home/sarthak.malla/.zshrc ]; then
        source /home/sarthak.malla/.zshrc
    else
        source /apps/local/conda_init.sh
    fi
    conda activate /home/sarthak.malla/.conda/envs/dllm
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py --run-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v1/gsm8k_cot/limit300/seed42

By default the run root must contain completed D4, D8, C4, and C8 directories.
Use --arms to select comparisons, including CD4/CD8, and repeat
--arm-dir ARM=/absolute/directory to read historical arms from other run tags.
--allow-launcher-hash-change permits only run_ablation.py's hash to differ;
all sampler/evaluation hashes and other experimental controls must still match.
--output-directory defaults to <run-root>/analysis. Only analysis outputs are
written; evaluation artifacts are read without changes. Uses the standard library.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import re


ARMS = {"D4": ("soft_full", 4, 0.0), "D8": ("soft_full", 8, 0.0),
        "C4": ("top_confidence", 4, 0.0), "C8": ("top_confidence", 8, 0.0),
        "CD4": ("soft_full", 4, 1.0), "CD8": ("soft_full", 8, 1.0),
        "I4": ("soft_full", 4, 1.0), "I8": ("soft_full", 8, 1.0),
        "IE4": ("soft_full", 4, 1.0), "IE8": ("soft_full", 8, 1.0),
        "CS4": ("soft_full", 4, 1.0), "CS8": ("soft_full", 8, 1.0)}
SEED_SETTINGS = {
    "I4": ("incoming", 0.0), "I8": ("incoming", 0.0),
    "IE4": ("incoming", 1.0), "IE8": ("incoming", 1.0),
    "CS4": ("confidence", 0.0), "CS8": ("confidence", 0.0),
}
SEED_DEFAULTS = {"dependency_seed_strategy": "legacy", "dependency_seed_entropy_weight": 0.0}
DEFAULT_ARMS = ("D4", "D8", "C4", "C8")
ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
LAUNCHER_SOURCE = str(ROOT / "examples/path_selection/ablation/7_fixed_size_candidate_search/run_ablation.py")
FILTERS = ("flexible-extract", "strict-match")
VARYING_ARGUMENTS = {"dependency_parallel_variant", "candidate_budget", "dependency_confidence_exponent", *SEED_DEFAULTS}
IDENTITY_FIELDS = ("doc", "arguments", "target", "doc_hash", "prompt_hash", "target_hash")
PREFIX = "results.json_entropy_drop"


def require(condition, message):
    """Reject incomplete or incomparable input even when Python assertions are off."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    """Read a required JSON artifact with its path in error messages."""
    require(path.is_file(), f"Missing artifact: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def single_file(directory, pattern):
    """Reject ambiguous result sets rather than selecting a latest file."""
    paths = sorted(directory.glob(pattern))
    require(len(paths) == 1, f"Expected one {pattern} in {directory}; found {len(paths)}")
    return paths[0]


def iter_json_array(path):
    """Stream one example at a time from a large JSON-array diagnostic file."""
    require(path.is_file(), f"Missing artifact: {path}")
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer = ""
        exhausted = False

        def refill():
            nonlocal buffer, exhausted
            chunk = stream.read(65536)
            exhausted = not chunk
            buffer += chunk

        def ensure_text():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not exhausted:
                refill()
                buffer = buffer.lstrip()
            require(bool(buffer), f"Truncated JSON array: {path}")

        ensure_text()
        require(buffer.startswith("["), f"Expected JSON array: {path}")
        buffer = buffer[1:]
        first = True
        while True:
            ensure_text()
            if buffer.startswith("]"):
                require(first, f"Trailing comma in JSON array: {path}")
                buffer = buffer[1:]
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if exhausted:
                        raise ValueError(f"Invalid/truncated diagnostic JSON: {path}") from error
                    refill()
            require(isinstance(value, dict), f"Expected an example object: {path}")
            buffer = buffer[end:]
            yield value
            first = False
            ensure_text()
            separator, buffer = buffer[0], buffer[1:]
            if separator == "]":
                break
            require(separator == ",", f"Invalid array separator in {path}")
        require(not (buffer + stream.read()).strip(), f"Trailing data in {path}")


def normalize_value(value):
    """Compare launcher string arguments with harness-parsed scalar arguments."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if parsed is None or isinstance(parsed, (bool, int, float)):
            return parsed
    return value


def model_arguments(raw):
    """Parse the harness argument string, preserving embedded value delimiters."""
    if isinstance(raw, str):
        items = re.split(r",(?=[A-Za-z_][A-Za-z_0-9]*=)", raw)
        pairs = [item.split("=", 1) for item in items]
        require(all(len(pair) == 2 for pair in pairs), "Malformed model_args string")
        require(len({pair[0] for pair in pairs}) == len(pairs), "Duplicate model argument")
        raw = dict(pairs)
    require(isinstance(raw, dict), "model_args must be a dictionary or argument string")
    return {key: normalize_value(value) for key, value in raw.items()}


def finite_number(value, context, *, nonnegative=False):
    """Reject missing, nonfinite, or invalid numeric telemetry."""
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"Expected a number for {context}: {value!r}")
    value = float(value)
    require(math.isfinite(value) and (not nonnegative or value >= 0),
            f"Invalid numeric value for {context}: {value}")
    return value


def distribution(values):
    """Exact empirical scalar summaries, including empty distributions."""
    counts = values if isinstance(values, Counter) else Counter(values)
    count = sum(counts.values())
    if not count:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    items = sorted(counts.items())

    def quantile(q):
        offset = (count - 1) * q
        lower, upper = math.floor(offset), math.ceil(offset)
        cumulative, low_value, high_value = 0, None, None
        for value, frequency in items:
            cumulative += frequency
            if low_value is None and cumulative > lower:
                low_value = value
            if cumulative > upper:
                high_value = value
                break
        return low_value + (high_value - low_value) * (offset - lower)

    return {"count": count, "mean": math.fsum(value * frequency for value, frequency in items) / count,
            "min": items[0][0], "p50": quantile(.5), "p90": quantile(.9), "max": items[-1][0]}


def exact_mcnemar(wins, losses):
    """Two-sided exact binomial McNemar p-value, including zero discordances."""
    discordant = wins + losses
    if not discordant:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, i)
                          for i in range(min(wins, losses) + 1)) / (2 ** discordant))


def load_samples(path, result, limit):
    """Validate per-filter outcomes, identities, and the exact evaluated prefix."""
    samples = {name: {} for name in FILTERS}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            name, doc_id = row.get("filter"), row.get("doc_id")
            require(name in samples, f"Unexpected filter at {path}:{line_number}: {name}")
            require(isinstance(doc_id, int) and not isinstance(doc_id, bool) and doc_id >= 0,
                    f"Invalid document ID at {path}:{line_number}")
            require(doc_id not in samples[name], f"Duplicate {name} document {doc_id}: {path}")
            require(all(key in row for key in IDENTITY_FIELDS), f"Missing sample identity fields: {path}")
            require(row.get("exact_match") in (0, 1), f"Nonbinary exact-match outcome: {path}")
            samples[name][doc_id] = row
    count = len(samples[FILTERS[0]])
    require(count > 0, f"No evaluation samples: {path}")
    expected = result["n-samples"]["gsm8k_cot"]
    require(expected["effective"] == count, f"Effective sample count mismatch: {path}")
    require(count == (expected["original"] if limit is None else min(limit, expected["original"])),
            f"Run is partial or has an unexpected evaluation limit: {path}")
    ids = set(range(count))
    for name in FILTERS:
        require(set(samples[name]) == ids, f"Expected document IDs 0..{count - 1} for {name}: {path}")
        reproduced = sum(row["exact_match"] for row in samples[name].values()) / count
        reported = result["results"]["gsm8k_cot"][f"exact_match,{name}"]
        require(math.isclose(reproduced, reported, rel_tol=0, abs_tol=1e-12),
                f"Reported {name} accuracy disagrees with samples: {path}")
    for doc_id in ids:
        reference = samples[FILTERS[0]][doc_id]
        for key in (*IDENTITY_FIELDS, "resps"):
            require(reference.get(key) == samples[FILTERS[1]][doc_id].get(key),
                    f"Filter records disagree on {key}, document {doc_id}: {path}")
    return samples


def diagnostic_shards(directory, runtime, manifest, document_count):
    """Validate one/two-GPU sidecars and reject incomplete or padded rank layouts."""
    world = runtime.get("world_size")
    require(isinstance(world, int) and not isinstance(world, bool) and world in (1, 2)
            and manifest.get("world_size") == world,
            f"Expected matching one- or two-GPU manifest/runtime world sizes: {directory}")
    require(document_count % world == 0,
            f"Distributed evaluation count must be divisible by world size; padding is unsupported: {directory}")
    direct = directory / f"{PREFIX}_diagnostics.json"
    shard_manifest_path = directory / f"{PREFIX}_diagnostics_manifest.json"
    if world == 1:
        require(not shard_manifest_path.exists(), f"Mixed single/distributed diagnostics: {directory}")
        require(not list(directory.glob(f"{PREFIX}_diagnostics_rank*.json")),
                f"Unexpected rank diagnostics for single-GPU run: {directory}")
        require(not list(directory.glob(f"{PREFIX}_runtime_rank*.json")),
                f"Unexpected rank runtime artifacts for single-GPU run: {directory}")
        require(runtime.get("rank") == 0 and not runtime.get("distributed", False),
                f"Incorrect single-GPU runtime identity: {directory}")
        return [(0, direct, runtime)], world

    require(not direct.exists(), f"Mixed single/distributed diagnostics: {directory}")
    require(runtime.get("distributed") is True, f"Distributed aggregate runtime is required: {directory}")
    shard_manifest = read_json(shard_manifest_path)
    require(shard_manifest.get("schema_version") == 1 and shard_manifest.get("distributed") is True
            and shard_manifest.get("world_size") == world, f"Invalid diagnostic shard manifest: {shard_manifest_path}")
    paths = [directory / f"{PREFIX}_diagnostics_rank{rank:05d}-of-{world:05d}.json" for rank in range(world)]
    require(shard_manifest.get("shards") == [str(path) for path in paths],
            f"Diagnostic shard paths/order do not match this run: {shard_manifest_path}")
    require(set(directory.glob(f"{PREFIX}_diagnostics_rank*.json")) == set(paths),
            f"Missing or extra diagnostic rank files: {directory}")
    runtime_paths = [directory / f"{PREFIX}_runtime_rank{rank:05d}-of-{world:05d}.json" for rank in range(world)]
    require(runtime.get("rank_runtime_paths") == [str(path) for path in runtime_paths],
            f"Runtime shard paths/order do not match this run: {directory}")
    require(set(directory.glob(f"{PREFIX}_runtime_rank*.json")) == set(runtime_paths),
            f"Missing or extra runtime rank files: {directory}")
    shards, rank_seconds = [], []
    for rank, (path, runtime_path) in enumerate(zip(paths, runtime_paths)):
        rank_runtime = read_json(runtime_path)
        require(rank_runtime.get("rank") == rank and rank_runtime.get("world_size") == world
                and rank_runtime.get("sampler_type") == "entropy_drop", f"Runtime rank identity mismatch: {runtime_path}")
        rank_seconds.append(finite_number(rank_runtime.get("generation_total_seconds"),
                                          "rank generation seconds", nonnegative=True))
        shards.append((rank, path, rank_runtime))
    recorded_seconds = runtime.get("generation_rank_seconds")
    require(isinstance(recorded_seconds, list) and len(recorded_seconds) == world,
            f"Missing generation times by rank: {directory}")
    require(all(math.isclose(finite_number(recorded, "recorded rank seconds", nonnegative=True), actual,
                             rel_tol=1e-9, abs_tol=1e-6)
                for recorded, actual in zip(recorded_seconds, rank_seconds)),
            f"Aggregate/per-rank generation times differ: {directory}")
    require(math.isclose(finite_number(runtime.get("generation_total_seconds"), "generation seconds", nonnegative=True),
                         max(rank_seconds), rel_tol=1e-9, abs_tol=1e-6),
            f"Generation time does not match rank maximum: {directory}")
    require(math.isclose(finite_number(runtime.get("generation_work_seconds"), "generation work seconds", nonnegative=True),
                         sum(rank_seconds), rel_tol=1e-9, abs_tol=1e-6),
            f"Generation work does not match sum of rank times: {directory}")
    require(runtime.get("generation_batch_count") == document_count // world,
            f"Aggregate batch count differs from the expected documents per rank: {directory}")
    return shards, world


def candidate_token_pairs(candidate, context):
    """Read exact position/value pairs; historical position-only records stay unknown."""
    token_ids = candidate.get("token_ids")
    if token_ids is None:
        return None
    positions = candidate["positions"]
    require(isinstance(token_ids, list) and len(token_ids) == len(positions)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_ids),
            f"Invalid position-aligned token_ids: {context}")
    return tuple(sorted(zip(positions, token_ids)))


def compare_token_commits(first, second):
    """Compare exact logged proposals, commitments, and final response IDs by document."""
    require(first.keys() == second.keys(), "Token comparison requires matching document IDs.")
    available = all(record["first_proposed"] is not None and all(step is not None for step in record["commits"])
                    for records in (first, second) for record in records.values())
    if not available:
        return {"available": False, "documents": len(first),
                "reason": "Exact per-step token IDs were not saved in one or both runs; position equality is insufficient."}
    counts = Counter()
    documents = []
    for doc_id in sorted(first):
        left, right = first[doc_id], second[doc_id]
        require(len(left["commits"]) == len(right["commits"]), "Token comparison requires matching step counts.")
        flags = {
            "first_proposed_positions_equal": tuple(p for p, _ in left["first_proposed"]) == tuple(p for p, _ in right["first_proposed"]),
            "first_proposed_tokens_equal": left["first_proposed"] == right["first_proposed"],
            "first_committed_positions_equal": tuple(p for p, _ in left["commits"][0]) == tuple(p for p, _ in right["commits"][0]),
            "first_committed_tokens_equal": left["commits"][0] == right["commits"][0],
            "entire_commit_trace_equal": left["commits"] == right["commits"],
        }
        final_left = dict(pair for step in left["commits"] for pair in step)
        final_right = dict(pair for step in right["commits"] for pair in step)
        require(final_left.keys() == final_right.keys(), "Final token positions differ between paired documents.")
        flags["final_response_token_ids_equal"] = final_left == final_right
        same_steps = sum(a == b for a, b in zip(left["commits"], right["commits"]))
        first_divergence = next((i for i, (a, b) in enumerate(zip(left["commits"], right["commits"])) if a != b), None)
        counts.update({key: int(value) for key, value in flags.items()})
        counts["matching_commit_steps"] += same_steps
        counts["total_commit_steps"] += len(left["commits"])
        counts["matching_final_token_positions"] += sum(final_left[p] == final_right[p] for p in final_left)
        counts["total_final_token_positions"] += len(final_left)
        documents.append({"doc_id": doc_id, **flags, "matching_commit_steps": same_steps,
                          "first_divergent_commit_step": first_divergence})
    return {"available": True, "documents": len(first), **dict(counts), "per_document": documents}


def analyze_diagnostics(directory, runtime, manifest, settings, document_count):
    """Validate every fixed-size trace and aggregate rank-strided document metrics."""
    shards, world = diagnostic_shards(directory, runtime, manifest, document_count)
    budget = int(settings["candidate_budget"])
    length, block_size = int(settings["max_new_tokens"]), int(settings["block_size"])
    summaries = {key: [] for key in ("candidate_entropy_cost", "selected_entropy_cost", "first_action_entropy_cost",
                                    "selected_entropy_per_token", "actions_per_example", "model_calls_per_example",
                                    "candidate_mean_conflict", "candidate_max_conflict",
                                    "selected_mean_conflict", "selected_max_conflict")}
    overlaps, changed, symmetric = Counter(), Counter(), Counter()
    realized_counts, action_sizes, fallback_sources, selected_sources = Counter(), Counter(), Counter(), Counter()
    totals, timings = Counter(), Counter()
    seen_documents, rank_metrics = set(), []
    token_records = {}
    expected_records = math.ceil(document_count / world)
    for rank, path, rank_runtime in shards:
        records = 0
        rank_base_calls, rank_lookahead_calls, rank_document_ids = 0, 0, []
        for record in iter_json_array(path):
            local_index = record.get("example_index")
            require(local_index == records, f"Nonconsecutive diagnostic example indices: {path}")
            records += 1
            # Task.doc_iterator uses create_iterator(...): islice(documents,
            # rank, limit, world_size). eval.py preserves that request order.
            doc_id = world * local_index + rank
            padding = doc_id >= document_count
            require(local_index < expected_records, f"Unexpected extra diagnostic trace: {path}")
            if padding:
                require(world > 1 and local_index == expected_records - 1,
                        f"Unexpected out-of-range trace, not recognized distributed padding: {path}")
                totals["padding_traces"] += 1
            else:
                require(doc_id not in seen_documents, f"Duplicate diagnostic document {doc_id}: {path}")
                seen_documents.add(doc_id)
                rank_document_ids.append(doc_id)
            steps = record.get("steps")
            require(isinstance(steps, list) and len(steps) == length // 4,
                    f"Missing/partial fixed-k trace for example {local_index}: {path}")
            committed, all_candidate_positions = set(), []
            commit_tokens, first_proposed_tokens = [], None
            calls = 0
            for index, step in enumerate(steps):
                context = f"{path}: example {local_index}, step {index}"
                remaining = length - 4 * index
                eligible = block_size - (4 * index) % block_size
                require(step.get("global_step_index") == index and step.get("remaining_response_masks") == remaining,
                        f"Inconsistent decoding progress: {context}")
                require(step.get("response_tokens") == length, f"Inconsistent response length: {context}")
                # Derive block membership from committed-token progress: the
                # existing sampler's block_index telemetry can be shadowed.
                derived_block = 4 * index // block_size
                require(step.get("cardinality_strategy") == "fixed" and step.get("commit_k") == 4,
                        f"Expected fixed cardinality four: {context}")
                require(step.get("dependency_parallel_variant") == settings["dependency_parallel_variant"],
                        f"Diagnostic proposal variant mismatch: {context}")
                require(step.get("dependency_confidence_exponent") == settings["dependency_confidence_exponent"],
                        f"Diagnostic confidence exponent mismatch: {context}")
                for key, default in SEED_DEFAULTS.items():
                    require(step.get(key, default) == settings.get(key, default),
                            f"Diagnostic {key} mismatch: {context}")
                require(step.get("candidate_budget_requested") == budget and step.get("candidate_action_space_size") == eligible,
                        f"Candidate budget/action-space mismatch: {context}")
                expected_count = min(budget, math.comb(eligible, 4))
                candidates = step.get("candidates")
                require(isinstance(candidates, list), f"Full candidate retention is required: {context}")
                valid = [candidate for candidate in candidates if candidate.get("valid") is True]
                require(len(valid) == expected_count and step.get("candidate_count_realized") == expected_count,
                        f"Candidate pool collapsed or count is incorrect: {context}")
                require(step.get("candidate_collapse") is False, f"Candidate collapse flag: {context}")
                sets, candidate_by_index = [], {}
                token_pairs_by_index, shared_predictions = {}, {}
                for candidate in valid:
                    positions = candidate.get("positions")
                    require(isinstance(positions, list) and len(positions) == 4
                            and all(isinstance(position, int) and not isinstance(position, bool) for position in positions),
                            f"Missing/non-four candidate positions: {context}")
                    position_set = frozenset(positions)
                    require(len(position_set) == 4 and candidate.get("action_size") == 4,
                            f"Candidate positions are duplicated or size is incorrect: {context}")
                    require(not (position_set & committed), f"Candidate includes an already committed position: {context}")
                    sets.append(position_set)
                    all_candidate_positions.append((derived_block, position_set))
                    candidate_index = candidate.get("index")
                    require(isinstance(candidate_index, int) and candidate_index not in candidate_by_index,
                            f"Invalid/duplicate candidate index: {context}")
                    candidate_by_index[candidate_index] = candidate
                    pairs = candidate_token_pairs(candidate, context)
                    token_pairs_by_index[candidate_index] = pairs
                    for position, token_id in pairs or ():
                        require(position not in shared_predictions or shared_predictions[position] == token_id,
                                f"Candidates disagree on a shared base token prediction: {context}")
                        shared_predictions[position] = token_id
                    cost = finite_number(candidate.get("immediate_action_cost"), context + " entropy cost", nonnegative=True)
                    finite_number(candidate.get("verifier_score"), context + " verifier score")
                    finite_number(candidate.get("raw_verifier_score"), context + " raw verifier score")
                    mean_conflict = finite_number(candidate.get("mean_within_set_conflict"), context + " mean conflict", nonnegative=True)
                    max_conflict = finite_number(candidate.get("max_within_set_conflict"), context + " maximum conflict", nonnegative=True)
                    if not padding:
                        summaries["candidate_entropy_cost"].append(cost)
                        summaries["candidate_mean_conflict"].append(mean_conflict)
                        summaries["candidate_max_conflict"].append(max_conflict)
                        totals["candidate_count"] += 1
                        if candidate.get("fallback") is True:
                            totals["candidate_fallback_count"] += 1
                            fallback_sources[candidate.get("fallback_source") or "unspecified"] += 1
                require(len(set(sets)) == len(sets), f"Duplicate candidate position sets: {context}")
                require(len({pairs is None for pairs in token_pairs_by_index.values()}) == 1,
                        f"Partial token logging within candidate pool: {context}")
                selected = step.get("selected_candidate")
                require(isinstance(selected, dict) and selected.get("index") in candidate_by_index,
                        f"Selected candidate is absent from pool: {context}")
                require(selected == candidate_by_index[selected["index"]],
                        f"Selected candidate record disagrees with pool: {context}")
                commit_tokens.append(token_pairs_by_index[selected["index"]])
                if index == 0:
                    first_proposed_tokens = token_pairs_by_index[valid[0]["index"]]
                committed.update(selected["positions"])
                base_calls = step.get("captured_base_forward_count")
                lookahead_calls = step.get("lookahead_model_calls")
                require(base_calls == 1 and lookahead_calls == expected_count,
                        f"Unexpected model-call accounting for chunk1/cfg0: {context}")
                calls += base_calls + lookahead_calls
                rank_base_calls += base_calls
                rank_lookahead_calls += lookahead_calls
                totals["actual_base_model_calls"] += base_calls
                totals["actual_lookahead_model_calls"] += lookahead_calls
                if padding:
                    continue
                totals["actions"] += 1
                totals["document_base_model_calls"] += base_calls
                totals["document_lookahead_model_calls"] += lookahead_calls
                realized_counts[expected_count] += 1
                action_sizes[selected["action_size"]] += 1
                selected_cost = selected["immediate_action_cost"]
                summaries["selected_entropy_cost"].append(selected_cost)
                summaries["selected_entropy_per_token"].append(selected_cost / 4)
                summaries["selected_mean_conflict"].append(selected["mean_within_set_conflict"])
                summaries["selected_max_conflict"].append(selected["max_within_set_conflict"])
                if index == 0:
                    summaries["first_action_entropy_cost"].append(selected_cost)
                if selected.get("fallback") is True:
                    totals["selected_fallback_count"] += 1
                    selected_sources[selected.get("fallback_source") or "unspecified"] += 1
                for first, second in combinations(sets, 2):
                    overlap = len(first & second)
                    overlaps[overlap / len(first | second)] += 1
                    changed[4 - overlap] += 1
                    symmetric[len(first ^ second)] += 1
                for name, value in step.get("timing_seconds", {}).items():
                    timings[name] += finite_number(value, context + " timing " + name, nonnegative=True)
                consistency_n = step.get("immediate_token_consistency_total", 0)
                consistency_k = step.get("immediate_token_consistency_count", 0)
                require(isinstance(consistency_n, int) and isinstance(consistency_k, int)
                        and 0 <= consistency_k <= consistency_n, f"Invalid consistency telemetry: {context}")
                totals["consistency_total"] += consistency_n
                totals["consistency_count"] += consistency_k
            require(len(committed) == length and max(committed) - min(committed) + 1 == length,
                    f"Committed positions do not cover one contiguous response: {path}, example {local_index}")
            response_start = min(committed)
            for block, positions in all_candidate_positions:
                require(all(response_start + block * block_size <= position < response_start + (block + 1) * block_size
                            for position in positions), f"Candidate position outside active response block: {path}")
            if not padding:
                require(len({pairs is None for pairs in commit_tokens}) == 1,
                        f"Partial token logging across decoding steps: {path}, example {local_index}")
                token_records[doc_id] = {"first_proposed": first_proposed_tokens, "commits": commit_tokens}
                summaries["actions_per_example"].append(len(steps))
                summaries["model_calls_per_example"].append(calls)
        require(records == expected_records, f"Diagnostic trace count mismatch: {path}")
        require(rank_runtime.get("generation_batch_count") == records,
                f"Runtime/trace batch count mismatch; cache reuse or partial artifacts: {path}")
        batch_seconds = rank_runtime.get("generation_batch_seconds")
        require(isinstance(batch_seconds, list) and len(batch_seconds) == records,
                f"Runtime batch timing count mismatch: {path}")
        seconds = [finite_number(value, "generation batch seconds", nonnegative=True) for value in batch_seconds]
        require(math.isclose(sum(seconds), rank_runtime["generation_total_seconds"], rel_tol=1e-9, abs_tol=1e-6),
                f"Runtime batch timings do not reproduce total: {path}")
        rank_metrics.append({"rank": rank, "documents": len(rank_document_ids), "document_ids": rank_document_ids,
                             "base_model_calls": rank_base_calls, "lookahead_model_calls": rank_lookahead_calls,
                             "model_calls": rank_base_calls + rank_lookahead_calls,
                             "generation_seconds": rank_runtime["generation_total_seconds"],
                             "diagnostics_path": str(path)})
    require(seen_documents == set(range(document_count)), f"Diagnostic documents do not match samples: {directory}")
    token_document_count = sum(record["first_proposed"] is not None for record in token_records.values())
    require(token_document_count in (0, document_count), f"Partial token logging across documents: {directory}")
    require(totals["padding_traces"] == world * expected_records - document_count,
            f"Unexpected distributed padding count: {directory}")
    return {
        **{key: distribution(values) for key, values in summaries.items()},
        **dict(totals),
        "world_size": world,
        "token_ids_available": token_document_count == document_count,
        "_token_records": token_records,
        "rank_metrics": rank_metrics,
        "padding_traces": totals["padding_traces"],
        "candidate_fallback_count": totals["candidate_fallback_count"],
        "selected_fallback_count": totals["selected_fallback_count"],
        "candidate_fallback_sources": dict(fallback_sources),
        "selected_fallback_sources": dict(selected_sources),
        "realized_candidate_count_histogram": dict(realized_counts),
        "selected_action_size_histogram": dict(action_sizes),
        "pairwise_jaccard": distribution(overlaps),
        "pairwise_replaced_positions": distribution(changed),
        "pairwise_symmetric_difference": distribution(symmetric),
        "pairwise_replaced_positions_histogram": dict(changed),
        "next_pass_consistency": (totals["consistency_count"] / totals["consistency_total"]
                                  if totals["consistency_total"] else None),
        "summed_document_step_timing_seconds": dict(timings),
        "actual_model_calls": totals["actual_base_model_calls"] + totals["actual_lookahead_model_calls"],
        "document_model_calls": totals["document_base_model_calls"] + totals["document_lookahead_model_calls"],
    }


def load_run(directory, arm, *, allow_launcher_hash_change=False):
    """Load a complete arm, validate configuration, and return metrics, samples, signature."""
    manifest = read_json(directory / "manifest.json")
    completed = read_json(directory / "completed.json")
    require(manifest.get("arm") == arm and completed.get("arm") == arm and completed.get("returncode") == 0,
            f"Missing/mismatched successful completion identity: {directory}")
    require(manifest.get("task") == "gsm8k_cot" and manifest.get("num_fewshot") == 5,
            f"Unexpected task/few-shot configuration: {directory}")
    require(manifest.get("evaluation_seed") == "0,1234,1234,1234", f"Unexpected evaluation seeds: {directory}")
    require(manifest.get("response_cache") is False, f"Uncached telemetry is required: {directory}")
    require(isinstance(manifest.get("source_hashes"), dict) and bool(manifest["source_hashes"]),
            f"Source hashes are required for comparison: {directory}")
    source_hashes = dict(manifest["source_hashes"])
    if allow_launcher_hash_change:
        required = {LAUNCHER_SOURCE, str(ROOT / "dllm/core/samplers/dependency.py"),
                    str(ROOT / "examples/path_selection/eval.py")}
        require(required <= source_hashes.keys(), f"Launcher exception requires sampler/evaluation source hashes: {directory}")
        source_hashes.pop(LAUNCHER_SOURCE)
    result_path = single_file(directory, "results_*.json")
    sample_path = single_file(directory, "samples_gsm8k_cot_*.jsonl")
    result = read_json(result_path)
    settings = model_arguments(result["config"]["model_args"])
    require(settings == model_arguments(manifest["model_args"]), f"Result/manifest model settings differ: {directory}")
    variant, budget, confidence_exponent = ARMS[arm]
    seed_strategy, seed_entropy_weight = SEED_SETTINGS.get(arm, ("legacy", 0.0))
    fixed = {"sampler_type": "entropy_drop", "proposal_strategy": "dependency", "diagnostic_retention": "full",
             "diagnostic_metadata": True, "dependency_cardinality_strategy": "fixed", "dependency_commit_k": 4,
             "max_new_tokens": 256, "block_size": 64, "temperature": 0, "cfg_scale": 0,
             "candidate_chunk_size": 1, "candidate_budget": budget, "dependency_parallel_variant": variant,
             "dependency_confidence_exponent": confidence_exponent,
             "dependency_seed_strategy": seed_strategy,
             "dependency_seed_entropy_weight": seed_entropy_weight}
    for key, value in fixed.items():
        require(settings.get(key, SEED_DEFAULTS.get(key)) == value,
                f"Unexpected {key} for {arm}: {settings.get(key)!r}; expected {value!r}")
    require(settings.get("pretrained") == manifest.get("checkpoint"), f"Checkpoint mismatch: {directory}")
    require(settings.get("dependency_generation_seed") == manifest.get("generation_seed"), f"Generation seed mismatch: {directory}")
    require(normalize_value(result["config"].get("batch_size")) == 1, f"batch_size=1 is required: {directory}")
    require(result["config"].get("use_cache") is None, f"Response cache is not permitted: {directory}")
    require(result["config"].get("limit") == manifest.get("limit"), f"Saved evaluation limit differs from manifest: {directory}")
    for key, value in (("random_seed", 0), ("numpy_seed", 1234), ("torch_seed", 1234), ("fewshot_seed", 1234)):
        require(result["config"].get(key) == value, f"Unexpected evaluation {key}: {directory}")
    require(result.get("n-shot", {}).get("gsm8k_cot") == 5, f"Reported few-shot setting differs: {directory}")
    task_config = dict(result["configs"]["gsm8k_cot"])
    metadata = dict(task_config.get("metadata", {}))
    for key, value in settings.items():
        require(key in metadata and normalize_value(metadata[key]) == value,
                f"Task metadata differs from model arguments ({key}): {directory}")
    task_config["metadata"] = {key: value for key, value in metadata.items() if key not in VARYING_ARGUMENTS}
    samples = load_samples(sample_path, result, manifest.get("limit"))
    count = len(samples[FILTERS[0]])
    runtime_path = directory / f"{PREFIX}_runtime.json"
    runtime = read_json(runtime_path)
    require(runtime.get("sampler_type") == "entropy_drop", f"Unexpected runtime sampler: {directory}")
    diagnostics = analyze_diagnostics(directory, runtime, manifest, settings, count)
    summary = {
        "arm": arm, "directory": str(directory), "result_path": str(result_path), "sample_path": str(sample_path),
        "manifest_path": str(directory / "manifest.json"), "runtime_path": str(runtime_path),
        "source_hashes": manifest["source_hashes"],
        "model_args": settings, "documents": count, "metrics": {}, "diagnostics": diagnostics,
        "generation_seconds": finite_number(runtime["generation_total_seconds"], "generation seconds", nonnegative=True),
        "generation_work_seconds": finite_number(runtime.get("generation_work_seconds", runtime["generation_total_seconds"]),
                                                  "generation work seconds", nonnegative=True),
        "generation_rank_seconds": [rank["generation_seconds"] for rank in diagnostics["rank_metrics"]],
        "launcher_wall_seconds": finite_number(completed["wall_seconds"], "launcher wall seconds", nonnegative=True),
    }
    for name in FILTERS:
        correct = int(sum(row["exact_match"] for row in samples[name].values()))
        summary["metrics"][name] = {"correct": correct, "accuracy": correct / count}
    signature = {
        "model_args": {key: value for key, value in settings.items() if key not in VARYING_ARGUMENTS},
        "evaluation_config": {key: value for key, value in result["config"].items() if key not in {"model_args", "use_cache"}},
        "task_config": task_config,
        "manifest": {key: manifest.get(key) for key in ("task", "num_fewshot", "evaluation_seed", "generation_seed", "limit",
                                                       "checkpoint", "world_size", "response_cache", "source_hashes")},
        "result_provenance": {key: result.get(key) for key in ("versions", "task_hashes", "git_hash", "upper_git_hash",
                                                              "transformers_version", "lm_eval_version", "system_instruction",
                                                              "system_instruction_sha", "fewshot_as_multiturn", "chat_template_sha")},
    }
    signature["manifest"]["source_hashes"] = source_hashes
    return summary, samples, signature


def resolve_arm_directories(run_root, arms, overrides=()):
    """Resolve explicit historical inputs without copying or changing their artifacts."""
    require(len(arms) >= 2 and len(set(arms)) == len(arms) and all(arm in ARMS for arm in arms),
            "Select at least two distinct known arms.")
    directories = {arm: run_root / arm for arm in arms}
    overridden = set()
    for override in overrides:
        arm, separator, raw_path = override.partition("=")
        require(separator and arm in directories and arm not in overridden,
                f"Expected one --arm-dir ARM=/absolute/path for a selected arm: {override}")
        path = Path(raw_path)
        require(path.is_absolute(), f"Use an absolute --arm-dir path: {override}")
        directories[arm] = path.resolve()
        overridden.add(arm)
    require(len(set(directories.values())) == len(directories), "Each arm must have its own artifact directory.")
    return directories


def analyze(run_root, arms=DEFAULT_ARMS, *, arm_dirs=(), allow_launcher_hash_change=False):
    """Validate selected arms before computing document-paired comparisons."""
    directories = resolve_arm_directories(run_root, arms, arm_dirs)
    runs, samples, signatures = {}, {}, {}
    for arm, directory in directories.items():
        runs[arm], samples[arm], signatures[arm] = load_run(directory, arm, allow_launcher_hash_change=allow_launcher_hash_change)
    reference_arm = arms[0]
    reference = signatures[reference_arm]
    for arm in arms:
        for field, value in reference.items():
            require(signatures[arm][field] == value,
                    f"{arm}/{reference_arm} mismatch in {field}; only declared arm settings may differ")
        for name in FILTERS:
            require(samples[arm][name].keys() == samples[reference_arm][name].keys(), f"{arm}/{reference_arm} document IDs differ")
            for doc_id, row in samples[arm][name].items():
                for field in IDENTITY_FIELDS:
                    require(row[field] == samples[reference_arm][name][doc_id][field],
                            f"{arm}/{reference_arm} {field} mismatch for {name}, document {doc_id}")
    comparisons = []
    token_comparisons = []
    for challenger, baseline in (("D8", "D4"), ("C8", "C4"), ("D4", "C4"), ("D8", "C8"),
                                 ("CD4", "D4"), ("CD8", "D8"), ("CD4", "C4"), ("CD8", "C8"), ("CD8", "CD4"),
                                 ("I4", "CS4"), ("I8", "CS8"), ("IE4", "I4"), ("IE8", "I8"),
                                 ("IE4", "CS4"), ("IE8", "CS8"), ("I8", "I4"), ("IE8", "IE4"), ("CS8", "CS4"),
                                 ("I4", "CD4"), ("I8", "CD8"), ("IE4", "CD4"), ("IE8", "CD8"),
                                 ("CS4", "CD4"), ("CS8", "CD8")):
        if challenger not in runs or baseline not in runs:
            continue
        token_comparisons.append({"challenger": challenger, "reference": baseline,
                                  **compare_token_commits(runs[challenger]["diagnostics"]["_token_records"],
                                                         runs[baseline]["diagnostics"]["_token_records"])})
        for name in FILTERS:
            first, second = samples[challenger][name], samples[baseline][name]
            wins = sum(first[key]["exact_match"] > second[key]["exact_match"] for key in first)
            losses = sum(first[key]["exact_match"] < second[key]["exact_match"] for key in first)
            comparisons.append({"challenger": challenger, "reference": baseline, "filter": name,
                                "documents": len(first), "wins": wins, "losses": losses,
                                "ties": len(first) - wins - losses, "difference_pp": 100 * (wins - losses) / len(first),
                                "exact_mcnemar_p_unadjusted": exact_mcnemar(wins, losses)})
    interactions = {name: 100 * ((runs["D8"]["metrics"][name]["accuracy"] - runs["D4"]["metrics"][name]["accuracy"])
                                - (runs["C8"]["metrics"][name]["accuracy"] - runs["C4"]["metrics"][name]["accuracy"]))
                    for name in FILTERS} if all(arm in runs for arm in DEFAULT_ARMS) else {}
    require(comparisons, "Selected arms have no planned paired comparison.")
    for run in runs.values():
        del run["diagnostics"]["_token_records"]
    return {"run_root": str(run_root), "runs": runs, "paired_comparisons": comparisons,
            "token_comparisons": token_comparisons,
            "allow_launcher_hash_change": allow_launcher_hash_change,
            "descriptive_difference_in_candidate_count_gains_pp": interactions,
            "checks": ["All requested successful completions and full artifacts are present",
                       "Model settings match except declared construction, candidate count, confidence exponent, and seed controls",
                       ("Source hashes match except the explicitly exempted ablation launcher" if allow_launcher_hash_change
                        else "All recorded source hashes match"),
                       "Documents, prompts, targets, and sample metrics match their declared evaluation scope",
                       "Every valid candidate contains four distinct positions; each pool is unique and has its feasible requested count",
                       "All response positions are committed exactly once, with complete runtime/model-call accounting"],
            "limits": ["Fixed k4 disables entropy-budget stopping; entropy cost is measured, not matched",
                       "C ranks seeds, companions, and refill by confidence; attention capture and conflict diagnostics are retained, so it is not an optimized no-attention implementation",
                       "Runs branch into different states: per-run diversity summaries are not comparisons of identical masked contexts",
                       "Position overlap is geometric diversity, not proof of different reasoning or correct outcomes",
                       "Exact McNemar p-values are exploratory and unadjusted for multiple comparisons; no across-seed claim",
                       "One- and two-GPU artifact layouts are supported, with the same world size required across all arms",
                       "Distributed sample counts must be divisible by world size; extra traces or padding are rejected",
                       "Maximum rank generation time is a measured proxy, not elapsed wall time; summed rank generation time measures generation work",
                       "Rank generation timings exclude per-batch waiting, trimming, logging, and model loading; launcher wall time is reported separately and excludes queue time"]}


def write_outputs(summary, output_directory):
    """Write the complete validated report and compact tabular measurements."""
    rows = []
    for arm, run in summary["runs"].items():
        diagnostic = run["diagnostics"]
        rows.append({"arm": arm, "documents": run["documents"],
                     "world_size": diagnostic["world_size"],
                     "flexible_accuracy": run["metrics"]["flexible-extract"]["accuracy"],
                     "strict_accuracy": run["metrics"]["strict-match"]["accuracy"],
                     "model_calls_per_example": diagnostic["model_calls_per_example"]["mean"],
                     "actual_model_calls_including_padding": diagnostic["actual_model_calls"],
                     "generation_seconds": run["generation_seconds"], "generation_work_seconds": run["generation_work_seconds"],
                     "launcher_wall_seconds": run["launcher_wall_seconds"],
                     "mean_pairwise_jaccard": diagnostic["pairwise_jaccard"]["mean"],
                     "mean_replaced_positions": diagnostic["pairwise_replaced_positions"]["mean"],
                     "mean_candidate_entropy_nats": diagnostic["candidate_entropy_cost"]["mean"],
                     "mean_selected_entropy_nats": diagnostic["selected_entropy_cost"]["mean"],
                     "mean_candidate_conflict": diagnostic["candidate_mean_conflict"]["mean"],
                     "mean_selected_conflict": diagnostic["selected_mean_conflict"]["mean"],
                     "candidate_fallback_count": diagnostic["candidate_fallback_count"],
                     "selected_fallback_count": diagnostic["selected_fallback_count"],
                     "padding_traces": diagnostic["padding_traces"]})
    lines = ["# Fixed-size candidate-search ablation", "",
             f"Validated {rows[0]['documents']} matching GSM8K documents across {', '.join(summary['runs'])}. "
             "All candidates have four positions; final block rounds have one feasible unique set.", "",
             "D uses soft-full construction with confidence exponent 0. CD uses the same construction with exponent 1. "
             "C ranks seed positions, companions, and refill by confidence. I uses confidence-weighted incoming seeds; "
             "IE adds exp(-H) to I; CS uses confidence-only seeds. I/IE/CS all retain CD companion and refill ranking. "
             "All retain position-noise diversity and attention capture, "
             "deterministic token values, and the same entropy-drop selector.", "",
             "| Arm | Flexible | Strict | Calls/example | Max rank generation hours | Summed rank generation GPU-hours | Launcher wall hours | Mean Jaccard | Mean replaced positions |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {100 * row['flexible_accuracy']:.2f}% | {100 * row['strict_accuracy']:.2f}% | "
                     f"{row['model_calls_per_example']:.2f} | {row['generation_seconds'] / 3600:.3f} | "
                     f"{row['generation_work_seconds'] / 3600:.3f} | "
                     f"{row['launcher_wall_seconds'] / 3600:.3f} | "
                     f"{row['mean_pairwise_jaccard']:.4f} | {row['mean_replaced_positions']:.3f} |")
    lines.extend(["", f"Each arm uses {rows[0]['world_size']} GPU rank(s). Max rank generation hours take the maximum of each rank's summed batch generation times; "
                  "summed rank generation GPU-hours add those rank totals. These measured proxies exclude per-batch waiting, trimming, logging, and model loading; "
                  "the ranks can alternate which finishes a batch last, so the maximum of their generation totals is not elapsed wall time. "
                  "Launcher wall hours include the evaluation process and completion checks, including loading and waiting, and exclude queue time. "
                  "Model calls sum executed base and verification forwards across ranks; per-example counts use the total document count.", "",
                  "Each overlap observation is an unordered pair within one candidate pool. "
                  "Replaced positions means 4 minus intersection size; symmetric-difference size is twice that. "
                  "These are pair-weighted summaries across visited states. Single-candidate pools have no pair and contribute no overlap observation.", "",
                  "| Candidate-set entropy (nats) | Candidate mean / p90 | Selected mean / p90 | First selected mean | Candidate / selected refills |",
                  "|---|---:|---:|---:|---:|"])
    for arm, run in summary["runs"].items():
        diagnostic = run["diagnostics"]
        candidate, selected = diagnostic["candidate_entropy_cost"], diagnostic["selected_entropy_cost"]
        lines.append(f"| {arm} | {candidate['mean']:.3f} / {candidate['p90']:.3f} | {selected['mean']:.3f} / {selected['p90']:.3f} | "
                     f"{diagnostic['first_action_entropy_cost']['mean']:.3f} | "
                     f"{diagnostic['candidate_fallback_count']} / {diagnostic['selected_fallback_count']} |")
    lines.extend(["", "| Challenger − reference | Filter | Difference (pp) | Wins / losses | Exact McNemar p |",
                  "|---|---|---:|---:|---:|"])
    for pair in summary["paired_comparisons"]:
        lines.append(f"| {pair['challenger']} − {pair['reference']} | {pair['filter']} | {pair['difference_pp']:+.2f} | "
                     f"{pair['wins']} / {pair['losses']} | {pair['exact_mcnemar_p_unadjusted']:.6g} |")
    lines.extend(["", "Differences pair final outcomes on the same documents, not intermediate decoding states. "
                  "P-values test discordant outcomes and are unadjusted. A nonsignificant result does not establish equivalence.", "",
                  ("The ablation launcher hash was explicitly exempted from comparison; all other recorded source hashes match. "
                   "Full source hashes are retained per arm in summary.json." if summary["allow_launcher_hash_change"]
                   else "All recorded source hashes match."), ""])
    interactions = summary["descriptive_difference_in_candidate_count_gains_pp"]
    if interactions:
        lines.extend(["The descriptive change in the N8−N4 gain between D and C is "
                      f"{interactions['flexible-extract']:+.2f} pp (flexible) and "
                      f"{interactions['strict-match']:+.2f} pp (strict). "
                      "No uncertainty interval is assigned to this interaction.", ""])
    lines.extend(["Exact token comparisons use (absolute position, token ID) pairs, not decoded text or position sets alone. "
                  "Final-response equality includes all generated positions, including special tokens and any text beyond evaluation stop strings. "
                  "Later-step comparisons describe actual runs after their contexts may have diverged.", "",
                  "| Challenger − reference | First proposed positions / tokens equal | First committed positions / tokens equal | Entire commit trace equal | Final response IDs equal |",
                  "|---|---:|---:|---:|---:|"])
    for pair in summary["token_comparisons"]:
        label = f"{pair['challenger']} − {pair['reference']}"
        if not pair["available"]:
            lines.append(f"| {label} | unavailable: token IDs not saved | unavailable | unavailable | unavailable |")
            continue
        n = pair["documents"]
        lines.append(f"| {label} | {pair['first_proposed_positions_equal']}/{n} / {pair['first_proposed_tokens_equal']}/{n} | "
                     f"{pair['first_committed_positions_equal']}/{n} / {pair['first_committed_tokens_equal']}/{n} | "
                     f"{pair['entire_commit_trace_equal']}/{n} | {pair['final_response_token_ids_equal']}/{n} |")
    lines.extend(["", "Per-document token equality and first divergent commitment step are recorded in summary.json.", ""])
    lines.extend(f"- {limit}." for limit in summary["limits"])
    lines.extend(["", "Sources:", ""])
    for arm, run in summary["runs"].items():
        lines.append(f"- {arm}: [results]({run['result_path']}), [samples]({run['sample_path']}), "
                     f"[manifest]({run['manifest_path']}), [runtime]({run['runtime_path']}).")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_directory / "report.md").write_text("\n".join(lines) + "\n")
    token_rows = [{key: value for key, value in pair.items() if key != "per_document"}
                  for pair in summary["token_comparisons"]]
    for filename, records in (("run_metrics.csv", rows), ("paired_comparisons.csv", summary["paired_comparisons"]),
                              ("token_comparisons.csv", token_rows)):
        with (output_directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in records for key in row)))
            writer.writeheader()
            writer.writerows(records)


def main():
    """Validate every arm before creating any analysis output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=DEFAULT_ARMS)
    parser.add_argument("--arm-dir", action="append", default=[], metavar="ARM=/absolute/path")
    parser.add_argument("--allow-launcher-hash-change", action="store_true",
                        help="Allow only the ablation launcher's recorded hash to differ; preserve all other provenance checks.")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if not args.run_root.is_absolute() or (args.output_directory is not None and not args.output_directory.is_absolute()):
        parser.error("Use absolute paths for --run-root and --output-directory.")
    run_root = args.run_root.resolve()
    output = args.output_directory.resolve() if args.output_directory else run_root / "analysis"
    try:
        directories = resolve_arm_directories(run_root, args.arms, args.arm_dir)
        artifact_directories = {*directories.values(), *(run_root / arm for arm in ARMS)}
        require(output != run_root and not any(output == path or path in output.parents for path in artifact_directories),
                "Analysis output must be separate from arm artifacts.")
        summary = analyze(run_root, args.arms, arm_dirs=args.arm_dir,
                          allow_launcher_hash_change=args.allow_launcher_hash_change)
        write_outputs(summary, output)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Analysis rejected incomplete or incomparable inputs: {error}\n")
    print(f"Validated {len(summary['runs'])} arms; report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
