"""Summarize completed training-free experiment units without treating absence as evidence.

After sourcing /home/sarthak.malla/.zshrc and activating dllm, the user runs:
    srun -p "$PARTITION" -q "$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/analyze.py --output-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_two_gpu
Analysis is resumable between stages. It writes only the analysis subdirectory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

if __package__:
    from .artifacts import atomic_json, stable_hash
else:
    from artifacts import atomic_json, stable_hash


METRICS = ("flexible_correct", "strict_correct")


def average(values):
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else None


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def paired_summary(records, *, repeats=10000, seed=1234):
    """Summarize left-minus-right outcomes, resampling whole problems.

    Each problem receives equal weight, regardless of how many candidate/pair
    observations it contributes. Intervals describe this development corpus;
    zero discordances can yield degenerate bootstrap intervals and never certify
    the one-percentage-point noninferiority margin.
    """
    if not records:
        return None
    by_problem = defaultdict(list)
    for record in records:
        by_problem[int(record["doc_id"])].append(record)
    differences = [mean(float(item["left"]) - float(item["right"]) for item in group)
                   for _, group in sorted(by_problem.items())]
    rng = random.Random(seed)
    samples = [mean(rng.choices(differences, k=len(differences))) for _ in range(repeats)]
    wins = sum(bool(item["left"]) and not bool(item["right"]) for item in records)
    losses = sum(bool(item["right"]) and not bool(item["left"]) for item in records)
    discordant = wins + losses
    # Exact McNemar is only meaningful here for one outcome pair per problem.
    exact_p = None
    if len(records) == len(by_problem):
        exact_p = (min(1.0, 2 * sum(math.comb(discordant, index)
                                   for index in range(min(wins, losses) + 1)) / (2 ** discordant))
                   if discordant else 1.0)
    return {
        "problems": len(by_problem), "observations": len(records),
        "left_accuracy": mean(mean(float(item["left"]) for item in group) for group in by_problem.values()),
        "right_accuracy": mean(mean(float(item["right"]) for item in group) for group in by_problem.values()),
        "wins": wins, "losses": losses,
        "difference_pp": 100 * mean(differences),
        "ci95_pp": [100 * percentile(samples, 0.025), 100 * percentile(samples, 0.975)],
        "exact_mcnemar_p": exact_p,
        "inference": "exploratory; does not certify noninferiority",
    }


def load_units(root, kind, manifest_hash):
    units = {}
    sources = ([root] if (root / "manifest.json").is_file() else
               [path.parent for path in sorted((root / "workers").glob("worker*/manifest.json"))])
    for source in sources:
        for path in sorted((source / kind).glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("_manifest_hash") != manifest_hash:
                raise ValueError(f"Mismatched artifact provenance: {path}")
            value.pop("_manifest_hash")
            key = f"{source.name}:{path.stem}" if source != root and kind == "completed" else path.stem
            if key in units and stable_hash(units[key]) != stable_hash(value):
                raise ValueError(f"Conflicting worker artifacts for {kind}/{key}: {path}")
            units[key] = value
    return units


def _branch(branches, key, origin):
    if key not in branches:
        raise ValueError(f"Completed diagnostic {origin} refers to missing branch {key!r}")
    return branches[key]


def _pair(branches, left_key, right_key, metric, origin):
    left, right = _branch(branches, left_key, origin), _branch(branches, right_key, origin)
    if left["doc_id"] != right["doc_id"] or left["state_key"] != right["state_key"]:
        raise ValueError(f"Comparison does not share a frozen state: {origin}")
    if left.get("continuation_config_hash") != right.get("continuation_config_hash"):
        raise ValueError(f"Comparison changes the continuation policy: {origin}")
    return {"doc_id": left["doc_id"], "left": left[metric], "right": right[metric]}


def diagnostic_summaries(diagnostics, branches):
    paired = defaultdict(list)
    selectors = defaultdict(list)
    selector_by_state = {
        record["state_key"]: {item["pool"]: item for item in record.get("comparisons", [])}
        for record in diagnostics.values() if record.get("experiment") == "selectors"
        and "state_key" in record
    }
    for key, record in diagnostics.items():
        experiment = record.get("experiment")
        if experiment == "selectors":
            for comparison in record.get("comparisons", []):
                pool = comparison["pool"]
                selectors[pool].append(comparison)
                for metric in METRICS:
                    observation = _pair(branches, comparison["entropy_branch"], comparison["cheap_branch"], metric, key)
                    paired[f"selector/{pool}/{metric}/entropy-minus-cheap"].append(observation)
                    if not comparison["exact_agreement"]:
                        paired[f"selector/{pool}/{metric}/disagreements-only/entropy-minus-cheap"].append(observation)
        elif experiment == "precedence":
            for group in record.get("groups", []):
                label = group.get("label", "group")
                if label == "seed_first_companion_pair":
                    label += "/" + group.get("construction_order_kind", "greedy")
                pool = group.get("pool", "adaptive")
                for mode, field in (("seed_first", "seed_first_branch"), ("reverse", "reverse_branch")):
                    if not group.get(field):
                        continue
                    for metric in METRICS:
                        paired[f"precedence/{pool}/{label}/{metric}/{mode}-minus-simultaneous"].append(
                            _pair(branches, group[field], group["simultaneous_branch"], metric, key))
        elif experiment == "proposals":
            pools = record.get("proposal_pools", {})
            if not pools:
                continue
            if not pools.get("dependency") or not pools.get("confidence"):
                raise ValueError(f"Proposal comparison lacks a pool: {key}")
            for metric in METRICS:
                outcomes = {name: any(_branch(branches, branch_key, key)[metric] for branch_key in keys)
                            for name, keys in pools.items()}
                paired[f"proposal/{metric}/confidence-oracle-minus-dependency-oracle"].append(
                    {"doc_id": record["doc_id"], "left": outcomes["confidence"], "right": outcomes["dependency"]})
                selection = selector_by_state.get(record["state_key"], {}).get("fixed4")
                if selection is not None:
                    for selector in ("entropy", "cheap"):
                        chosen = _branch(branches, selection[f"{selector}_branch"], key)
                        paired[f"proposal/{metric}/dependency-oracle-minus-{selector}-choice"].append(
                            {"doc_id": record["doc_id"], "left": outcomes["dependency"], "right": chosen[metric]})
    overlap = {
        pool: {
            "states": len(items), "exact_agreement": average(item["exact_agreement"] for item in items),
            "disagreement_states": sum(not item["exact_agreement"] for item in items),
            "mean_jaccard": average(item["jaccard"] for item in items),
            "mean_entropy_action_size": average(item["entropy_size"] for item in items),
            "mean_cheap_action_size": average(item["cheap_size"] for item in items),
        }
        for pool, items in selectors.items()
    }
    return {name: paired_summary(items) for name, items in sorted(paired.items())}, overlap


def candidate_pool_summaries(states):
    """Quantify proposal redundancy and uncertainty without conflating trajectories."""
    by_pool = defaultdict(list)
    for state in states.values():
        for name, records in state.get("pools", {}).items():
            masks = [set(record["positions"]) for record in records]
            pairs = [len(left & right) / len(left | right)
                     for index, left in enumerate(masks) for right in masks[index + 1:] if left | right]
            by_pool[name].append({
                "candidate_count": len(records),
                "unique_position_sets": len({tuple(sorted(mask)) for mask in masks}),
                "within_pool_jaccard": average(pairs),
                "mean_action_size": average(len(mask) for mask in masks),
                "mean_group_entropy": average(sum(record["entropy"]) for record in records),
                "mean_group_confidence": average(average(record["confidence"]) for record in records),
            })
    return {name: {"states": len(items), **{
        field: average(item[field] for item in items) for field in items[0]
    }} for name, items in sorted(by_pool.items())}


def _band(value, boundaries):
    if value is None:
        return "unavailable"
    lower = 0.0
    for upper in boundaries:
        if value < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return f"[{lower:g},inf)"


def stability_summaries(probes, states, diagnostics=None):
    """Expose observed flips within marginal-uncertainty and conflict strata."""
    rows = []
    group_rows = []
    seen_rows, seen_groups = set(), set()
    for record in (diagnostics or {}).values():
        for group in record.get("probes", []):
            if not isinstance(group, dict):
                continue
            state_key = record["state_key"]
            state = states.get(state_key, {})
            probe_key = group["probe_key"]
            if probe_key not in probes:
                raise ValueError(f"Stability record references a missing probe: {probe_key}")
            group_identity = (state_key, group.get("pool"), group.get("candidate_index"))
            if group_identity not in seen_groups:
                seen_groups.add(group_identity)
                group_rows.append({"doc_id": record["doc_id"], "size": group["group_size"],
                                   "any_flip": group["any_flip"]})
            for feature in group.get("features", []):
                representations = [("conditional", feature)]
                if feature.get("absolute_attention"):
                    representations.append(("absolute", {**feature, **feature["absolute_attention"]}))
                for representation, item in representations:
                    identity = (*group_identity, feature["companion"], representation)
                    if identity in seen_rows:
                        continue
                    seen_rows.add(identity)
                    row = {
                        "doc_id": record["doc_id"], "state_key": state_key,
                        "probe_key": probe_key, "stage": state.get("threshold"),
                        "pool": group.get("pool"), "candidate_index": group.get("candidate_index"),
                        "group_size": group["group_size"], "representation": representation,
                        "position": item["companion"], "seed": item["seed"],
                        "base_confidence": item["confidence"], "base_entropy": item["entropy"],
                        "base_margin": item["top2_margin"], "seed_confidence": item["seed_confidence"],
                        "changed": feature["flipped"],
                        "original_probability": feature["original_value_probability"],
                        "refreshed_original_probability": feature["refreshed_original_value_probability"],
                        "probability_loss": feature["probability_loss"],
                        "total_variation": feature["total_variation"],
                        "attention_companion_to_seed": item["companion_to_seed"],
                        "attention_seed_to_companion": item["seed_to_companion"],
                        "normalized_conflict": item["symmetric_conflict"],
                        "conflict_scale": item["conflict_scale"],
                        "confidence_weight": item["confidence_weight"], "weighted_penalty": item["penalty"],
                        "selected_key_mass": item["selected_key_mass"],
                        "attention_mass_by_region": item.get("attention_mass_by_region"),
                    }
                    rows.append(row)
    dimensions = {
        "confidence": lambda row: _band(row.get("base_confidence"), (0.5, 0.8, 0.9, 0.95, 1.000001)),
        "seed_confidence": lambda row: _band(row.get("seed_confidence"), (0.5, 0.8, 0.9, 0.95, 1.000001)),
        "entropy": lambda row: _band(row.get("base_entropy"), (0.25, 0.5, 1, 2, 4)),
        "margin": lambda row: _band(row.get("base_margin"), (0.05, 0.1, 0.25, 0.5)),
        "stage": lambda row: str(row.get("stage")),
        "size": lambda row: str(row["group_size"]),
    }
    dimensions["joint"] = lambda row: "|".join(
        f"{name}={dimensions[name](row)}"
        for name in ("stage", "size", "seed_confidence", "entropy", "margin")
    )
    risk_features = {
        "normalized_conflict": (0.05, 0.1, 0.25, 0.5, 1.000001),
        "weighted_penalty": (0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        "attention_companion_to_seed": (0.0001, 0.001, 0.01, 0.05, 0.1),
    }
    stratified = []
    # Confidence is always controlled alongside the feature/dimension being shown.
    for risk_feature, boundaries in risk_features.items():
        for dimension, function in dimensions.items():
            buckets = defaultdict(list)
            for row in rows:
                risk_band = _band(row.get(risk_feature), boundaries)
                buckets[(row["representation"], function(row), dimensions["confidence"](row), risk_band)].append(row)
            for (representation, band, confidence, risk_band), items in sorted(buckets.items()):
                # Joint strata are deliberately reported even when sparse, with
                # sample counts; no smoothing or learned predictor is fitted.
                stratified.append({
                    "representation": representation, "dimension": dimension, "band": band,
                    "confidence_band": confidence, "risk_feature": risk_feature, "risk_band": risk_band,
                    "observations": len(items), "problems": len({item["doc_id"] for item in items}),
                    "flip_rate": average(item.get("changed") for item in items),
                    "mean_probability_loss": average(item.get("probability_loss") for item in items),
                    "mean_total_variation": average(item.get("total_variation") for item in items),
                })
    return {
        "companions": sum(row["representation"] == "conditional" for row in rows), "groups": len(group_rows),
        "problems": len({row["doc_id"] for row in rows}),
        "flip_rate": average(row.get("changed") for row in rows if row["representation"] == "conditional"),
        "any_flip_group_rate": average(row["any_flip"] for row in group_rows),
        "strata": stratified, "rows": rows,
        "interpretation": "Attention association and token stability do not establish correctness or independence.",
    }


def benchmark_summaries(records, expected_documents):
    groups = defaultdict(dict)
    for record in records.values():
        arm, doc_id = record["arm"], int(record["doc_id"])
        if doc_id in groups[arm]:
            raise ValueError(f"Duplicate benchmark outcome: {arm}, document {doc_id}")
        groups[arm][doc_id] = record
    summaries = {}
    comparisons = {}
    for arm, by_doc in sorted(groups.items()):
        items = list(by_doc.values())
        summaries[arm] = {
            "documents": len(items), "complete": set(by_doc) == set(expected_documents),
            "flexible_accuracy": average(item["flexible_correct"] for item in items),
            "strict_accuracy": average(item["strict_correct"] for item in items),
            "mean_evaluated_rows": average(item.get("accounting", {}).get("evaluated_rows") for item in items),
            "mean_model_calls": average(item.get("accounting", {}).get("model_calls") for item in items),
            "mean_wall_seconds": average(item.get("wall_seconds") for item in items),
            "mean_generation_seconds": average(item.get("generation_seconds", item.get("wall_seconds")) for item in items),
            "mean_reconstruction_seconds": average(item.get("reconstruction_seconds") for item in items),
            "mean_proposal_seconds": average(item.get("proposal_seconds") for item in items),
            "mean_action_size": average(action.get("size", len(action.get("positions", [])))
                                        for item in items for action in item.get("actions", [])),
        }
        references = ["reference_entropy", "reference_cheap"]
        if arm.startswith("threshold_"):
            if "_block256_" in arm:
                references.append(arm.replace("_block256_", "_block64_"))
            if arm.endswith("_incoming"):
                references.append(arm.removesuffix("_incoming") + "_confidence")
        for reference in references:
            if arm == reference or reference not in groups:
                continue
            reference_docs = groups[reference]
            shared = sorted(set(by_doc) & set(reference_docs))
            for metric in METRICS:
                comparisons[f"{arm}-minus-{reference}/{metric}"] = paired_summary([
                    {"doc_id": doc, "left": by_doc[doc][metric], "right": reference_docs[doc][metric]}
                    for doc in shared
                ])
    complete = {arm: values for arm, values in summaries.items() if values["complete"]}
    reference = complete.get("reference_entropy")
    nomination = None
    if reference:
        candidates = [(arm, values) for arm, values in complete.items()
                      if values["flexible_accuracy"] >= reference["flexible_accuracy"] - 0.010000001
                      and values["mean_evaluated_rows"] is not None]
        if candidates:
            arm, values = min(candidates, key=lambda pair: (pair[1]["mean_evaluated_rows"],
                                                           pair[1]["mean_wall_seconds"], pair[0]))
            nomination = {"arm": arm, "basis": "Lowest measured mean evaluations within one point of reference development accuracy; latency breaks ties.",
                          "confirmation_required": True, "noninferiority_established": False,
                          "mean_evaluated_rows": values["mean_evaluated_rows"]}
    frontier = []
    for arm, values in complete.items():
        if values["mean_evaluated_rows"] is None or values["mean_wall_seconds"] is None:
            continue
        dominated = any(
            other != arm and other_values["mean_evaluated_rows"] is not None
            and other_values["mean_wall_seconds"] is not None
            and other_values["flexible_accuracy"] >= values["flexible_accuracy"]
            and other_values["mean_evaluated_rows"] <= values["mean_evaluated_rows"]
            and other_values["mean_wall_seconds"] <= values["mean_wall_seconds"]
            and (other_values["flexible_accuracy"] > values["flexible_accuracy"]
                 or other_values["mean_evaluated_rows"] < values["mean_evaluated_rows"]
                 or other_values["mean_wall_seconds"] < values["mean_wall_seconds"])
            for other, other_values in complete.items()
        )
        if not dominated:
            frontier.append(arm)
    return summaries, comparisons, nomination, frontier


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def mechanism_findings(paired, benchmark_pairs, stability):
    """Label measured directional benefits without interpreting absence as failure."""
    comparisons = {**paired, **benchmark_pairs}
    questions = {
        "entropy_selection_benefit": "selector/adaptive/flexible_correct/entropy-minus-cheap",
        "entropy_selection_benefit_fixed4": "selector/fixed4/flexible_correct/entropy-minus-cheap",
        "proposal_failure_relative_to_confidence_pool": "proposal/flexible_correct/confidence-oracle-minus-dependency-oracle",
        "entropy_scoring_misses_successful_candidates": "proposal/flexible_correct/dependency-oracle-minus-entropy-choice",
        "cheap_scoring_misses_successful_candidates": "proposal/flexible_correct/dependency-oracle-minus-cheap-choice",
        "unsafe_simultaneous_commitment": "precedence/adaptive/natural/flexible_correct/seed_first-minus-simultaneous",
        "absolute_attention_accuracy_benefit": "attention_mass-minus-reference_cheap/flexible_correct",
        "affordable_growth_accuracy_benefit": "affordable_companions-minus-reference_cheap/flexible_correct",
    }
    for threshold in ("0.80", "0.90", "0.95"):
        questions[f"full_response_accuracy_benefit_at_{threshold}"] = (
            f"threshold_{threshold}_block256_confidence-minus-threshold_{threshold}_block64_confidence/flexible_correct")
        questions[f"incoming_ranking_accuracy_benefit_at_{threshold}"] = (
            f"threshold_{threshold}_block64_incoming-minus-threshold_{threshold}_block64_confidence/flexible_correct")
    findings = {}
    for name, comparison in questions.items():
        result = comparisons.get(comparison)
        status = "unresolved"
        reason = "The required paired comparison is incomplete or unavailable."
        if result:
            lower, upper = result["ci95_pp"]
            if lower > 0:
                status = "supported_in_development"
                reason = "The exploratory interval favors the stated directional benefit."
            elif upper < 0:
                status = "unsupported_in_development"
                reason = "The exploratory interval favors the opposite direction."
            else:
                reason = "The exploratory interval includes zero; this does not establish equivalence."
        findings[name] = {"status": status, "comparison": comparison, "reason": reason}
    findings["conflict_predicts_stability_beyond_confidence"] = {
        "status": "unresolved",
        "reason": ("Review confidence-conditioned and joint strata; marginal stability rates alone cannot establish added conflict value."
                   if stability["companions"] else "No completed seed-probe stability observations are available."),
    }
    return findings


def _number(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def build_report(root: Path):
    manifest_paths = ([root / "manifest.json"] if (root / "manifest.json").is_file() else
                      sorted((root / "workers").glob("worker*/manifest.json")))
    if not manifest_paths:
        raise FileNotFoundError(f"No experiment manifest or worker manifests under {root}")
    manifest = json.loads(manifest_paths[0].read_text())
    manifest_hash = stable_hash({key: value for key, value in manifest.items() if key != "created_at"})
    for path in manifest_paths[1:]:
        candidate = json.loads(path.read_text())
        if stable_hash({key: value for key, value in candidate.items() if key != "created_at"}) != manifest_hash:
            raise ValueError(f"Workers have different configurations, sources, hardware, or document identities: {path}")
    kinds = ("documents", "states", "collection", "branches", "probes", "diagnostics", "benchmarks", "ledger", "completed", "reuse", "invocations")
    units = {kind: load_units(root, kind, manifest_hash) for kind in kinds}
    if manifest.get("kind") == "policy_benchmark":
        if __package__:
            from .benchmark_analysis import build_benchmark_report
        else:
            from benchmark_analysis import build_benchmark_report
        return build_benchmark_report(root, manifest, units, paired_summary)
    settings = manifest.get("configuration", manifest.get("settings", manifest))
    expected_docs = settings.get("document_ids", list(range(100)))
    paired, overlap = diagnostic_summaries(units["diagnostics"], units["branches"])
    pool_summary = candidate_pool_summaries(units["states"])
    stability = stability_summaries(units["probes"], units["states"], units["diagnostics"])
    benchmarks, benchmark_pairs, nomination, frontier = benchmark_summaries(units["benchmarks"], expected_docs)
    ledger = units["ledger"]
    totals = ({field: sum(record.get("accounting", {}).get(field, 0) for record in ledger.values())
               for field in ("model_calls", "evaluated_rows", "input_tokens", "model_seconds")}
              if ledger else None)
    phases = defaultdict(list)
    for record in ledger.values():
        phases[record.get("phase", "unspecified")].append(record)
    cost_by_phase = {phase: {field: sum(item.get("accounting", {}).get(field, 0) for item in items)
                              for field in ("model_calls", "evaluated_rows", "input_tokens", "model_seconds")}
                     for phase, items in sorted(phases.items())}
    launches = {path.stem: json.loads(path.read_text())
                for path in sorted((root / "launches").glob("*.json"))}
    summary = {
        "manifest_hash": manifest_hash, "expected_documents": expected_docs,
        "coverage": {kind: len(items) for kind, items in units.items()},
        "worker_manifests": [str(path) for path in manifest_paths],
        "completed": units["completed"], "selector_overlap": overlap,
        "candidate_pools": pool_summary,
        "paired_diagnostics": paired, "stability": {key: value for key, value in stability.items() if key != "rows"},
        "benchmarks": benchmarks, "paired_benchmarks": benchmark_pairs,
        "development_frontier": frontier, "development_nomination": nomination,
        "mechanism_findings": mechanism_findings(paired, benchmark_pairs, stability),
        "recorded_physical_work": totals, "cost_by_phase": cost_by_phase,
        "invocation_worker_seconds": (sum(record.get("wall_seconds", 0) for record in units["invocations"].values())
                                    if units["invocations"] else None),
        "model_setup_worker_seconds": (sum(record.get("setup_seconds", 0) for record in units["invocations"].values())
                                if units["invocations"] else None),
        "launch_elapsed_seconds": (sum(record.get("wall_seconds", 0) for record in launches.values())
                                   if launches else None),
        "launches": launches,
        "limitations": [
            "Missing or unfinished stages are not experimental evidence.",
            "One hundred development problems cannot automatically certify the one-point accuracy margin.",
            "An observed accuracy gap within one point does not establish the one-point preservation margin.",
            "Diagnostic branches use the same cheap continuation; they are not complete-policy benchmarks.",
            "Bootstrap intervals resample problems, not individual states or companions; zero discordance can make intervals degenerate.",
            "Recorded completed/failed transactions exclude work after the last durable record if a process was forcibly killed.",
            "Attention is a proxy and model consistency is not mathematical correctness.",
        ],
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    write_csv(output / "paired_comparisons.csv", [{"comparison": key, **value} for key, value in {**paired, **benchmark_pairs}.items() if value])
    write_csv(output / "benchmarks.csv", [{"arm": key, **value} for key, value in benchmarks.items()])
    write_csv(output / "stability_strata.csv", stability["strata"])
    write_csv(output / "companion_stability.csv", stability["rows"])
    lines = ["# Training-free decoding: experiment report", "",
             f"Run: [{root.name}]({root})", "",
             f"Completed collection: {len(units['collection'])}/{len(expected_docs)} problems; saved states: {len(units['states'])}/{3 * len(expected_docs)}.",
             "Missing stages remain untested. All findings below are exploratory.", "",
             "## Selector and commitment comparisons", "",
             "Positive differences favor the left-hand method named in the comparison. Intervals resample whole problems.", "",
             "| Comparison | Problems | Wins / losses | Difference (pp) | 95% interval (pp) |",
             "|---|---:|---:|---:|---|" ]
    for name, values in paired.items():
        if values:
            interval = values["ci95_pp"]
            lines.append(f"| {name} | {values['problems']} | {values['wins']} / {values['losses']} | {_number(values['difference_pp'])} | [{_number(interval[0])}, {_number(interval[1])}] |")
    if not paired:
        lines.extend(["", "No completed paired diagnostics are available."])
    lines.extend(["", "| Pool | States | Mean candidates | Within-pool Jaccard | Group entropy | Group confidence |",
                  "|---|---:|---:|---:|---:|---:|"])
    for pool, values in pool_summary.items():
        lines.append(f"| {pool} | {values['states']} | {_number(values['candidate_count'])} | {_number(values['within_pool_jaccard'])} | {_number(values['mean_group_entropy'])} | {_number(values['mean_group_confidence'])} |")
    lines.extend(["", "## Instability", "",
                  f"Measured companions: {stability['companions']}; token flip rate: {_number(stability['flip_rate'])}; any-flip group rate: {_number(stability['any_flip_group_rate'])}.",
                  "Inspect confidence-conditioned conflict curves in stability_strata.csv and the raw directed/weighted features in companion_stability.csv. Marginal correlations alone do not establish value beyond confidence.",
                  "", "## Complete decoding benchmarks", "",
                  "| Arm | Documents | Flexible accuracy | Strict accuracy | Mean evaluations | Mean seconds | Complete |",
                  "|---|---:|---:|---:|---:|---:|---|" ])
    for arm, values in benchmarks.items():
        lines.append(f"| {arm} | {values['documents']} | {_number(100 * values['flexible_accuracy'], 2)}% | {_number(100 * values['strict_accuracy'], 2)}% | {_number(values['mean_evaluated_rows'])} | {_number(values['mean_wall_seconds'])} | {values['complete']} |")
    lines.extend(["", "## Decisions", ""])
    for name, finding in summary["mechanism_findings"].items():
        lines.append(f"- {name}: **{finding['status']}**. {finding['reason']}")
    lines.append("")
    for name, values in paired.items():
        if "flexible_correct" not in name or not values:
            continue
        low, high = values["ci95_pp"]
        evidence = "left-hand benefit supported in this development corpus" if low > 0 else (
            "right-hand benefit supported in this development corpus" if high < 0 else "accuracy difference unresolved")
        lines.append(f"- {name}: {evidence}.")
    if nomination:
        lines.extend(["", f"Development nomination: **{nomination['arm']}**. {nomination['basis']}",
                      "This nominates a larger confirmation run; it does not establish noninferiority."])
    else:
        lines.extend(["", "No nomination: complete the reference and candidate benchmarks before choosing a decoder."])
    lines.extend(["", "Directional accuracy labels above do not assess evaluation savings by themselves. Use the complete benchmark frontier for the accuracy–evaluation–latency tradeoff. When no added mechanism improves that tradeoff, retain the inexpensive reference. Do not combine changes solely because their individual point estimates are favorable.",
                  "", "## Accounting and limits", "",
                  f"Recorded physical evaluations: {_number(None if totals is None else totals['evaluated_rows'], 0)}; calls: {_number(None if totals is None else totals['model_calls'], 0)}. Costs are summed from disjoint ledger transactions, not added again from branch/probe summaries.", ""])
    lines.extend([f"Recorded invocation work: {_number(summary['invocation_worker_seconds'])} worker-seconds; setup: {_number(summary['model_setup_worker_seconds'])} worker-seconds. Setup includes model loading and request preparation. Summed concurrent launch duration: {_number(summary['launch_elapsed_seconds'])} elapsed seconds, excluding queue time. Worker-seconds are not job elapsed time.", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--run-root", type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(args.output_root.resolve())
    print(json.dumps({"coverage": summary["coverage"], "development_nomination": summary["development_nomination"]}, indent=2))


if __name__ == "__main__":
    main()
