"""Audit real top-k CTCA traces on CPU, without loading models.

    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/offline_master_gate.py

Missing probability mass is bounded, never renormalized away. These local
counterfactuals cannot establish correctness or a changed decoding trajectory.
"""

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = "ctca-gsm8k6-offline-mastergate-20261118"
CASES = [("ctca-gsm8k4-benchmark-replay-20261029", doc, "historical_loss")
         for doc in (23, 18, 9, 11)] + [
             ("ctca-gsm8k2-gain-replay-20261104", doc, "historical_gain")
             for doc in (3, 6)]


def h(value):
    return -value * math.log(value) if value > 0 else 0.0


def residual(values):
    mass = sum(values)
    if any(not math.isfinite(x) or x < 0 for x in values) or mass > 1 + 1e-5:
        raise ValueError("Invalid saved probability mass")
    return max(0.0, 1 - mass)


def uncertainty(values, vocabulary_size):
    """Entropy bounds: tail concentrated in one ID versus spread uniformly."""
    tail = residual(values)
    unseen = vocabulary_size - len(values)
    if vocabulary_size < 2 or unseen < 0 or (unseen == 0 and tail > 1e-5):
        raise ValueError("Invalid vocabulary size")
    base = sum(map(h, values)) + h(tail)
    normalizer = math.log(vocabulary_size)
    return [max(0., min(1., base / normalizer)),
            max(0., min(1., (base + tail * math.log(max(1, unseen))) / normalizer))]


def distribution_uncertainty(distribution):
    return uncertainty([entry["probability"] for entry in distribution["top_k"]],
                       distribution["vocabulary_size"])


def agreement_bounds(support):
    """Bound equal-weight JS and overlap using the known union and its tail."""
    # smoke.py::fusion_support emits one paired row per unique vocabulary ID.
    # Decoding is not injective: identical text can label distinct coordinates.
    # Preserve those rows separately; never merge or reject them by spelling.
    p, q = [row[1] for row in support], [row[2] for row in support]
    rp, rq = residual(p), residual(q)
    mixture = [(x + y) / 2 for x, y in zip(p, q)]
    rm = (rp + rq) / 2
    coarse_js = (sum(map(h, mixture)) + h(rm)
                 - (sum(map(h, p)) + h(rp) + sum(map(h, q)) + h(rq)) / 2)
    coarse_js = max(0., coarse_js)
    # Inside the tail, JS cannot exceed its mixture mass times binary entropy.
    tail_js = rm * (h(rp / (2 * rm)) + h(rq / (2 * rm))) if rm else 0.
    overlap = sum(min(x, y) for x, y in zip(p, q))
    return {"js": [max(0., 1 - (coarse_js + tail_js) / math.log(2)),
                   min(1., max(0., 1 - coarse_js / math.log(2)))],
            "overlap": [max(0., min(1., overlap)), min(1., overlap + min(rp, rq))],
            "master_remaining_mass": rp, "auxiliary_remaining_mass": rq}


def decision(row, metric, rule, agreement_threshold, uncertainty_threshold):
    low, high = row[metric]
    if rule == "agreement_only":
        confident, uncertain = True, False
    else:
        p, q = row["master_uncertainty"], row["projected_auxiliary_uncertainty"]
        combine = max if rule == "both" else min
        confident = combine(p[1], q[1]) <= uncertainty_threshold
        uncertain = combine(p[0], q[0]) > uncertainty_threshold
    if high < agreement_threshold or uncertain:
        return "master"
    if low >= agreement_threshold and confident:
        return "ensemble"
    return "ambiguous"


def configurations():
    for metric, rule, threshold in itertools.product(
            ("js", "overlap"), ("agreement_only", "both", "one"), (.4, .5, .7)):
        for uncertainty_threshold in ((None,) if rule == "agreement_only" else (.5, .8)):
            yield (metric, rule, threshold, uncertainty_threshold)


def step_rows(records):
    forwards = {(r["batch_index"], r["master"]["absolute_position"]): r
                for r in records["forward_position"]}
    commits = {(r["batch_index"], r["absolute_position"]): r
               for r in records["commit_decision"]}
    natives = {(r["batch_index"], r["generation_position"]): r
               for r in records["auxiliary_masked_position"]}
    for fusion in records["fusion_candidate"]:
        key = fusion["batch_index"], fusion["absolute_position"]
        forward = forwards[key]
        master = forward["master"]["distribution"]
        projected = fusion["projected_auxiliary_distribution"]
        spatial = forward["spatial_warp"]["distribution"]
        support = fusion["fusion_support"]
        try:
            bounds = agreement_bounds(support)
        except ValueError as error:
            yield {"step": fusion["step"], "batch_index": key[0], "position": key[1],
                   "unusable_reason": str(error)}
            continue
        p, q = [r[1] for r in support], [r[2] for r in support]
        contributor = max(forward["auxiliary"]["contributors"], key=lambda r: r["overlap"])
        native = natives[(key[0], contributor["generation_position"])]["distribution"]
        yield {"step": fusion["step"], "batch_index": key[0], "position": key[1],
               "duplicate_decoded_support_labels": len(support) - len({r[0] for r in support}),
               **bounds, "master_uncertainty": uncertainty(p, master["vocabulary_size"]),
               "projected_auxiliary_uncertainty": uncertainty(q, projected["vocabulary_size"]),
               "spatial_auxiliary_uncertainty": distribution_uncertainty(spatial),
               "native_contributor_uncertainty": distribution_uncertainty(native),
               "native_contributor_overlap": contributor["overlap"],
               "master_token": master["top_k"][0]["token"],
               "master_confidence": master["top_k"][0]["probability"],
               "fused_token": fusion["predicted_token"]["token"],
               "fused_confidence": fusion["fused_probability"],
               "source_token": spatial["top_k"][0]["token"],
               "source_probability": spatial["top_k"][0]["probability"],
               "projected_token": projected["top_k"][0]["token"],
               "selected": commits[key]["selected"],
               "saved_weights": fusion["weights"],
               "confidence_change_if_fallback": master["top_k"][0]["probability"] - fusion["fused_probability"]}


def audit_trace(path):
    records = defaultdict(list)
    step = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] not in {"forward_position", "fusion_candidate", "commit_decision",
                                    "auxiliary_masked_position"}:
                continue
            current = tuple(row["step"])
            if step is not None and current != step:
                yield from step_rows(records)
                records = defaultdict(list)
            step = current
            records[row["event"]].append(row)
    if records:
        yield from step_rows(records)


def resolve_trace(value):
    # Viewer manifests contain repo-served /.logs URLs or real absolute paths.
    return ROOT / value.lstrip("/") if value.startswith("/.logs/") else Path(value)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    report = {"study_id": STUDY, "limitations": [
        "Exploratory previously inspected cases; no held-out threshold selection.",
        "Historical gain/loss labels do not label individual token correctness.",
        "Top-k tails are bounded; ambiguous gates must not be treated as accepts or rejects.",
        "Only local proposals and confidence changes; no commitment reranking or live rollout.",
        "Native contributor entropy and post-projection entropy use different vocabulary normalizers.",
        "Dominant native contributor is diagnostic, not the full spatial mixture."], "cases": []}
    report["audit_version"] = 2
    report["support_provenance"] = "smoke.py fusion_support: distinct paired rows are distinct unique token IDs, even when decoded labels coincide"
    for study, doc, label in CASES:
        manifest = ROOT / ".logs" / study / f"doc{doc}.json"
        metadata = json.loads(manifest.read_text())
        trace = resolve_trace(next(p for p in metadata["traces"] if p.endswith("-ensemble.jsonl")))
        reproduced = next(r["matches_original_response"] for r in metadata["reproduction"]
                          if r["condition"] == "ensemble")
        counters = {config: Counter() for config in configurations()}
        count = 0
        destination = output / f"doc{doc}-positions.jsonl"
        with destination.open("w") as stream:
            for row in audit_trace(trace):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                if "unusable_reason" in row:
                    for counts in counters.values():
                        counts["unusable"] += 1
                    continue
                mismatch = (row["source_probability"] >= .5 and
                            row["source_token"] != row["projected_token"] and
                            not row["source_token"].startswith(("<|", "[Mask]")))
                for config, counts in counters.items():
                    action = decision(row, *config)
                    counts[action] += 1
                    if row["selected"]:
                        counts["selected_" + action] += 1
                    if mismatch:
                        counts["projection_mismatch_" + action] += 1
                    if action == "master" and row["master_token"] != row["fused_token"]:
                        counts["changed_proposal"] += 1
                        counts["selected_changed_proposal"] += int(row["selected"])
        report["cases"].append({"doc": doc, "historical_label": label,
                                 "trace": str(trace), "manifest": str(manifest),
                                 "ensemble_reproduced": reproduced, "positions": count,
                                 "position_report": str(destination),
                                 "gates": [{"metric": c[0], "rule": c[1],
                                            "agreement_threshold": c[2], "uncertainty_threshold": c[3],
                                            "counts": dict(v)} for c, v in counters.items()]})
        print(f"doc{doc}: {count} positions; ensemble reproduced={reproduced}", flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".logs" / STUDY)
    run(parser.parse_args().output_dir.resolve())
