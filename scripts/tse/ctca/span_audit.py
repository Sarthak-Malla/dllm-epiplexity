"""Audit text-span matches in saved CTCA traces without loading either model.

Run: python scripts/tse/ctca/span_audit.py --study-id ctca-gsm8k4-benchmark-replay-20261029
The report is written to .logs/STUDY_ID/span_audit.json.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
MAX_SPAN = 3


def candidates(distribution):
    return {item["token"]: float(item["probability"])
            for item in distribution["top_k"]}


def high_confidence_projection_mismatch(row):
    source = row["source_top"]
    return (row["selected"] and row["source_top_probability"] >= 0.5
            and "<|" not in source
            and source != row["projected_auxiliary_top"])


def match_spans(source, position, master, committed):
    """Find same-text master paths; never complete a source prefix by guessing."""
    matches = []
    blocked = []

    def walk(next_position, text, pieces, probability):
        if text == source:
            matches.append({"positions": list(range(position, next_position)),
                            "pieces": pieces, "master_product": probability})
            return
        if len(pieces) == MAX_SPAN or not source.startswith(text):
            return
        if next_position in committed:
            blocked.append({"position": next_position,
                            "committed_token": committed[next_position]["predicted_token"],
                            "committed_step": committed[next_position]["step"],
                            "prefix": text})
            return
        forward = master.get(next_position)
        if forward is None:
            return
        for token, token_probability in candidates(forward["master"]["distribution"]).items():
            if source.startswith(text + token):
                walk(next_position + 1, text + token, pieces + [token],
                     probability * token_probability)

    walk(position, "", [], 1.0)
    return sorted(matches, key=lambda item: item["master_product"], reverse=True), blocked


def audit_step(step, records, committed):
    auxiliary = {x["generation_position"]: x for x in records["auxiliary_masked_position"]}
    master = {x["master"]["absolute_position"]: x for x in records["forward_position"]}
    decisions = {x["absolute_position"]: x for x in records["commit_decision"]}
    fused = {x["absolute_position"]: x for x in records["fusion_candidate"]}
    selected_scores = sorted(
        (x["selection_score"] for x in decisions.values() if x["selected"]),
        reverse=True,
    )
    threshold = selected_scores[-1] if selected_scores else None
    rows = []
    for position, forward in master.items():
        contributors = forward["auxiliary"]["contributors"]
        if not contributors:
            continue
        contributor = max(contributors, key=lambda item: item["overlap"])
        native = auxiliary.get(contributor["generation_position"])
        if native is None:
            continue
        source_candidates = native["distribution"]["top_k"]
        findings = []
        for source in source_candidates:
            matches, blocked = match_spans(source["token"], position, master, committed)
            if matches or blocked:
                findings.append({"source": source["token"],
                                 "source_probability": source["probability"],
                                 "matches": matches[:3], "blocked": blocked[:1]})
        top = source_candidates[0]
        top_finding = next((x for x in findings if x["source"] == top["token"]), None)
        span_options = []
        for finding in findings:
            for match in finding["matches"]:
                if len(match["positions"]) < 2:
                    continue
                master_geometric_mean = match["master_product"] ** (1 / len(match["positions"]))
                master_weight = fused.get(position, {}).get("weights", {}).get("master", 0.5)
                if not isinstance(master_weight, (float, int)):
                    master_weight = 0.5
                score = master_weight * master_geometric_mean + (1 - master_weight) * finding["source_probability"]
                span_options.append({"source": finding["source"],
                                     "positions": match["positions"],
                                     "pieces": match["pieces"],
                                     "approximate_fused_score": score,
                                     "above_single_slot_threshold": threshold is not None and score >= threshold})
        span_options.sort(key=lambda option: option["approximate_fused_score"], reverse=True)
        top_status = ("exact" if top_finding and top_finding["matches"] else
                      "blocked" if top_finding and top_finding["blocked"] else
                      "unscored_from_saved_top64")
        decision = decisions.get(position)
        projected_top = (fused.get(position, {}).get("projected_auxiliary_distribution", {})
                         .get("top_k", [{}])[0].get("token"))
        rows.append({"step": list(step), "position": position,
                     "selected": bool(decision and decision["selected"]),
                     "committed_token": decision["predicted_token"] if decision and decision["selected"] else None,
                     "source_top": top["token"], "source_top_probability": top["probability"],
                     "projected_auxiliary_top": projected_top,
                     "source_top_status": top_status,
                     "source_top_match": (top_finding or {}).get("matches", []),
                     "source_top_blocked": (top_finding or {}).get("blocked", []),
                     "best_multi_slot_option": span_options[0] if span_options else None,
                     "single_slot_commit_threshold": threshold,
                     "matched_candidates": len([x for x in findings if x["matches"]]),
                     "top64_coverage": native["distribution"]["top_k_mass"]})
    for decision in decisions.values():
        if decision["selected"]:
            committed[decision["absolute_position"]] = decision
    return rows


def audit_trace(path):
    rows = []
    committed = {}
    records = None
    current_step = None
    interesting = {"auxiliary_masked_position", "forward_position", "fusion_candidate", "commit_decision"}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["event"] not in interesting:
                continue
            step = tuple(record["step"])
            if current_step is not None and step != current_step:
                rows.extend(audit_step(current_step, records, committed))
                records = None
            if records is None:
                current_step = step
                records = {event: [] for event in interesting}
            records[record["event"]].append(record)
    if records is not None:
        rows.extend(audit_step(current_step, records, committed))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.study_id):
        parser.error("study-id must be a simple directory name")
    study = ROOT / ".logs" / args.study_id
    report = {"study_id": args.study_id, "method": "recorded top-64 parallel probability product; no model rescoring", "docs": {}}
    manifests = sorted((path for path in study.glob("doc*.json")
                        if re.fullmatch(r"doc\d+\.json", path.name)),
                       key=lambda path: int(path.stem[3:]))
    if not manifests:
        parser.error(f"no doc*.json replay manifests in {study}")
    for manifest_path in manifests:
        doc = int(manifest_path.stem[3:])
        manifest = json.loads(manifest_path.read_text())
        trace = next(ROOT / path.lstrip("/") for path in manifest["traces"]
                     if path.endswith("-ensemble.jsonl"))
        rows = audit_trace(trace)
        counts = Counter(row["source_top_status"] for row in rows)
        mismatches = [row for row in rows if high_confidence_projection_mismatch(row)]
        report["docs"][str(doc)] = {"trace": str(trace.relative_to(ROOT)),
                                    "counts": dict(counts),
                                    "high_confidence_projection_mismatches": len(mismatches),
                                    "positions": rows}
        print(f"doc{doc}: {dict(counts)}; high-confidence non-special mismatches={len(mismatches)}", flush=True)
    destination = study / "span_audit.json"
    destination.write_text(json.dumps(report, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
