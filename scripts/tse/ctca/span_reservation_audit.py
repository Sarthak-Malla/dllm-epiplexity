"""Audit schedule-preserving protection of cross-tokenizer text spans.

Run from the repository root:
    python scripts/tse/ctca/span_reservation_audit.py --study-id STUDY_ID
This is an offline counterfactual on saved traces; it does not replay decoding.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_tokenizer

from native_span_rescore import MODEL_PATHS, ROOT, native_ids, trace_steps


def span_proposals(records, tokenizer, threshold, max_span):
    auxiliary = {row["generation_position"]: row
                 for row in records["auxiliary_masked_position"]}
    decisions = {row["absolute_position"]: row
                 for row in records["commit_decision"]}
    active = set(decisions)
    proposals = {}
    for row in records["forward_position"]:
        master = row["master"]
        position = master["absolute_position"]
        contributors = row["auxiliary"]["contributors"]
        if not contributors or position not in active:
            continue
        contributor = max(contributors, key=lambda item: item["overlap"])
        source = auxiliary.get(contributor["generation_position"])
        if source is None:
            continue
        top = source["distribution"]["top_k"][0]
        if top["probability"] < threshold:
            continue
        ids = native_ids(tokenizer, top["token"], max_span)
        if ids is None or len(ids) < 2:
            continue
        slots = tuple(range(position, position + len(ids)))
        if not all(slot in active for slot in slots):
            continue
        key = (slots, top["token"])
        proposals[key] = {"text": top["token"], "master_token_ids": ids,
                          "positions": list(slots),
                          "source_probability": top["probability"],
                          "source_generation_position": source["generation_position"],
                          "master_top": master["distribution"]["top_k"][0]}
    return list(proposals.values())


def select_with_protection(decisions, proposals):
    """Keep the scheduled count by filling from the best unprotected positions."""
    active = {row["absolute_position"]: row for row in decisions}
    budget = next((row["scheduled_transfer_count"] for row in decisions), 0)
    protected = set()
    accepted = []
    rejected = []
    for proposal in sorted(proposals, key=lambda item: item["source_probability"], reverse=True):
        expanded = protected.union(proposal["positions"])
        if len(active) - len(expanded) >= budget:
            protected = expanded
            accepted.append(proposal)
        else:
            rejected.append(proposal)
    unprotected = [row for row in decisions if row["absolute_position"] not in protected]
    selected = sorted(unprotected, key=lambda row: (-row["selection_score"],
                                                    row["absolute_position"]))[:budget]
    if len(selected) != budget:
        raise ValueError("Protection failed to preserve the scheduled transfer count")
    return selected, protected, accepted, rejected


def audit_step(records, tokenizer, threshold, max_span):
    decisions = records["commit_decision"]
    selected = {row["absolute_position"] for row in decisions if row["selected"]}
    active = {row["absolute_position"] for row in decisions}
    budget = next((row["scheduled_transfer_count"] for row in decisions), 0)
    proposals = span_proposals(records, tokenizer, threshold, max_span)
    counterfactual, protected, accepted, rejected = select_with_protection(decisions, proposals)
    counterfactual_positions = {row["absolute_position"] for row in counterfactual}
    for proposal in proposals:
        proposal["conflicting_commits"] = sorted(selected.intersection(proposal["positions"]))
    remaining = len(active - protected)
    return {"budget": budget, "active_count": len(active),
            "protected_positions": sorted(protected),
            "accepted_proposals": len(accepted), "rejected_proposals": len(rejected),
            "original_selected_positions": sorted(selected),
            "counterfactual_selected_positions": sorted(counterfactual_positions),
            "counterfactual_selected_tokens": [
                {"position": row["absolute_position"], "token": row["predicted_token"],
                 "score": row["selection_score"]} for row in counterfactual],
            "displaced_original_commits": sorted(selected.intersection(protected)),
            "can_fill_budget_elsewhere": remaining >= budget,
            "proposals": proposals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-study-id", default=None,
                        help="Trace study; defaults to study-id when it has manifests")
    parser.add_argument("--source-probability-threshold", type=float, default=0.4)
    parser.add_argument("--max-span", type=int, default=3)
    args = parser.parse_args()
    default_source = "ctca-gsm8k4-benchmark-replay-20261029"
    source_study_id = args.source_study_id or (
        args.study_id if list((ROOT / ".logs" / args.study_id).glob("doc*.json"))
        else default_source
    )
    for name in (args.study_id, source_study_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            parser.error("study IDs must be simple directory names")
    if not 0 < args.source_probability_threshold <= 1 or args.max_span < 2:
        parser.error("threshold must be in (0,1] and max-span at least two")
    source = ROOT / ".logs" / source_study_id
    destination = ROOT / ".logs" / args.study_id
    destination.mkdir(parents=True, exist_ok=True)
    tokenizers = {name: get_tokenizer(ModelArguments(
        model_name_or_path=MODEL_PATHS[name], dtype="bfloat16"))
                  for name in ("a", "b")}
    report = {"source_study_id": source_study_id,
              "source_probability_threshold": args.source_probability_threshold,
              "max_span": args.max_span,
              "method": "offline original-trace protection audit; no counterfactual generation",
              "docs": {}}
    for manifest_path in sorted(source.glob("doc*.json")):
        if not re.fullmatch(r"doc\d+\.json", manifest_path.name):
            continue
        doc = int(manifest_path.stem[3:])
        manifest = json.loads(manifest_path.read_text())
        trace = next(ROOT / path.lstrip("/") for path in manifest["traces"]
                     if path.endswith("-ensemble.jsonl"))
        master = re.search(rf"doc{doc}-([ab])-ensemble", trace.name).group(1)
        steps = {}
        counts = Counter()
        for step, records in trace_steps(trace):
            entry = audit_step(records, tokenizers[master],
                               args.source_probability_threshold, args.max_span)
            if not entry["proposals"]:
                continue
            key = f"{step[0]}.{step[1]}"
            steps[key] = entry
            counts["steps_with_proposals"] += 1
            counts["proposals"] += len(entry["proposals"])
            counts["accepted_proposals"] += entry["accepted_proposals"]
            counts["rejected_proposals"] += entry["rejected_proposals"]
            counts["displaced_commits"] += len(entry["displaced_original_commits"])
            counts["cannot_fill_budget_elsewhere"] += not entry["can_fill_budget_elsewhere"]
        report["docs"][str(doc)] = {"master": master, "counts": dict(counts), "steps": steps}
        print(f"doc{doc}: {dict(counts)}", flush=True)
    output = destination / "reservation_audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
