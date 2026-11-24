"""Audit archived paired text commits on the cscc-cpu-p partition.

    sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_archived_reconciliation_conflicts_slurm.sh

Load the pinned tokenizers and saved traces. All scoring and continuations
used for policy evaluation are archived; this program loads no model weights.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tie.commit_actions import contextual_ids
from dllm.pipelines.tie.exact_text_fusion import surrounding_text
from dllm.pipelines.tie.relay import Vocabulary, transfer_blocks
from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_tokenizer
from scripts.tse.ctca.independent_reconciliation import CHECKPOINTS


STUDY = "ctca-llada-dream-real-conflict-audit-20261124"
SOURCE_ROOT = ROOT / ".logs" / "mdlm-gsm8k-llada-dream-commit-actions"
RUNS = {
    "block_end_pilot": ("pilot/shared_audit/423957", 32, (16, 48, 80)),
    "midblock_pilot": ("pilot/shared_audit_midblock/423992", 32, (12, 44, 76)),
    "midblock_fit": ("fit/shared_audit_midblock/424098", 168, (12, 44, 76)),
    "midblock_validation": ("validation/shared_audit_midblock/424099", 100, (12, 44, 76)),
}


def choose_from_margins(owner_margin, peer_margin):
    """Scores are mean log-probability differences for peer text minus owner text."""
    joint = "peer" if owner_margin + peer_margin > 0 else "owner"
    if min(owner_margin, peer_margin) > 0:
        unanimous = "peer"
    elif max(owner_margin, peer_margin) < 0:
        unanimous = "owner"
    else:
        unanimous = "defer"
    return joint, unanimous


def verify_replay(event, check, baseline, owner):
    """Verify that the two continuations differ in this scheduled gap alone."""
    branch = event["alternatives"]["peer_text"]
    gap = check["gap"]
    slots, ids = event["default_slots"], event["default_ids"]
    if not gap or gap != list(range(gap[0], gap[-1] + 1)):
        raise ValueError("Expected a nonempty, contiguous gap")
    if len(slots) != len(ids) or len(set(slots)) != len(slots) or not set(gap) <= set(slots):
        raise ValueError("Gap must be part of the unique scheduled owner slots")
    original = dict(zip(slots, ids))
    if [original[pos] for pos in gap] != check["owner_ids_a"] or len(check["owner_ids_b"]) != len(gap):
        raise ValueError("The saved owner IDs do not match the annotated gap")
    updated = {**original, **dict(zip(gap, check["owner_ids_b"]))}
    if branch["slots"] != slots or branch["ids"] != [updated[pos] for pos in slots] or original == updated:
        raise ValueError("Peer replay did not change just the annotated gap text")
    if branch["owner_score"] != check["owner_margin"] or branch["peer_score"] != check["peer_margin"]:
        raise ValueError("Replay margins disagree with the verified text check")

    alternate = branch["completion"]
    block_offset = sum(len(block) for block in event["draft"][:event["block"]])
    for completion, writes in ((baseline, original), (alternate, updated)):
        if completion["member"] != owner or not isinstance(completion["correct"], bool):
            raise ValueError("Missing a Boolean label on the owner continuation")
        if any(completion["reasoning_ids"][block_offset + pos] != token for pos, token in writes.items()):
            raise ValueError("The saved continuation did not preserve the scheduled write")
    forwards = baseline["forward_calls"]
    if forwards != alternate["forward_calls"] or forwards <= event["step"]:
        raise ValueError("Continuation budgets differ or contain no subsequent forwards")
    return alternate["correct"], forwards - event["step"]


def verify_tokens(event, check, source, target):
    """Recheck exact text boundaries and token widths in the pinned vocabularies."""
    blocks = event["draft"]
    index = event["block"]
    block = blocks[index]
    lo, hi = check["gap"][0], check["gap"][-1] + 1
    if lo <= 0 or hi >= len(block) or block[lo:hi] != [source.mask_id] * (hi - lo):
        raise ValueError("Gap is not a bounded native mask run")
    left_ids, right_ids = surrounding_text(block, lo, hi, source)
    if not left_ids or not right_ids:
        raise ValueError("Gap is missing a visible boundary")
    left, right = source.decode(left_ids), source.decode(right_ids)
    for suffix in ("a", "b"):
        text = check[suffix]
        if not text or "\ufffd" in left + text + right:
            raise ValueError("Candidate or boundary contains invalid text")
        if contextual_ids(source, left, text, right) != check[f"owner_ids_{suffix}"]:
            raise ValueError("Owner contextual IDs differ from the archive")
        if contextual_ids(target, left, text, right) != check[f"peer_ids_{suffix}"]:
            raise ValueError("Peer contextual IDs differ from the archive")
    part = [*blocks[:index], block[:lo], block[lo:hi], block[hi:], *blocks[index + 1:]]
    transferred = transfer_blocks(part, source, target)
    target_left, target_right = target.encode(left), target.encode(right)
    if (transferred[index][-len(target_left):] != target_left
            or transferred[index + 2][:len(target_right)] != target_right):
        raise ValueError("Native transfer changed one of the two gap boundaries")
    owner_width, peer_width = len(check["owner_ids_a"]), len(check["peer_ids_a"])
    if owner_width != hi - lo or peer_width != len(check["peer_ids_b"]):
        raise ValueError("The alternatives require different native widths")
    return left, right, owner_width, peer_width


def candidate(doc_id, owner, peer, baseline, event, check, vocabularies):
    peer_correct, continuation = verify_replay(event, check, baseline, owner)
    left, right, owner_width, peer_width = verify_tokens(event, check, vocabularies[owner], vocabularies[peer])
    owner_margin = check["owner_margin"] / owner_width
    peer_margin = check["peer_margin"] / peer_width
    if not all(map(math.isfinite, (owner_margin, peer_margin))):
        raise ValueError("Nonfinite saved margin")
    joint, unanimous = choose_from_margins(owner_margin, peer_margin)
    widths = {owner: owner_width, peer: peer_width}
    return {
        "doc_id": doc_id, "owner": owner, "step": event["step"], "block": event["block"],
        "gap": check["gap"], "left": left, "right": right,
        "texts": {"owner": check["a"], "peer": check["b"]},
        "native_widths": [widths[name] for name in CHECKPOINTS],
        "mean_log_margins_peer_minus_owner": {owner: owner_margin, peer: peer_margin},
        "correct": {"owner": baseline["correct"], "peer": peer_correct},
        "joint_choice": joint, "unanimous_choice": unanimous,
        "pair_scoring_forwards": {owner: 2 * owner_width, peer: 2 * peer_width},
        "additional_verification_after_peer_proposal": 2 * owner_width + peer_width,
        "saved_continuation_forwards_per_choice": continuation,
    }


def summarize(candidates):
    counts = Counter({key: 0 for key in (
        "owner_correct", "peer_correct", "peer_gains", "peer_losses", "both_correct", "both_wrong",
        "joint_correct", "joint_peer_switches", "unanimous_correct", "unanimous_wrong", "unanimous_deferred",
        "correct_joint_choices_deferred", "wrong_joint_choices_deferred", "deferred_wrong_with_correct_alternative",
        "unanimous_or_keep_owner_correct", "pair_scoring_forwards", "additional_verification_forwards",
    )})
    for row in candidates:
        labels = row["correct"]
        owner, peer = labels["owner"], labels["peer"]
        joint, unanimous = row["joint_choice"], row["unanimous_choice"]
        counts["owner_correct"] += owner
        counts["peer_correct"] += peer
        counts["peer_gains"] += peer and not owner
        counts["peer_losses"] += owner and not peer
        counts["both_correct"] += owner and peer
        counts["both_wrong"] += not owner and not peer
        counts["joint_correct"] += labels[joint]
        counts["joint_peer_switches"] += joint == "peer"
        if unanimous == "defer":
            counts["unanimous_deferred"] += 1
            counts["correct_joint_choices_deferred" if labels[joint] else "wrong_joint_choices_deferred"] += 1
            counts["deferred_wrong_with_correct_alternative"] += not labels[joint] and any(labels.values())
        else:
            counts["unanimous_correct" if labels[unanimous] else "unanimous_wrong"] += 1
        counts["unanimous_or_keep_owner_correct"] += labels["owner" if unanimous == "defer" else unanimous]
        counts["pair_scoring_forwards"] += sum(row["pair_scoring_forwards"].values())
        counts["additional_verification_forwards"] += row["additional_verification_after_peer_proposal"]
    return {"pairs": len(candidates), "questions": len({row["doc_id"] for row in candidates}), **counts}


def audit_run(path, expected_documents, steps, vocabularies):
    candidates = []
    exclusions = []
    documents = set()
    coverage = {name: Counter() for name in vocabularies}
    scoring_forwards = Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = row["doc_id"]
            if row["mode"] != "shared_audit" or doc_id in documents or row["owners"].keys() != vocabularies.keys():
                raise ValueError(f"Unexpected archive record at {path}, doc {doc_id}")
            documents.add(doc_id)
            for owner, branch in row["owners"].items():
                peer = next(name for name in vocabularies if name != owner)
                counts = coverage[owner]
                scoring_forwards.update({name: cost[0] for name, cost in branch["proposal_scoring_cost"].items()})
                events = branch["checkpoints"]
                if tuple(event["step"] for event in events) != steps:
                    raise ValueError(f"Unexpected checkpoints at {path}, doc {doc_id}, owner {owner}")
                for event in events:
                    counts["checkpoints"] += 1
                    counts["unaligned_canvases"] += "unaligned" in event
                    counts["checkpoints_with_text_checks"] += bool(event["text_checks"])
                    valid = [check for check in event["text_checks"] if "rejected" not in check]
                    counts["rejected_text_checks"] += len(event["text_checks"]) - len(valid)
                    disagreements = [check for check in valid if check["a"] != check["b"]]
                    counts["text_agreements"] += len(valid) - len(disagreements)
                    counts["text_disagreements"] += len(disagreements)
                    if len(disagreements) != int("peer_text" in event["alternatives"]):
                        counts["excluded_replays"] += 1
                        exclusions.append({"doc_id": doc_id, "owner": owner, "step": event["step"],
                                           "reason": "Disagreement lacks a unique paired replay"})
                        continue
                    if disagreements:
                        try:
                            pair = candidate(doc_id, owner, peer, branch["baseline"], event, disagreements[0], vocabularies)
                        except (ValueError, IndexError, KeyError) as error:
                            counts["excluded_replays"] += 1
                            exclusions.append({"doc_id": doc_id, "owner": owner, "step": event["step"],
                                               "reason": str(error)})
                        else:
                            counts["verified_replays"] += 1
                            candidates.append(pair)
    if len(documents) != expected_documents:
        raise ValueError(f"Expected {expected_documents} questions in {path}, found {len(documents)}")
    pairs = sorted({tuple(row["native_widths"]) for row in candidates})
    return {
        "source": str(path), "documents": len(documents), "steps": steps, "coverage": coverage,
        "excluded_replays": exclusions,
        "all_archived_proposal_scoring_forwards": scoring_forwards, "overall": summarize(candidates),
        "by_owner": {name: summarize([row for row in candidates if row["owner"] == name]) for name in vocabularies},
        "by_width": [{"native_widths": pair, **summarize([row for row in candidates if tuple(row["native_widths"]) == pair])}
                     for pair in pairs],
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    vocabularies = {
        name: Vocabulary.from_tokenizer(get_tokenizer(ModelArguments(model_name_or_path=str(path))))
        for name, path in CHECKPOINTS.items()
    }
    report = {"study_id": STUDY, "native_width_order": list(CHECKPOINTS), "runs": {}}
    for name, (directory, documents, steps) in RUNS.items():
        path = SOURCE_ROOT / directory / "trajectory.jsonl"
        results = audit_run(path, documents, steps, vocabularies)
        report["runs"][name] = results
        print(f"{name}: {json.dumps(results['overall'], sort_keys=True)}", flush=True)
    output = args.output or ROOT / ".logs" / STUDY / os.environ.get("SLURM_JOB_ID", "manual") / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
