"""Compare sparse position matching on archived LLaDA–Dream probabilities.

    source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
    conda activate dllm
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/offline_position_topk.py

This CPU audit uses saved CTCA traces, tokenizer snapshots, and anchor profiles.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tse.ctca.cache import tokenizer_vocab_signature
from scripts.tse.ctca.projection_temperature_sweep import load_tokenizers
from scripts.tse.sandbox.ctca_position_correspondence import MIN_SIMILARITY, monotonic_pairs
from scripts.tse.smoke import _decode_token


STUDY = "ctca-gsm8k6-llada-dream-position-topk-20261123"
REPLAYS = {
    "ctca-gsm8k4-benchmark-replay-20261029": ((23, "b"), (18, "b"), (9, "b"), (11, "a")),
    "ctca-gsm8k2-gain-replay-20261104": ((3, "a"), (6, "b")),
}
CHECKPOINTS = tuple((block, step) for block in (0, 1, 4, 8, 15) for step in (0, 4))
TOP_K = (8, 16, 32, 64)
RADIUS = 4
PROFILE_CACHE = (
    ROOT / ".cache" / "ctca" /
    "relative_distribution_profiles_eddfdfad3eb384ca41d4da5aa4e762a389b8b18a40c8949f5fed788ffab58522.pt"
)


def load_profiles(cache_path):
    """Use native vocab IDs only when the tokenizer and cache signatures agree."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload["metadata"]
    profiles = {"a": payload["tensors"]["auxiliary_profiles"], "b": payload["tensors"]["master_profiles"]}
    tokenizers = load_tokenizers()
    for role, key in (("a", "auxiliary_vocab"), ("b", "master_vocab")):
        if tokenizer_vocab_signature(tokenizers[role]) != metadata[key]:
            raise ValueError(f"Token IDs for model {role} do not match {cache_path}")
    return profiles, tokenizers, metadata


def unique_decoded_ids(tokenizer, vocabulary_size):
    """Reverse only unambiguous individual token decodes from the trace schema."""
    ids_by_text = {}
    for token_id in set(tokenizer.get_vocab().values()):
        if token_id < vocabulary_size:
            text = _decode_token(tokenizer, token_id)
            ids_by_text[text] = -1 if text in ids_by_text else token_id
    return ids_by_text


def read_checkpoints(path):
    event_types = {"forward_position", "auxiliary_masked_position"}
    events = defaultdict(lambda: {"forward_position": [], "auxiliary_masked_position": []})
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            step = tuple(row.get("step", ()))
            if step in CHECKPOINTS and row["event"] in event_types:
                if row["batch_index"] != 0:
                    raise ValueError(f"Expected a single-sample trace at {path}")
                events[step][row["event"]].append(row)
    return events


def projected_rows(distributions, ids_by_text, profiles):
    """Mix the cached profiles of top-k tokens at their original probabilities."""
    if any(len(row["top_k"]) < TOP_K[-1] or row["vocabulary_size"] != profiles.shape[0]
           for row in distributions):
        raise ValueError("Distributions and the cached profiles must share vocabulary and top-k coverage")
    top_rows = [row["top_k"][:TOP_K[-1]] for row in distributions]
    native_mass = torch.tensor([[entry["probability"] for entry in row] for row in top_rows])
    token_ids = torch.tensor([[ids_by_text.get(entry["token"], -1) for entry in row] for row in top_rows])
    usable_mass = native_mass.masked_fill(token_ids < 0, 0)
    token_ids.clamp_min_(0)
    mixtures = defaultdict(list)
    for ids, values in zip(token_ids.split(32), usable_mass.split(32)):
        mixture = torch.zeros(ids.shape[0], profiles.shape[1])
        previous = 0
        for k in TOP_K:
            gathered = profiles[ids[:, previous:k]].float()
            contribution = torch.bmm(values[:, None, previous:k], gathered).squeeze(1)
            mixture = mixture + contribution
            mixtures[k].append(F.normalize(mixture, dim=-1))
            previous = k
    return {k: torch.cat(rows) for k, rows in mixtures.items()}, native_mass.cumsum(1), usable_mass.cumsum(1)


def score_checkpoint(events, profiles, ids_by_text, master_role, radius=RADIUS):
    peer_role = "b" if master_role == "a" else "a"
    master_rows = sorted(events["forward_position"], key=lambda row: row["master"]["generation_position"])
    peer_rows = sorted(events["auxiliary_masked_position"], key=lambda row: row["generation_position"])
    master_positions = [row["master"]["generation_position"] for row in master_rows]
    peer_positions = [row["generation_position"] for row in peer_rows]
    peer_index = {position: j for j, position in enumerate(peer_positions)}
    if not master_rows or not peer_rows:
        raise ValueError("Each checkpoint must contain native distributions from both models")
    references = []
    for row in master_rows:
        contributors = row["auxiliary"]["contributors"]
        if len(contributors) != 1 or contributors[0]["overlap"] != 1:
            raise ValueError("Expected an exact, unique scaffold mask correspondence")
        references.append(peer_index[contributors[0]["generation_position"]])
    if len(set(references)) != len(references) or sorted(references) != references:
        raise ValueError("The scaffold reference must be strictly monotonic")
    master, master_mass, master_usable = projected_rows(
        [row["master"]["distribution"] for row in master_rows], ids_by_text[master_role], profiles[master_role]
    )
    peer, peer_mass, peer_usable = projected_rows(
        [row["distribution"] for row in peer_rows], ids_by_text[peer_role], profiles[peer_role]
    )
    offsets = [peer_positions[j] - i for i, j in zip(master_positions, references)]
    distances = (torch.tensor(master_positions)[:, None] - torch.tensor(peer_positions)).abs()
    in_window = distances <= radius
    same_index = [(i, peer_index[pos]) for i, pos in enumerate(master_positions) if pos in peer_index]
    result = {"reference_offsets": offsets, "all_pairs": distances.numel(),
              "pairs_in_window": int(in_window.sum()), "top_k": {}}
    methods = defaultdict(dict)
    master_indices = torch.arange(len(master_rows))
    reference_indices = torch.tensor(references)
    references_in_window = in_window[master_indices, reference_indices]
    for k in TOP_K:
        scores = master[k] @ peer[k].T
        allowed = scores > MIN_SIMILARITY
        banded = allowed & in_window
        best_scores, best_positions = scores.masked_fill(~banded, -torch.inf).max(dim=1)
        best_before_threshold = scores.masked_fill(~in_window, -torch.inf).argmax(dim=1)
        methods[k] = {
            "same_index": dict(same_index),
            f"row_argmax_window{radius}": {i: j for i, j in enumerate(best_positions.tolist())
                                          if best_scores[i] > -torch.inf},
            f"monotonic_window{radius}": dict(monotonic_pairs(scores, banded)),
            "monotonic_full": dict(monotonic_pairs(scores, allowed)),
        }
        result["top_k"][k] = {
            "references_above_threshold": int(allowed[master_indices, reference_indices].sum()),
            "references_best_in_window_before_threshold": int(
                ((best_before_threshold == reference_indices) & references_in_window).sum()
            ),
            "master_native_mass": master_mass[:, k - 1].tolist(),
            "peer_native_mass": peer_mass[references, k - 1].tolist(),
            "master_profiled_mass": master_usable[:, k - 1].tolist(),
            "peer_profiled_mass": peer_usable[references, k - 1].tolist(),
        }
    for k, entries in methods.items():
        scores_by_method = {}
        for method, pairs in entries.items():
            correct_offsets = Counter(offsets[i] for i, j in enumerate(references) if pairs.get(i) == j)
            correct = sum(correct_offsets.values())
            scores_by_method[method] = {
                "correct": correct,
                "correct_shifted": correct - correct_offsets[0],
                "correct_by_offset": dict(sorted(correct_offsets.items())),
                "matched": len(pairs),
                "same_as_top64": sum(pairs.get(i) == methods[TOP_K[-1]][method].get(i) for i in range(len(references))),
            }
        result["top_k"][k]["methods"] = scores_by_method
    return result


def summarize(checkpoints, radius=RADIUS):
    offsets = [offset for row in checkpoints for offset in row["reference_offsets"]]
    summary = {
        "checkpoints": len(checkpoints), "reference_pairs": len(offsets),
        "reference_pairs_shifted": sum(offset != 0 for offset in offsets),
        "reference_pairs_in_window": sum(abs(offset) <= radius for offset in offsets),
        "all_pairs": sum(row["all_pairs"] for row in checkpoints),
        "pairs_in_window": sum(row["pairs_in_window"] for row in checkpoints),
        "reference_offset_counts": dict(sorted(Counter(offsets).items())), "top_k": {},
    }
    for k in TOP_K:
        rows = [row["top_k"][k] for row in checkpoints]
        metrics = {key: sum(row[key] for row in rows) for key in
                   ("references_above_threshold", "references_best_in_window_before_threshold")}
        for key in ("master_native_mass", "peer_native_mass", "master_profiled_mass", "peer_profiled_mass"):
            values = torch.tensor([value for row in rows for value in row[key]])
            metrics[key] = {"mean": float(values.mean()), "p10": float(values.quantile(0.1))}
        metrics["methods"] = {
            method: {key: sum(row["methods"][method][key] for row in rows)
                     for key in ("correct", "correct_shifted", "matched", "same_as_top64")}
            for method in rows[0]["methods"]
        }
        for method, counts in metrics["methods"].items():
            correct_offsets = Counter()
            for row in rows:
                correct_offsets.update(row["methods"][method]["correct_by_offset"])
            counts["correct_by_offset"] = dict(sorted(correct_offsets.items()))
        summary["top_k"][k] = metrics
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=PROFILE_CACHE)
    parser.add_argument("--radius", type=int, default=RADIUS, help="Maximum absolute gap between generation positions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.radius < 0:
        parser.error("--radius must be non-negative")
    torch.set_num_threads(2)
    profiles, tokenizers, metadata = load_profiles(args.profiles)
    ids_by_text = {role: unique_decoded_ids(tokenizers[role], table.shape[0]) for role, table in profiles.items()}
    report = {"study_id": STUDY, "top_k": TOP_K, "checkpoint_grid": CHECKPOINTS, "cases": [],
              "anchor_temperature": metadata["anchor_temperature"], "anchors": metadata["selected_anchors"],
              "minimum_cosine_exclusive": MIN_SIMILARITY, "window_radius": args.radius,
              "profile_cache": str(args.profiles.resolve())}
    checkpoints = []
    for replay, cases in REPLAYS.items():
        for doc, role in cases:
            manifest = json.loads((ROOT / ".logs" / replay / f"doc{doc}.json").read_text())
            path = ROOT / next(trace for trace in manifest["traces"] if trace.endswith("-ensemble.jsonl")).lstrip("/")
            events = read_checkpoints(path)
            if set(events) != set(CHECKPOINTS):
                raise ValueError(f"Missing requested checkpoints in {path}: {set(CHECKPOINTS) - events.keys()}")
            rows = [{"step": step, **score_checkpoint(events[step], profiles, ids_by_text, role, args.radius)}
                    for step in CHECKPOINTS]
            print(f"doc {doc}, master {role}: scored {len(rows)} checkpoints", flush=True)
            checkpoints.extend(rows)
            report["cases"].append({"doc": doc, "master": role, "trace": str(path),
                                    "summary": summarize(rows, args.radius), "checkpoints": rows})
    report["summary"] = summarize(checkpoints, args.radius)
    report["by_master"] = {
        role: summarize([row for case in report["cases"] if case["master"] == role for row in case["checkpoints"]],
                        args.radius)
        for role in ("a", "b")
    }
    filename = "report.json" if args.radius == RADIUS else f"report-window{args.radius}.json"
    output = (args.output or ROOT / ".logs" / STUDY / filename).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
