"""Sweep cached CTCA projection temperature at selected full-trace decisions.

Run after activating the dllm environment:
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/projection_temperature_sweep.py

This is a top-source-token counterfactual, not a full decoder replay.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from dllm.pipelines.dream.models.tokenization_dream import DreamTokenizer


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "ctca-gsm8k4-anchor-projection-temp-sweep-20261104"
CASES = (
    ("doc23-centimeters", ".logs/ctca-gsm8k4-benchmark-replay-20261029/352946/doc23-b-ensemble.jsonl", "b", (0, 2), 69),
    ("doc11-period-newline", ".logs/ctca-gsm8k4-benchmark-replay-20261029/352946/doc11-a-ensemble.jsonl", "a", (1, 3), 541),
    ("doc3-period-newline", ".logs/ctca-gsm8k2-gain-replay-20261104/367906/doc3-a-ensemble.jsonl", "a", (0, 7), 500),
    ("doc6-toul", ".logs/ctca-gsm8k2-gain-replay-20261104/367906/doc6-b-ensemble.jsonl", "b", (0, 6), 88),
)
TEMPERATURES = (0.01, 0.025, 0.05, 0.1, 0.2)


def load_tokenizers():
    llada = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, local_files_only=True
    )
    dream_path = snapshot_download("Dream-org/Dream-v0-Instruct-7B", local_files_only=True)
    dream = DreamTokenizer.from_pretrained(dream_path, local_files_only=True)
    return {"a": llada, "b": dream}


def decoded(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def single_id(tokenizer, surface):
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(ids) == 1 and decoded(tokenizer, ids[0]) == surface:
        return int(ids[0])
    matches = [i for i in range(len(tokenizer)) if decoded(tokenizer, i) == surface]
    if len(matches) != 1:
        raise ValueError(f"Expected one token for {surface!r}, found {matches[:8]}")
    return matches[0]


def load_cache(auxiliary_name, master_name):
    matches = []
    for path in (ROOT / ".cache/ctca").glob("relative_sparse_topk_*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload["metadata"]
        if (meta["auxiliary_id"] == auxiliary_name
                and meta["master_id"] == master_name
                and meta.get("anchor_temperature") == 0.01
                and meta.get("projection_temperature") == 0.05
                and meta.get("projection_top_k") == 64
                and meta.get("selected_anchors") == meta.get("total_anchors")
                and meta.get("selected_anchors", 0) > 0):
            matches.append((path, payload))
    if not matches:
        raise FileNotFoundError(
            f"No full-anchor baseline projection for {auxiliary_name} -> {master_name}"
        )
    if len(matches) != 1:
        raise ValueError(f"Ambiguous baseline projection caches: {[str(p) for p, _ in matches]}")
    return matches[0][1]


def load_event(path, step, position):
    forward = None
    decision = None
    for line in (ROOT / path).open():
        row = json.loads(line)
        if tuple(row.get("step", ())) != step:
            continue
        if row["event"] == "forward_position" and row["master"]["absolute_position"] == position:
            forward = row
        elif row["event"] == "commit_decision" and row["absolute_position"] == position:
            decision = row
    if forward is None or decision is None:
        raise ValueError(f"Missing event at {path}:{step}:{position}")
    return forward, decision


def evaluate_case(case, tokenizers):
    label, path, master_role, step, position = case
    auxiliary_role = "b" if master_role == "a" else "a"
    auxiliary = tokenizers[auxiliary_role]
    master = tokenizers[master_role]
    forward, decision = load_event(path, step, position)
    source = forward["spatial_warp"]["distribution"]["top_k"][0]
    source_id = single_id(auxiliary, source["token"])
    payload = load_cache(
        "GSAI-ML/LLaDA-8B-Instruct" if auxiliary_role == "a" else "Dream-org/Dream-v0-Instruct-7B",
        "GSAI-ML/LLaDA-8B-Instruct" if master_role == "a" else "Dream-org/Dream-v0-Instruct-7B",
    )
    indices = payload["tensors"]["top_indices"][source_id]
    weights = payload["tensors"]["top_weights"][source_id]
    source_spelling = auxiliary.convert_ids_to_tokens(source_id)
    shared_id = master.get_vocab().get(source_spelling)
    exact_shared = (
        shared_id is not None
        and source_id not in set(auxiliary.all_special_ids)
        and shared_id not in set(master.all_special_ids)
    )
    master_probabilities = {
        item["token"]: float(item["probability"])
        for item in forward["master"]["distribution"]["top_k"]
    }
    results = []
    for temperature in TEMPERATURES:
        if exact_shared:
            destinations = {decoded(master, shared_id): 1.0}
        else:
            ratio = payload["metadata"]["projection_temperature"] / temperature
            adjusted = torch.softmax(weights.clamp_min(1e-45).log() * ratio, dim=0)
            destinations = {}
            for token_id, mass in zip(indices.tolist(), adjusted.tolist()):
                token = decoded(master, token_id)
                destinations[token] = destinations.get(token, 0.0) + mass
        candidate_tokens = set(master_probabilities) | set(destinations)
        scores = sorted(
            ((0.5 * master_probabilities.get(token, 0.0)
              + 0.5 * float(source["probability"]) * destinations.get(token, 0.0), token)
             for token in candidate_tokens),
            reverse=True,
        )
        omitted_bound = 0.5 * (
            1.0 - float(source["probability"])
            + float(forward["master"]["distribution"]["remaining_mass"])
        )
        results.append({
            "projection_temperature": temperature,
            "estimated_fused_top": scores[0][1],
            "estimated_fused_top_probability": scores[0][0],
            "runner_up": scores[1][1],
            "margin": scores[0][0] - scores[1][0],
            "omitted_mass_score_bound": omitted_bound,
            "top_result_robust_to_omitted_mass": scores[0][0] - scores[1][0] > omitted_bound,
            "projected_source_top": max(destinations.items(), key=lambda item: item[1])[0],
            "source_top_destination_mass": max(destinations.values()),
        })
    return {
        "case": label, "trace": str(ROOT / path), "step": list(step), "position": position,
        "source_token": source["token"], "source_probability": source["probability"],
        "source_id": source_id, "source_raw_spelling": source_spelling,
        "exact_shared": exact_shared, "observed_commit": decision["predicted_token"],
        "master_top64_mass": forward["master"]["distribution"]["top_k_mass"],
        "spatial_top64_mass": forward["spatial_warp"]["distribution"]["top_k_mass"],
        "results": results,
    }


def main():
    tokenizers = load_tokenizers()
    results = [evaluate_case(case, tokenizers) for case in CASES]
    output = ROOT / ".logs" / STUDY_ID / "cached_projection_temperature.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    for row in results:
        print(f"{row['case']}: source {row['source_token']!r} p={row['source_probability']:.4f}, observed {row['observed_commit']!r}")
        for value in row["results"]:
            print(f"  T={value['projection_temperature']:.3f} -> {value['estimated_fused_top']!r} "
                  f"(margin {value['margin']:.4f}, omitted bound {value['omitted_mass_score_bound']:.4f}, "
                  f"robust={value['top_result_robust_to_omitted_mass']})")


if __name__ == "__main__":
    main()
