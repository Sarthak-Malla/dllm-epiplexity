"""Sweep anchor temperature for selected CTCA projection errors.

Run after activating the dllm environment:
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/anchor_temperature_sweep.py --device cpu

Only the original top-64 destinations and explicit same-text candidates are
compared. This is not a full-vocabulary rebuild or a decoder replay.
"""

import json
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from dllm.pipelines.tse.ctca.relative import collect_relative_anchor_ids
from scripts.tse.ctca.projection_temperature_sweep import (
    CASES,
    ROOT,
    STUDY_ID,
    decoded,
    load_cache,
    load_event,
    load_tokenizers,
    single_id,
)


ANCHOR_TEMPERATURES = (0.003, 0.01, 0.03, 0.1)
PROJECTION_TEMPERATURES = (0.05, 0.1, 0.2)
DESIRED = {
    "doc23-centimeters": (" cent", " cm"),
    "doc11-period-newline": (".", "\n"),
    "doc3-period-newline": (".", "\n"),
    "doc6-toul": (" T", "oul"),
}
MODEL_NAMES = {
    "a": "GSAI-ML/LLaDA-8B-Instruct",
    "b": "Dream-org/Dream-v0-Instruct-7B",
}
EMBEDDING_KEYS = {
    "a": "model.transformer.wte.weight",
    "b": "model.embed_tokens.weight",
}


def load_embeddings(role):
    path = Path(snapshot_download(MODEL_NAMES[role], local_files_only=True))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    key = EMBEDDING_KEYS[role]
    with safe_open(path / index["weight_map"][key], framework="pt", device="cpu") as weights:
        return weights.get_tensor(key)


def cosine_to_anchors(embeddings, token_ids, anchor_ids, device):
    tokens = F.normalize(embeddings.index_select(0, token_ids).to(device).float(), dim=-1)
    anchors = F.normalize(embeddings.index_select(0, anchor_ids).to(device).float(), dim=-1)
    return tokens @ anchors.T


@torch.no_grad()
def evaluate_case(case, tokenizers, embeddings, device):
    label, path, master_role, step, position = case
    auxiliary_role = "b" if master_role == "a" else "a"
    auxiliary = tokenizers[auxiliary_role]
    master = tokenizers[master_role]
    forward, decision = load_event(path, step, position)
    source = forward["spatial_warp"]["distribution"]["top_k"][0]
    source_id = single_id(auxiliary, source["token"])
    cache = load_cache(MODEL_NAMES[auxiliary_role], MODEL_NAMES[master_role])
    original_ids = cache["tensors"]["top_indices"][source_id].long()
    original_weights = cache["tensors"]["top_weights"][source_id].float()
    selected = collect_relative_anchor_ids(
        auxiliary, master,
        auxiliary_vocab_size=embeddings[auxiliary_role].shape[0],
        master_vocab_size=embeddings[master_role].shape[0],
        num_anchors="auto", min_anchors=128,
    )
    if selected.selected_anchors != cache["metadata"]["selected_anchors"]:
        raise ValueError("Anchor set differs from cached projection")

    desired_pairs = []
    unavailable = []
    for text in DESIRED[label]:
        try:
            desired_pairs.append((text, single_id(master, text)))
        except ValueError:
            unavailable.append(text)
    candidate_ids = list(dict.fromkeys(original_ids.tolist() + [i for _, i in desired_pairs]))
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    source_cosine = cosine_to_anchors(
        embeddings[auxiliary_role], torch.tensor([source_id]),
        selected.auxiliary_anchor_ids, device,
    )
    master_cosine = cosine_to_anchors(
        embeddings[master_role], candidate_tensor,
        selected.master_anchor_ids, device,
    )
    master_probabilities = {
        item["token"]: float(item["probability"])
        for item in forward["master"]["distribution"]["top_k"]
    }
    result = {
        "case": label, "trace": str(ROOT / path), "step": list(step),
        "position": position, "source": source, "observed_commit": decision["predicted_token"],
        "candidate_count": len(candidate_ids), "candidate_scope": "cached_top64_plus_desired",
        "unavailable_desired_single_tokens": unavailable, "anchor_count": selected.selected_anchors,
        "temperature_grid": [],
    }
    for anchor_temperature in ANCHOR_TEMPERATURES:
        source_profile = F.normalize(torch.softmax(source_cosine / anchor_temperature, dim=-1), dim=-1)
        master_profiles = F.normalize(torch.softmax(master_cosine / anchor_temperature, dim=-1), dim=-1)
        similarities = (source_profile @ master_profiles.T).squeeze(0)
        row = {
            "anchor_temperature": anchor_temperature,
            "candidate_top": decoded(master, candidate_ids[int(similarities.argmax())]),
            "desired": {
                text: {
                    "similarity": float(similarities[candidate_ids.index(token_id)]),
                    "rank_among_candidates": int((similarities > similarities[candidate_ids.index(token_id)]).sum()) + 1,
                }
                for text, token_id in desired_pairs
            },
            "projection_temperatures": [],
        }
        if anchor_temperature == 0.01:
            restored = torch.softmax(similarities[: len(original_ids)] / 0.05, dim=0).cpu()
            row["baseline_top64_weight_max_abs_error"] = float((restored - original_weights).abs().max())
            if row["baseline_top64_weight_max_abs_error"] > 0.01:
                raise ValueError("Recomputed baseline differs from cached projection")
        for projection_temperature in PROJECTION_TEMPERATURES:
            weights = torch.softmax(similarities / projection_temperature, dim=0).cpu()
            destinations = {}
            for token_id, weight in zip(candidate_ids, weights.tolist()):
                text = decoded(master, token_id)
                destinations[text] = destinations.get(text, 0.0) + weight
            candidate_texts = set(master_probabilities) | set(destinations)
            scores = sorted(
                ((0.5 * master_probabilities.get(text, 0.0)
                  + 0.5 * float(source["probability"]) * destinations.get(text, 0.0), text)
                 for text in candidate_texts),
                reverse=True,
            )
            row["projection_temperatures"].append({
                "projection_temperature": projection_temperature,
                "estimated_fused_top": scores[0][1],
                "estimated_fused_margin": scores[0][0] - scores[1][0],
                "source_projected_top": max(destinations.items(), key=lambda item: item[1])[0],
            })
        result["temperature_grid"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for local analysis")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(24)
    tokenizers = load_tokenizers()
    embeddings = {role: load_embeddings(role) for role in ("a", "b")}
    results = [evaluate_case(case, tokenizers, embeddings, device) for case in CASES]
    output = ROOT / ".logs" / STUDY_ID / "restricted_anchor_temperature_grid.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    for case in results:
        print(case["case"], "observed", repr(case["observed_commit"]))
        for row in case["temperature_grid"]:
            print("  anchor", row["anchor_temperature"], "source map", repr(row["candidate_top"]),
                  "fused", [x["estimated_fused_top"] for x in row["projection_temperatures"]])


if __name__ == "__main__":
    main()
