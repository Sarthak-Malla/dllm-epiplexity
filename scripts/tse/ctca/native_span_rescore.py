"""Rescore saved CTCA candidates in both native masked-model vocabularies.

Run in a two-GPU allocation from the repository root:
    python scripts/tse/ctca/native_span_rescore.py --study-id STUDY_ID
Use --prepare-only to inspect selected checkpoints without loading models.
Scores are mean masked-token log probabilities, not joint span probabilities.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import time

import torch

from dllm.pipelines.tse import TSEConfig
from dllm.pipelines.tse.ctca.canvas import build_model_canvas_view
from dllm.pipelines.tse.loader import load_tse_models
from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_tokenizer


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINTS = ((23, 0, 0, 69), (23, 0, 1, 69), (23, 0, 2, 69),
                       (18, 0, 0, 55), (9, 1, 0, 117),
                       (11, 0, 0, 497), (11, 1, 2, 541), (11, 1, 3, 541))
MODEL_PATHS = {"a": "GSAI-ML/LLaDA-8B-Instruct",
               "b": "Dream-org/Dream-v0-Instruct-7B"}


def trace_steps(path):
    """Stream trace groups without retaining the large JSONL in memory."""
    current = None
    records = defaultdict(list)
    events = {"forward_position", "auxiliary_masked_position", "commit_decision"}
    with path.open() as stream:
        for line in stream:
            if not any(f'"event": "{event}"' in line for event in events):
                continue
            row = json.loads(line)
            if row["event"] not in events:
                continue
            step = tuple(row["step"])
            if current is not None and step != current:
                yield current, records
                records = defaultdict(list)
            current = step
            records[row["event"]].append(row)
    if current is not None:
        yield current, records


def select_candidates(records, position, limit):
    forward = next((row for row in records["forward_position"]
                    if row["master"]["absolute_position"] == position), None)
    if forward is None:
        raise ValueError(f"No forward record for position {position}")
    contributor = max(forward["auxiliary"]["contributors"],
                      key=lambda row: row["overlap"])
    auxiliary = next(row for row in records["auxiliary_masked_position"]
                     if row["generation_position"] == contributor["generation_position"])
    sources = {}
    for label, distribution in (("master", forward["master"]["distribution"]),
                                ("auxiliary", auxiliary["distribution"])):
        for rank, entry in enumerate(distribution["top_k"][:limit], 1):
            text = entry["token"]
            sources.setdefault(text, []).append({"model": label, "rank": rank,
                                                   "probability": entry["probability"]})
    return sources, forward


def native_ids(tokenizer, text, max_span):
    if not text or text.startswith("<|") or text.startswith("[Mask]"):
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not 1 <= len(ids) <= max_span or any(x in tokenizer.all_special_ids for x in ids):
        return None
    if tokenizer.decode(ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != text:
        return None
    return ids


def apply_commits(generated, records, tokenizer):
    for row in records["commit_decision"]:
        if not row["selected"]:
            continue
        ids = native_ids(tokenizer, row["predicted_token"], 1)
        if ids is None:
            raise ValueError(f"Cannot reconstruct committed token {row['predicted_token']!r} "
                             f"at step {row['step']} position {row['absolute_position']}")
        index = row["absolute_position"] - records["_prompt_length"]
        if not 0 <= index < len(generated) or generated[index] != tokenizer.mask_token_id:
            raise ValueError(f"Invalid or duplicate commit at generation position {index}")
        generated[index] = ids[0]


def score_native(model, tokenizer, input_ids, positions, model_name):
    device = next(model.parameters()).device
    ids = input_ids.unsqueeze(0).to(device)
    attention = torch.ones_like(ids)
    kwargs = {"attention_mask": attention}
    if model_name == "b":
        position_ids = attention.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention == 0, 1)
        kwargs["position_ids"] = position_ids
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(ids, **kwargs).logits
        if model_name == "b":
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        selected = logits[0, positions].float()
        valid = torch.zeros(selected.shape[-1], dtype=torch.bool, device=device)
        valid[list({int(x) for x in tokenizer.get_vocab().values()
                    if 0 <= int(x) < len(valid)})] = True
        selected = selected.masked_fill(~valid, torch.finfo(selected.dtype).min)
        log_probs = torch.log_softmax(selected, dim=-1).cpu()
    return log_probs, time.perf_counter() - start


def conditional_span_log_probability(model, tokenizer, input_ids, start, native_ids,
                                     model_name, first_log_probability):
    """Score later tokens after filling only the candidate's preceding tokens."""
    total = float(first_log_probability)
    extra_seconds = 0.0
    for offset in range(1, len(native_ids)):
        conditioned = input_ids.clone()
        conditioned[start:start + offset] = torch.tensor(native_ids[:offset])
        next_probs, seconds_spent = score_native(
            model, tokenizer, conditioned, [start + offset], model_name)
        total += float(next_probs[0, native_ids[offset]])
        extra_seconds += seconds_spent
    return total, extra_seconds


def evaluate_checkpoint(models, tokenizers, master, prompt_ids, auxiliary_prompt_ids,
                        generated, position, sources, max_span, conditional=False):
    auxiliary = "b" if master == "a" else "a"
    start = position - len(prompt_ids)
    if not 0 <= start < len(generated):
        raise ValueError("Checkpoint position is outside generated canvas")
    view = build_model_canvas_view(auxiliary_prompt_ids, generated,
                                   master_tokenizer=tokenizers[master],
                                   model_tokenizer=tokenizers[auxiliary],
                                   master_mask_token_id=tokenizers[master].mask_token_id,
                                   model_mask_token_id=tokenizers[auxiliary].mask_token_id)
    model_inputs = {master: torch.cat((prompt_ids, generated)), auxiliary: view.input_ids}
    try:
        auxiliary_generation_index = view.offsets.index((float(start), float(start + 1)))
    except ValueError as error:
        raise ValueError(f"No auxiliary mask aligned to generation slot {start}") from error
    starts = {master: len(prompt_ids) + start,
              auxiliary: view.generation_start + auxiliary_generation_index}
    prepared = []
    for text, origins in sources.items():
        native = {name: native_ids(tokenizers[name], text, max_span)
                  for name in (master, auxiliary)}
        reason = None
        if any(ids is None for ids in native.values()):
            reason = "non_roundtripping_or_span_too_long"
        elif any(start + len(ids) > len(generated) or
                 any(int(generated[start + offset]) != tokenizers[master].mask_token_id
                     for offset in range(len(ids)))
                 for ids in native.values()):
            reason = "blocked_master_slots"
        elif any(any(int(model_inputs[name][starts[name] + offset]) != tokenizers[name].mask_token_id
                     for offset in range(len(ids)))
                 for name, ids in native.items()):
            reason = "blocked_native_slots"
        prepared.append({"text": text, "origins": origins,
                         "token_ids": native, "unavailable_reason": reason})
    widths = {name: max((len(row["token_ids"][name]) for row in prepared
                         if row["unavailable_reason"] is None), default=0)
              for name in (master, auxiliary)}
    log_probs = {}
    seconds = {}
    for name in (master, auxiliary):
        if widths[name] and models is not None:
            positions = list(range(starts[name], starts[name] + widths[name]))
            log_probs[name], seconds[name] = score_native(
                models.model_for(name), tokenizers[name], model_inputs[name], positions,
                name)
    for row in prepared:
        if row["unavailable_reason"] or models is None:
            continue
        row["sum_log_probability"] = {
            name: sum(float(log_probs[name][offset, token])
                      for offset, token in enumerate(row["token_ids"][name]))
            for name in (master, auxiliary)}
        row["mean_log_probability"] = {
            name: row["sum_log_probability"][name] / len(row["token_ids"][name])
            for name in (master, auxiliary)}
        row["ensemble_score"] = sum(row["mean_log_probability"].values()) / 2
        if conditional:
            row["conditional_sum_log_probability"] = {}
            row["conditional_forward_seconds"] = {}
            for name in (master, auxiliary):
                native = row["token_ids"][name]
                total, extra_seconds = conditional_span_log_probability(
                    models.model_for(name), tokenizers[name], model_inputs[name],
                    starts[name], native, name, log_probs[name][0, native[0]])
                row["conditional_sum_log_probability"][name] = total
                row["conditional_forward_seconds"][name] = extra_seconds
            row["conditional_mean_log_probability"] = {
                name: row["conditional_sum_log_probability"][name] /
                      len(row["token_ids"][name]) for name in (master, auxiliary)}
            row["conditional_ensemble_score"] = sum(
                row["conditional_mean_log_probability"].values()) / 2
    prepared.sort(key=lambda row: row.get("ensemble_score", -math.inf), reverse=True)
    if conditional and models is not None:
        conditional_order = sorted(
            (row for row in prepared if "conditional_ensemble_score" in row),
            key=lambda row: row["conditional_ensemble_score"], reverse=True)
        for rank, row in enumerate(conditional_order, 1):
            row["conditional_rank"] = rank
    return {"generation_position": start, "native_view_length": view.generation_length,
            "forward_seconds": seconds, "candidates": prepared}


def parse_checkpoint(value):
    match = re.fullmatch(r"(\d+):(\d+)\.(\d+):(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("checkpoint must be DOC:BLOCK.STEP:ABS_POSITION")
    return tuple(map(int, match.groups()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-study-id", default="ctca-gsm8k4-benchmark-replay-20261029")
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-span", type=int, default=3)
    parser.add_argument("--conditional", action="store_true",
                        help="Also rescore later native tokens after inserting earlier span tokens")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in (args.study_id, args.source_study_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            parser.error("study IDs must be simple directory names")
    if args.top_k < 1 or args.max_span < 1:
        parser.error("top-k and max-span must be positive")
    checkpoints = args.checkpoint or DEFAULT_CHECKPOINTS
    source = ROOT / ".logs" / args.source_study_id
    destination = ROOT / ".logs" / args.study_id
    destination.mkdir(parents=True, exist_ok=True)
    models = None
    if args.prepare_only:
        tokenizers = {name: get_tokenizer(ModelArguments(
            model_name_or_path=MODEL_PATHS[name], dtype="bfloat16"))
                      for name in ("a", "b")}
    else:
        config = TSEConfig(model_a_path=MODEL_PATHS["a"], model_b_path=MODEL_PATHS["b"],
                           model_a_device="cuda:0", model_b_device="cuda:1",
                           dtype="bfloat16", ctca_enabled=True)
        models = load_tse_models(config)
        tokenizers = {name: models.tokenizer_for(name) for name in ("a", "b")}
    report = {"method": "mean native masked-token log probability; arithmetic mean across models",
              "conditional": args.conditional,
              "source_study_id": args.source_study_id, "checkpoints": []}
    for doc in sorted({item[0] for item in checkpoints}):
        manifest = json.loads((source / f"doc{doc}.json").read_text())
        trace = next(ROOT / path.lstrip("/") for path in manifest["traces"]
                     if path.endswith("-ensemble.jsonl"))
        master = re.search(rf"doc{doc}-([ab])-ensemble", trace.name).group(1)
        prompt_paths = {name: trace.parent / f"doc{doc}-prompt-{name}.txt" for name in ("a", "b")}
        prompt_ids = tokenizers[master](prompt_paths[master].read_text(),
                                        return_tensors="pt")["input_ids"][0]
        auxiliary = "b" if master == "a" else "a"
        auxiliary_prompt_ids = tokenizers[auxiliary](prompt_paths[auxiliary].read_text(),
                                                       return_tensors="pt")["input_ids"][0]
        generated = torch.full((512,), tokenizers[master].mask_token_id, dtype=torch.long)
        requested = {(block, step, pos) for d, block, step, pos in checkpoints if d == doc}
        found = set()
        for step, records in trace_steps(trace):
            for block, index, position in sorted(requested):
                if step != (block, index):
                    continue
                sources, forward = select_candidates(records, position, args.top_k)
                entry = {"doc": doc, "master": master, "step": list(step),
                         "absolute_position": position, "source_candidates": sources,
                         "trace_master_top": forward["master"]["distribution"]["top_k"][0]}
                entry.update(evaluate_checkpoint(models, tokenizers, master,
                                                 prompt_ids, auxiliary_prompt_ids,
                                                 generated, position, sources, args.max_span,
                                                 conditional=args.conditional))
                if models is not None:
                    top = entry["trace_master_top"]
                    match = next((row for row in entry["candidates"]
                                  if row["text"] == top["token"] and
                                  row["unavailable_reason"] is None and
                                  len(row["token_ids"][master]) == 1), None)
                    if match is not None:
                        replayed = math.exp(match["mean_log_probability"][master])
                        entry["master_top_replay_check"] = {
                            "recorded_probability": top["probability"],
                            "replayed_probability": replayed,
                            "absolute_difference": abs(replayed - top["probability"]),
                        }
                report["checkpoints"].append(entry)
                found.add((block, index, position))
                print(f"doc{doc} step {block}.{index} pos {position}: "
                      f"{len(sources)} candidates", flush=True)
            records["_prompt_length"] = len(prompt_ids)
            apply_commits(generated, records, tokenizers[master])
            if found == requested:
                break
        if found != requested:
            raise ValueError(f"Missing checkpoints in doc{doc}: {requested - found}")
    output = destination / ("prepared.json" if args.prepare_only else "rescore.json")
    output.write_text(json.dumps(report, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
