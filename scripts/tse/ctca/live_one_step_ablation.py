"""Run live one-step CTCA ablations on saved GSM8K prompts.

    source ~/.zshrc
    conda activate dllm
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/live_one_step_ablation.py --prepare-only

Omit --prepare-only inside a GPU allocation to run the three live branches.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.tse.ctca.benchmark_replay import JOBS, parse_case, prompt, samples


STUDY = "ctca-gsm8k2-live-one-step-ablation-20261118"
DEFAULT_CASES = [(1, "a"), (2, "b")]
MODEL_PATHS = {"a": "GSAI-ML/LLaDA-8B-Instruct", "b": "Dream-org/Dream-v0-Instruct-7B"}
GENERATION = {"max_new_tokens": 512, "steps": 128, "block_size": 32}


def load_cases(cases):
    """Use document IDs and saved prompts, without consulting prior predictions."""
    sources = {role: samples(job) for role, job in JOBS.items()}
    for doc, master in cases:
        by_role = {role: rows[doc] for role, rows in sources.items()}
        if by_role["a"]["doc_hash"] != by_role["b"]["doc_hash"]:
            raise ValueError(f"Document mismatch for doc {doc}")
        yield {"doc": doc, "master": master, "gold": by_role[master]["target"],
               "prompts": {role: prompt(row) for role, row in by_role.items()},
               "prompt_source_jobs": JOBS}


def score_response(tokenizer, output_ids, prompt_ids, gold):
    """Score with the GSM8K flexible numeric pattern and stop strings."""
    prompt_length = len(tokenizer.decode(prompt_ids, skip_special_tokens=False))
    response = tokenizer.decode(output_ids, skip_special_tokens=False)[prompt_length:]
    for stop in ("Q:", "</s>", "<|im_end|>"):
        response = response.split(stop, 1)[0]
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    answer = next((value for value in matches[-1] if value), None) if matches else None

    def normalize(value):
        return value.replace(",", "").replace("$", "").removesuffix(".")

    return {"response": response, "extracted_answer": answer,
            "correct": answer is not None and normalize(answer) == normalize(str(gold))}


def run_pair(sampler, prompt_ids, auxiliary_ids, master_id, seed):
    """Pick the first live disagreement, then check and replay that intervention."""
    import torch

    target = None

    def snapshot(step, positions, master_ids, fusion_ids):
        return {"step": list(step), "history_index": len(sampler.last_canvas_history) - 1,
                "positions": positions.tolist(), "master_ids": master_ids.tolist(),
                "fusion_ids": fusion_ids.tolist()}

    def observe(step, positions, master_ids, fusion_ids):
        nonlocal target
        if target is None and not torch.equal(master_ids, fusion_ids):
            target = snapshot(step, positions, master_ids, fusion_ids)

    def sample(probe):
        torch.manual_seed(seed)
        tokens = sampler.sample(
            [prompt_ids], auxiliary_inputs=[auxiliary_ids], **GENERATION,
            selection_mode="baseline", baseline_model=master_id,
            capture_history=True, commit_probe=probe,
            master_begin_suppress_tokens=(126081, 126348) if master_id == "a" else (),
        )
        return tokens[0].cpu(), sampler.last_canvas_history

    print("  Running master baseline", flush=True)
    baseline, baseline_history = sample(observe)
    if target is None:
        return {"status": "no_eligible_difference"}, {"master": baseline}

    def replay(action):
        reached = False

        def probe(step, positions, master_ids, fusion_ids):
            nonlocal reached
            if list(step) != target["step"]:
                return None
            if snapshot(step, positions, master_ids, fusion_ids) != target:
                raise RuntimeError(f"{action} replay changed the selected positions or candidate IDs")
            index = target["history_index"]
            if not torch.equal(sampler.last_canvas_history[index], baseline_history[index]):
                raise RuntimeError(f"{action} replay has a different pre-commit canvas")
            reached = True
            return action

        tokens, history = sample(probe)
        if not reached or len(history) != len(baseline_history):
            raise RuntimeError(f"{action} replay did not preserve the target and schedule")
        return tokens, history

    print(f"  Running no-op replay for step {target['step']}", flush=True)
    noop, noop_history = replay("master")
    if not torch.equal(noop, baseline) or any(
        not torch.equal(expected, actual) for expected, actual in zip(baseline_history, noop_history)
    ):
        raise RuntimeError("No-op control did not reproduce the full baseline trajectory")

    print(f"  Running fusion replay for step {target['step']}", flush=True)
    fused, fused_history = replay("fusion")
    index = target["history_index"]
    if any(not torch.equal(control, branch) for control, branch in
           zip(baseline_history[:index + 1], fused_history[:index + 1])):
        raise RuntimeError("Fusion replay changed the trajectory before the intervention")
    expected = baseline_history[index + 1].clone()
    for (batch, position), token in zip(target["positions"], target["fusion_ids"]):
        expected[batch, position] = token
    if not torch.equal(expected, fused_history[index + 1]):
        raise RuntimeError("Fusion replay changed an unexpected position or token")
    return {"status": "completed", "intervention": target,
            "noop_full_trajectory_matches": True, "pre_intervention_trajectory_matches": True,
            "post_intervention_canvas_matches_expected": True}, {"master": baseline, "fusion": fused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--model-a-device", default="cuda:0")
    parser.add_argument("--model-b-device", default="cuda:0")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".logs" / STUDY)
    args = parser.parse_args()
    cases = args.case or DEFAULT_CASES
    if len(set(cases)) != len(cases):
        parser.error("Each document and master pair must be unique")
    print(f"Reading saved prompts for {len(cases)} cases", flush=True)
    selected = list(load_cases(cases))
    if args.prepare_only:
        print(json.dumps([{key: case[key] for key in ("doc", "master", "prompt_source_jobs")}
                          for case in selected], indent=2))
        return

    print("Importing model runtime", flush=True)
    import torch
    from dllm.pipelines.tse import CTCATSESampler, TSEConfig
    from dllm.pipelines.tse.loader import load_tse_models

    devices = {"a": args.model_a_device, "b": args.model_b_device}
    for device in set(devices.values()):
        print(f"Checking free memory on {device}", flush=True)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        minimum_gib = 36 if len(set(devices.values())) == 1 else 30
        if free_bytes < minimum_gib * 1024**3:
            raise RuntimeError(f"{device} needs at least {minimum_gib} GiB free for this pilot")

    config = TSEConfig(model_a_path=MODEL_PATHS["a"], model_b_path=MODEL_PATHS["b"],
                       model_a_device=devices["a"], model_b_device=devices["b"], ctca_enabled=True)
    print(f"Loading model checkpoints and tokenizers on {devices}", flush=True)
    models = load_tse_models(config)
    output = args.output_dir.resolve() / os.getenv("SLURM_JOB_ID", "manual")
    output.mkdir(parents=True, exist_ok=True)
    report = {"study_id": STUDY, "generation": GENERATION, "devices": devices,
              "seed": args.seed, "continuation": "greedy master with fixed CTCA reveal budget",
              "intervention_selection": "first scheduled step with a valid differing 50/50 token",
              "cases": []}
    for case in selected:
        master = case["master"]
        auxiliary = "b" if master == "a" else "a"
        tokenizers = {role: models.tokenizer_for(role) for role in devices}
        prompts = {role: tokenizer(case["prompts"][role], return_tensors="pt")["input_ids"][0]
                   for role, tokenizer in tokenizers.items()}
        print(f"Preparing projection for doc {case['doc']} with master {master}", flush=True)
        sampler = CTCATSESampler(
            models.model_for(master), models.model_for(auxiliary), tokenizers[master],
            tokenizers[auxiliary], devices[master], devices[auxiliary], master_id=master,
            auxiliary_id=auxiliary, cache_dir=str(ROOT / ".cache" / "ctca"),
            projection_chunk_size=512,
            master_cache_id=MODEL_PATHS[master], auxiliary_cache_id=MODEL_PATHS[auxiliary],
        )
        print(f"Running doc {case['doc']} with master {master}", flush=True)
        result, outputs = run_pair(sampler, prompts[master], prompts[auxiliary], master, args.seed)
        result.update(doc=case["doc"], master=master, gold=case["gold"],
                      prompt_source_jobs=case["prompt_source_jobs"])
        result["branches"] = {
            name: score_response(tokenizers[master], ids.tolist(), prompts[master].tolist(), case["gold"])
            for name, ids in outputs.items()
        }
        if "fusion" in outputs:
            control = int(result["branches"]["master"]["correct"])
            intervention = int(result["branches"]["fusion"]["correct"])
            result["correctness_delta"] = intervention - control
        report["cases"].append(result)
        destination = output / "report.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)
        print(f"doc {case['doc']}: {result['status']} ({result.get('correctness_delta', 'no pair')})", flush=True)
    print(destination)


if __name__ == "__main__":
    main()
