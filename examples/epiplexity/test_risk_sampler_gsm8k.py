"""
Smoke-test epiplexity samplers on a random GSM8K subset.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/test_risk_sampler_gsm8k.py --num_samples 50

For GPU cluster runs:
  srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/test_risk_sampler_gsm8k.py --num_samples 50
"""

import argparse
import random
import re

import torch
from datasets import load_dataset

from dllm.core.samplers.epiplexity_oracle import (
    OracleEpiplexitySampler,
    OracleEpiplexitySamplerConfig,
)
from dllm.core.samplers.epiplexity_risk import (
    RiskEpiplexitySampler,
    RiskEpiplexitySamplerConfig,
)
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--candidate_strategy", default="mixed")
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["greedy", "oracle", "risk"],
        choices=["greedy", "oracle", "risk"],
    )
    return parser.parse_args()


def extract_expected_answer(answer):
    """Extract GSM8K's final answer after the #### marker."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer)
    if match is None:
        return None
    return match.group(0).replace(",", "")


def extract_recent_numbers(text, window=3):
    """Extract the last few numeric strings from generated text."""
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return [number.replace(",", "") for number in numbers[-window:]]


def verify_ans(text, expected):
    """Simple heuristic: expected final number appears in the last few numbers."""
    if expected is None:
        return 0.0
    return 1.0 if expected in extract_recent_numbers(text) else 0.0


def sample_gsm8k_indices(dataset_size, num_samples, seed):
    rng = random.Random(seed)
    sample_count = min(num_samples, dataset_size)
    return rng.sample(range(dataset_size), sample_count)


def main():
    args = parse_args()

    print(f"Loading GSM8K split={args.split}...")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    sampled_indices = sample_gsm8k_indices(len(ds), args.num_samples, args.seed)
    print(
        f"Selected {len(sampled_indices)} random samples "
        f"from {len(ds)} examples with seed={args.seed}."
    )

    print("Loading LLaDA Model...")
    tokenizer = get_tokenizer(model_name_or_path=args.model_name)
    model_config = {
        "model_name_or_path": args.model_name,
        "trust_remote_code": True,
        "device_map": "cuda",
        "torch_dtype": "bfloat16",
    }
    model = get_model(**model_config)
    model.eval()

    samplers = {}
    configs = {}
    if "greedy" in args.samplers:
        samplers["greedy"] = MDLMSampler(model=model, tokenizer=tokenizer)
        configs["greedy"] = MDLMSamplerConfig(
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            remasking="low_confidence",
        )
    if "oracle" in args.samplers:
        samplers["oracle"] = OracleEpiplexitySampler(model=model, tokenizer=tokenizer)
        configs["oracle"] = OracleEpiplexitySamplerConfig(
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            oracle_candidate_strategy=args.candidate_strategy,
        )
    if "risk" in args.samplers:
        samplers["risk"] = RiskEpiplexitySampler(model=model, tokenizer=tokenizer)
        configs["risk"] = RiskEpiplexitySamplerConfig(
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            risk_candidate_strategy=args.candidate_strategy,
        )

    scores = {name: 0.0 for name in samplers}

    for row_idx, ds_idx in enumerate(sampled_indices, start=1):
        item = ds[ds_idx]
        prompt = item["question"]
        expected = extract_expected_answer(item["answer"])

        print(f"\n{'=' * 80}")
        print(f"SAMPLE {row_idx}/{len(sampled_indices)} | dataset_idx={ds_idx}")
        print(f"Expected: {expected}")
        print(prompt)
        print(f"{'=' * 80}")

        msgs = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids.to(
            model.device
        )
        inputs_list = [input_ids[0]]

        for sampler_name, sampler in samplers.items():
            print(f"Running {sampler_name}...")
            with torch.no_grad():
                output = sampler.sample(inputs_list, config=configs[sampler_name])
            answer = tokenizer.decode(
                output[0, input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            score = verify_ans(answer, expected)
            scores[sampler_name] += score

            print(f"\n[{sampler_name.upper()}] Output:\n{answer}")
            print(f"-> Correct by heuristic: {bool(score)}")

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS: ({len(sampled_indices)} GSM8K samples)")
    for sampler_name, score in scores.items():
        print(f"{sampler_name} average score: {score / len(sampled_indices):.4f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
