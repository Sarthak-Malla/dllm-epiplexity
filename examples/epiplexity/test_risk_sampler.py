"""
Smoke-test the decoding-risk epiplexity sampler.

Run from the repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/test_risk_sampler.py
"""

import re

import torch

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


TEST_PROMPTS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has "
        "only half of the money she needs. Her parents decided to give her $15 "
        "for that purpose, and her grandparents twice as much as her parents. "
        "How much more money does Betty need to buy the wallet?",
        "5",
    ),
]


def verify_ans(text, expected):
    """Simple heuristic to check if the expected number is near the end."""
    numbers = re.findall(r"\b\d+\b", text)
    if not numbers:
        return 0.0
    return 1.0 if expected in numbers[-3:] else 0.0


def test_samplers():
    print("Loading LLaDA Model...")
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = get_tokenizer(model_name_or_path=model_name)
    model_config = {
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": "cuda",
        "torch_dtype": "bfloat16",
    }
    model = get_model(**model_config)
    model.eval()

    greedy_sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    oracle_sampler = OracleEpiplexitySampler(model=model, tokenizer=tokenizer)
    risk_sampler = RiskEpiplexitySampler(model=model, tokenizer=tokenizer)

    greedy_config = MDLMSamplerConfig(
        steps=12,
        max_new_tokens=96,
        remasking="low_confidence",
    )
    oracle_config = OracleEpiplexitySamplerConfig(
        steps=12,
        max_new_tokens=96,
        oracle_candidate_strategy="mixed",
    )
    risk_config = RiskEpiplexitySamplerConfig(
        steps=12,
        max_new_tokens=96,
        risk_candidate_strategy="mixed",
    )

    scores = {
        "greedy": 0.0,
        "oracle": 0.0,
        "risk": 0.0,
    }

    for idx, (prompt, expected_answer) in enumerate(TEST_PROMPTS):
        print(f"\n{'=' * 60}")
        print(f"PROMPT {idx + 1}/{len(TEST_PROMPTS)}:")
        print(prompt)
        print(f"{'=' * 60}")

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

        print("Running Greedy MDLMSampler...")
        with torch.no_grad():
            greedy_out = greedy_sampler.sample(inputs_list, config=greedy_config)
        greedy_ans = tokenizer.decode(
            greedy_out[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        print("Running Entropy-drop Oracle Epiplexity Sampler...")
        with torch.no_grad():
            oracle_out = oracle_sampler.sample(inputs_list, config=oracle_config)
        oracle_ans = tokenizer.decode(
            oracle_out[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        print("Running Decoding-risk Epiplexity Sampler...")
        with torch.no_grad():
            risk_out = risk_sampler.sample(inputs_list, config=risk_config)
        risk_ans = tokenizer.decode(
            risk_out[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        print(f"\n[GREEDY] Output:\n{greedy_ans}")
        print(f"\n[ORACLE ENTROPY-DROP] Output:\n{oracle_ans}")
        print(f"\n[RISK] Output:\n{risk_ans}")

        scores["greedy"] += verify_ans(greedy_ans, expected_answer)
        scores["oracle"] += verify_ans(oracle_ans, expected_answer)
        scores["risk"] += verify_ans(risk_ans, expected_answer)

        print(f"\n-> Expected: {expected_answer}")
        print(f"-> Greedy Correct: {bool(verify_ans(greedy_ans, expected_answer))}")
        print(f"-> Oracle Correct: {bool(verify_ans(oracle_ans, expected_answer))}")
        print(f"-> Risk Correct: {bool(verify_ans(risk_ans, expected_answer))}")

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS: ({len(TEST_PROMPTS)} Prompts)")
    print(f"Greedy Average Score: {scores['greedy'] / len(TEST_PROMPTS)}")
    print(f"Oracle Average Score: {scores['oracle'] / len(TEST_PROMPTS)}")
    print(f"Risk Average Score: {scores['risk'] / len(TEST_PROMPTS)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_samplers()
