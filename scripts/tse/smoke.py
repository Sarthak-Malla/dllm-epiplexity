"""Run the Phase 2 two-GPU TSE baseline smoke test.

Run from the repository root with:

    python scripts/tse/smoke.py \
        --model-a GSAI-ML/LLaDA-8B-Base \
        --model-b GSAI-ML/LLaDA-8B-Instruct
"""

import argparse

import torch

from dllm.pipelines.tse import TSEConfig, TSESampler
from dllm.pipelines.tse.loader import load_tse_models


DEFAULT_QUESTION = (
    "A store has 24 apples. It sells 7 apples in the morning and twice as many "
    "in the afternoon as in the morning. How many apples are left?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--model-a-device", default="cuda:0")
    parser.add_argument("--model-b-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    return parser.parse_args()


def print_device_summary() -> None:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}:", torch.cuda.get_device_name(index))


def print_top_predictions(tokenizer, paired_logits, top_k: int) -> None:
    print("\nPaired active-position logits:")
    for step, (positions, logits_a, logits_b) in enumerate(paired_logits):
        print(f"step {step}: active positions={len(positions)}")
        for active_index, (batch_index, sequence_position) in enumerate(positions.tolist()):
            position_logits_a = logits_a[active_index].float()
            position_logits_b = logits_b[active_index].float()
            values_a, ids_a = torch.topk(position_logits_a, k=top_k)
            values_b, ids_b = torch.topk(position_logits_b, k=top_k)
            tokens_a = [
                tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in ids_a
            ]
            tokens_b = [
                tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in ids_b
            ]
            scores_a = [round(float(value), 3) for value in values_a]
            scores_b = [round(float(value), 3) for value in values_b]
            print(
                f"  batch={batch_index}, sequence_position={sequence_position}"
            )
            print("    model A:", list(zip(tokens_a, scores_a)))
            print("    model B:", list(zip(tokens_b, scores_b)))
            print("    top-1 agreement:", int(ids_a[0]) == int(ids_b[0]))


def main() -> None:
    args = parse_args()
    print_device_summary()

    config = TSEConfig(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        model_a_device=args.model_a_device,
        model_b_device=args.model_b_device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        capture_logits=True,
    )
    models = load_tse_models(config)
    tokenizer = models.tokenizer

    print("\nModel A:", args.model_a)
    print("  model_type:", models.model_a.config.model_type)
    print("  vocab_size:", models.model_a.config.vocab_size)
    print("  device:", next(models.model_a.parameters()).device)
    print("Model B:", args.model_b)
    print("  model_type:", models.model_b.config.model_type)
    print("  vocab_size:", models.model_b.config.vocab_size)
    print("  device:", next(models.model_b.parameters()).device)
    print("Tokenizer:", tokenizer.name_or_path)
    print("  length:", len(tokenizer))
    print("  mask token/id:", tokenizer.mask_token, tokenizer.mask_token_id)

    messages = [{"role": "user", "content": args.question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0]
    prompt_ids = torch.as_tensor(prompt_ids)
    print("\nPrompt:", args.question)
    print("Prompt token count:", prompt_ids.numel())
    print("Prompt IDs:", prompt_ids.tolist())

    sampler = TSESampler(
        models.model_a,
        models.model_b,
        tokenizer,
        args.model_a_device,
        args.model_b_device,
    )
    generated = sampler.sample(
        [prompt_ids],
        max_new_tokens=config.max_new_tokens,
        steps=config.steps,
        block_size=config.block_size,
        capture_logits=True,
    )

    print("Captured forward steps:", len(sampler.last_paired_logits))
    if sampler.last_paired_logits:
        print_top_predictions(tokenizer, sampler.last_paired_logits, args.top_k)
    print("\nDecoded canvas:")
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()