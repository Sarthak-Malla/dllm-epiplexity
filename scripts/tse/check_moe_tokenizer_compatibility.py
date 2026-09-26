"""Check tokenizer compatibility for the two LLaDA-MoE instruct checkpoints.

Run from the repository root with:

    python scripts/tse/check_moe_tokenizer_compatibility.py
"""

from transformers import AutoConfig, AutoTokenizer


CHECKPOINTS = {
    "moe_instruct": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    "moe_instruct_td": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct-TD",
}


def load_checkpoint_info(name: str, path: str):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    print(f"\n{name}: {path}")
    print("model_type:         ", config.model_type)
    print("config vocab_size:  ", config.vocab_size)
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    print("len(tokenizer):     ", len(tokenizer))
    print("mask_token:         ", tokenizer.mask_token)
    print("mask_token_id:      ", tokenizer.mask_token_id)
    print("special_tokens_map: ", tokenizer.special_tokens_map)
    print("all_special_ids:    ", tokenizer.all_special_ids)

    return config, tokenizer


def main() -> None:
    configs = {}
    tokenizers = {}

    for name, path in CHECKPOINTS.items():
        configs[name], tokenizers[name] = load_checkpoint_info(name, path)

    config_a = configs["moe_instruct"]
    config_b = configs["moe_instruct_td"]
    vocab_a = tokenizers["moe_instruct"].get_vocab()
    vocab_b = tokenizers["moe_instruct_td"].get_vocab()

    print("\nCompatibility checks")
    print("====================")
    print("Same model type:         ", config_a.model_type == config_b.model_type)
    print("Same config vocab size:  ", config_a.vocab_size == config_b.vocab_size)
    print("Same tokenizer length:   ", len(vocab_a) == len(vocab_b))
    print("Exact get_vocab equality:", vocab_a == vocab_b)

    only_a = sorted(set(vocab_a) - set(vocab_b))
    only_b = sorted(set(vocab_b) - set(vocab_a))
    id_mismatches = [
        (token, vocab_a[token], vocab_b[token])
        for token in set(vocab_a) & set(vocab_b)
        if vocab_a[token] != vocab_b[token]
    ]

    print("Tokens only in Instruct:   ", only_a[:20])
    print("Tokens only in Instruct-TD:", only_b[:20])
    print("Shared tokens with different IDs:", len(id_mismatches))
    print("First ID mismatches:", id_mismatches[:20])


if __name__ == "__main__":
    main()
