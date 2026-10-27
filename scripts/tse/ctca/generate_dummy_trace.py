"""Generate a small CPU CTCA trace with a masked sentence.

Run from the repository root after activating the dllm environment:
    OMP_NUM_THREADS=1 PYTHONPATH=/home/sarthak.malla/dllm-epiplexity python /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/generate_dummy_trace.py
Use --output PATH to write a different JSONL file.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from dllm.pipelines.tse.ctca_sampler import CTCATSESampler
from scripts.tse.smoke import make_ctca_trace_logger


DEFAULT_OUTPUT = Path(__file__).with_name("demo_dummy_trace.jsonl")
ORDINARY_TOKENS = ["The ", "cat ", "sat ", "on ", "the ", "mat", "rug", "."]


class SentenceTokenizer:
    """Minimal tokenizer whose decoded pieces form a readable sentence."""

    def __init__(self, mask_token: str, pad_token: str):
        self.tokens = ORDINARY_TOKENS + [pad_token, mask_token]
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.mask_token_id = self.vocab[mask_token]
        self.pad_token_id = self.vocab[pad_token]
        self.eos_token_id = self.pad_token_id
        self.bos_token_id = 0
        self.unk_token_id = 0
        self.all_special_ids = [self.pad_token_id, self.mask_token_id]

    def get_vocab(self):
        return dict(self.vocab)

    def convert_ids_to_tokens(self, token_id):
        return self.tokens[int(token_id)]

    def decode(self, token_ids, *, skip_special_tokens=True, **_kwargs):
        return "".join(
            self.tokens[int(token_id)]
            for token_id in token_ids
            if not skip_special_tokens or int(token_id) not in self.all_special_ids
        )

    def __call__(self, text, *, return_offsets_mapping=False, **_kwargs):
        ids, offsets = [], []
        position = 0
        pieces = sorted(ORDINARY_TOKENS, key=len, reverse=True)
        while position < len(text):
            piece = next((item for item in pieces if text.startswith(item, position)), None)
            if piece is None:
                position += 1
                continue
            ids.append(self.vocab[piece])
            offsets.append((position, position + len(piece)))
            position += len(piece)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class SentenceMaskedModel(torch.nn.Module):
    """Return fixed logits: a preferred noun followed by a period."""

    def __init__(self, vocab_size: int, noun_id: int):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embeddings = torch.nn.Embedding(vocab_size, 8)
        with torch.no_grad():
            torch.manual_seed(42)
            torch.nn.init.normal_(self.embeddings.weight)
        self.noun_id = noun_id

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask=None):
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -3.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self.noun_id] = 3.0
        if input_ids.shape[1] > 6:
            logits[:, 6, self.noun_id] = -3.0
            logits[:, 6, 7] = 4.0
        return SimpleNamespace(logits=logits)


def generate(output: Path) -> None:
    """Run both dummy models and save their complete two-step JSONL trace."""
    master_tokenizer = SentenceTokenizer("[MASK]", "[PAD]")
    auxiliary_tokenizer = SentenceTokenizer("<mask>", "<pad>")
    sampler = CTCATSESampler(
        SentenceMaskedModel(len(master_tokenizer.get_vocab()), master_tokenizer.vocab["mat"]),
        SentenceMaskedModel(len(auxiliary_tokenizer.get_vocab()), auxiliary_tokenizer.vocab["rug"]),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        num_anchors=3,
        min_anchors=3,
    )
    prompt = torch.tensor([0, 1, 2, 3, 4])
    trace = make_ctca_trace_logger(
        "a",
        "b",
        master_tokenizer,
        auxiliary_tokenizer,
        3,
        str(output),
        question_metadata={
            "label": "dummy masked sentence",
            "question": "The cat sat on the [MASK][MASK]",
            "masked_instance": "The cat sat on the [MASK][MASK]",
            "gold_answer": "mat.",
            "note": "First mask: master prefers mat; auxiliary prefers rug. Second mask: both prefer a period.",
        },
    )
    generated = sampler.sample(
        [prompt],
        auxiliary_inputs=[prompt],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        alpha=0.7,
        trace_callback=trace,
    )
    response = master_tokenizer.decode(generated[0, prompt.numel():].tolist())
    trace(
        "result",
        {
            "response": response,
            "gold_answer": "mat.",
            "matches_gold_by_last_number": None,
            "extraction_method": "exact_string_demo",
            "matches_gold": response == "mat.",
        },
    )
    print("Dummy masked sentence: The cat sat on the [MASK][MASK]")
    print("Expected completion: mat.")
    print("Generated completion:", response)
    print("Trace:", output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
