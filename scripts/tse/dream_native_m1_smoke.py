"""Run native Dream decoding for one M1 prompt-parity question.

Run from the repository root after activating the ``dllm`` environment:

    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/dream_native_m1_smoke.py \
        --trace-jsonl /home/sarthak.malla/dllm-epiplexity/.logs/dream-native-m1.jsonl

The runner loads only Dream, applies Dream's native chat template and sampler,
and records the right-shifted native logits and commit trajectory for one prompt.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

import dllm
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig
from dllm.utils.sampling import sample_trim


DEFAULT_QUESTION = (
    "A store has 24 apples. It sells 7 in the morning and 14 in the afternoon. "
    "How many apples remain?"
)


def parse_args() -> argparse.Namespace:
    """Parse the one-question native Dream experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--prompt-style",
        action="append",
        choices=("raw", "gsm8k_cot"),
        help=(
            "Prompt style to run. Repeat the flag to compare styles in one model "
            "load. Defaults to gsm8k_cot."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--alg",
        choices=("maskgit_plus", "topk_margin", "entropy"),
        default="entropy",
    )
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--right-shift-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Dream's native logits-to-position alignment (enabled by default).",
    )
    parser.add_argument("--trace-top-k", type=int, default=5)
    parser.add_argument(
        "--trace-all-logits",
        action="store_true",
        help="Record masked-position top-k logits at every denoising step.",
    )
    parser.add_argument(
        "--trace-jsonl",
        required=True,
        help="Destination for the structured one-question native Dream trace.",
    )
    parser.add_argument(
        "--trace-tensors",
        default=None,
        help=(
            "Optional .pt destination for full-vocabulary, first-step adapted "
            "logits used by the M1 parity check."
        ),
    )
    return parser.parse_args()


def prompt_content(question: str, style: str) -> str:
    """Return the task text placed in Dream's user chat turn."""
    if style == "raw":
        return question
    if style == "gsm8k_cot":
        return f"Q: {question}\n\nA:"
    raise ValueError(f"Unsupported prompt style: {style}")


def device_map(device: str) -> dict[str, int] | None:
    """Build the explicit model placement used by the standalone native run."""
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}
    if device == "cuda":
        return {"": 0}
    return None


def synchronize(device: torch.device) -> None:
    """Synchronize a CUDA device before measuring a wall-clock interval."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def decode_token(tokenizer, token_id: int) -> str:
    """Decode one token while keeping mask and unknown IDs visible in the trace."""
    if token_id == tokenizer.mask_token_id:
        return "[Mask]"
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is None:
        return f"<unassigned:{token_id}>"
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_logits(tokenizer, logits: torch.Tensor, count: int) -> list[dict]:
    """Convert a logit row to JSON-safe top-k token metadata."""
    values, token_ids = torch.topk(logits.float(), k=min(count, logits.shape[-1]))
    return [
        {
            "token_id": int(token_id),
            "token": decode_token(tokenizer, int(token_id)),
            "logit": float(value),
        }
        for value, token_id in zip(values, token_ids)
    ]


class NativeDreamTrace:
    """Write a concise native Dream trajectory for one prompt style."""

    def __init__(
        self,
        *,
        handle,
        tokenizer,
        style: str,
        prompt_length: int,
        trace_top_k: int,
        trace_all_logits: bool,
        logit_adapter: str,
        tensor_path: str | None,
    ) -> None:
        self.handle = handle
        self.tokenizer = tokenizer
        self.style = style
        self.prompt_length = prompt_length
        self.trace_top_k = trace_top_k
        self.trace_all_logits = trace_all_logits
        self.logit_adapter = logit_adapter
        self.tensor_path = Path(tensor_path) if tensor_path is not None else None
        self.previous_generation: torch.Tensor | None = None

    def write(self, record: dict) -> None:
        """Write and flush one JSONL record so interrupted runs remain useful."""
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()

    def logits_hook(
        self, step: int, canvas: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Record Dream's post-right-shift masked-position logits."""
        if step == 0 and self.tensor_path is not None:
            masked_positions = torch.where(canvas[0] == self.tokenizer.mask_token_id)[0]
            self.tensor_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "source": "dream_native",
                    "prompt_style": self.style,
                    "prompt_length": self.prompt_length,
                    "generation_positions": (
                        masked_positions - self.prompt_length
                    ).detach().cpu(),
                    "sampling_logits": logits[0, masked_positions].detach().cpu(),
                    "logit_adapter": self.logit_adapter,
                },
                self.tensor_path,
            )
        if step != 0 and not self.trace_all_logits:
            return logits
        masked_positions = torch.where(canvas[0] == self.tokenizer.mask_token_id)[0]
        for absolute_position in masked_positions.tolist():
            self.write(
                {
                    "event": "native_logits",
                    "prompt_style": self.style,
                    "step": step,
                    "absolute_position": absolute_position,
                    "generation_position": absolute_position - self.prompt_length,
                    "logit_adapter": self.logit_adapter,
                    "top_k": top_logits(
                        self.tokenizer,
                        logits[0, absolute_position],
                        self.trace_top_k,
                    ),
                }
            )
        return logits

    def tokens_hook(
        self,
        step: int | None,
        canvas: torch.Tensor,
        logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Record only newly committed generation tokens after every native step."""
        generation = canvas[0, self.prompt_length :].detach().cpu()
        if step is None:
            self.previous_generation = generation.clone()
            self.write(
                {
                    "event": "native_initial_canvas",
                    "prompt_style": self.style,
                    "generation_length": int(generation.numel()),
                    "active_mask_count": int(
                        (generation == self.tokenizer.mask_token_id).sum()
                    ),
                }
            )
            return canvas

        if self.previous_generation is None:
            raise RuntimeError("Native Dream token hook did not receive an initial canvas")
        committed = torch.where(
            (self.previous_generation == self.tokenizer.mask_token_id)
            & (generation != self.tokenizer.mask_token_id)
        )[0]
        self.write(
            {
                "event": "native_commit",
                "prompt_style": self.style,
                "step": step,
                "new_commits": [
                    {
                        "generation_position": int(position),
                        "absolute_position": self.prompt_length + int(position),
                        "token_id": int(generation[position]),
                        "token": decode_token(
                            self.tokenizer, int(generation[position])
                        ),
                    }
                    for position in committed.tolist()
                ],
                "active_mask_count": int(
                    (generation == self.tokenizer.mask_token_id).sum()
                ),
            }
        )
        self.previous_generation = generation.clone()
        return canvas


def main() -> None:
    """Load Dream once and run the requested prompt styles."""
    args = parse_args()
    prompt_styles = args.prompt_style or ["gsm8k_cot"]
    if args.max_new_tokens < 1 or args.steps < 1:
        raise ValueError("max-new-tokens and steps must be positive")

    transformers.set_seed(args.seed)
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=args.model,
        dtype=args.dtype,
    )
    model = dllm.utils.get_model(model_args, device_map=device_map(args.device)).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args)
    sampler = DreamSampler(model=model, tokenizer=tokenizer)
    target_device = torch.device(args.device)

    trace_path = Path(args.trace_jsonl)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for style in prompt_styles:
            transformers.set_seed(args.seed)
            user_content = prompt_content(args.question, style)
            messages = [{"role": "user", "content": user_content}]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
            prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
            trace = NativeDreamTrace(
                handle=handle,
                tokenizer=tokenizer,
                style=style,
                prompt_length=int(prompt_ids.numel()),
                trace_top_k=args.trace_top_k,
                trace_all_logits=args.trace_all_logits,
                logit_adapter=(
                    "dream_native_right_shift"
                    if args.right_shift_logits
                    else "identity"
                ),
                tensor_path=args.trace_tensors,
            )
            sampler_config = DreamSamplerConfig(
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                alg=args.alg,
                alg_temp=args.alg_temp,
                right_shift_logits=args.right_shift_logits,
                return_dict=True,
            )
            trace.write(
                {
                    "event": "native_run_configuration",
                    "prompt_style": style,
                    "model": args.model,
                    "device": args.device,
                    "question": args.question,
                    "user_content": user_content,
                    "rendered_prompt": rendered_prompt,
                    "prompt_ids": prompt_ids.tolist(),
                    "max_new_tokens": args.max_new_tokens,
                    "steps": args.steps,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "alg": args.alg,
                    "alg_temp": args.alg_temp,
                    "right_shift_logits": args.right_shift_logits,
                    "seed": args.seed,
                }
            )
            print(f"\n[Native Dream M1: {style}]")
            print("Rendered prompt:\n" + rendered_prompt)
            synchronize(target_device)
            start_time = time.perf_counter()
            output = sampler.sample(
                [prompt_ids.to(target_device)],
                sampler_config,
                generation_tokens_hook_func=trace.tokens_hook,
                generation_logits_hook_func=trace.logits_hook,
                return_dict=True,
            )
            synchronize(target_device)
            elapsed_seconds = time.perf_counter() - start_time
            sequence = output.sequences[0].detach().cpu().tolist()
            generated_text = sample_trim(tokenizer, [sequence], [prompt_ids.tolist()])[0]
            decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=False)
            trace.write(
                {
                    "event": "native_run_summary",
                    "prompt_style": style,
                    "elapsed_seconds": elapsed_seconds,
                    "generated_text": generated_text,
                    "decoded_sequence": decoded_sequence,
                    "final_sequence_ids": sequence,
                }
            )
            print("Generated text:", repr(generated_text))
            print("Native Dream sample elapsed seconds:", f"{elapsed_seconds:.4f}")

    print("Native Dream structured trace:", trace_path.resolve())


if __name__ == "__main__":
    main()
