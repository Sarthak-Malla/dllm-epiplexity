"""Run the two-GPU TSE smoke test.

Run from the repository root with:

    python scripts/tse/smoke.py \
        --model-a GSAI-ML/LLaDA-8B-Base \
        --model-b GSAI-ML/LLaDA-8B-Instruct

For CTCA, model A is LLaDA and model B is Dream. Add ``--ctca`` to print a
decoded trace of every canvas, model prediction, alignment, and fusion step.
"""

import argparse
from pathlib import Path
from typing import Any

import torch

import dllm


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
    parser.add_argument("--fusion-device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--trace-top-k",
        type=int,
        default=5,
        help="Number of decoded candidates to print per CTCA model prediction.",
    )
    parser.add_argument(
        "--baseline-model",
        choices=("a", "b"),
        default="b",
        help="Model used for baseline token commitment.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("baseline", "tse"),
        default="baseline",
        help="Use a baseline model or TSE selection.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature-a", type=float, default=1.0)
    parser.add_argument("--temperature-b", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        choices=("low_confidence", "random"),
        default="low_confidence",
    )
    parser.add_argument(
        "--weighting-mode",
        choices=("static", "online_entropy", "per_token_margin"),
        default="static",
    )
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--ctca", action="store_true")
    parser.add_argument("--master-model", choices=("a", "b"), default="a")
    parser.add_argument("--ctca-cache-dir", default=".cache/ctca")
    parser.add_argument("--ctca-force-rebuild", action="store_true")
    parser.add_argument("--ctca-anchor-temperature", type=float, default=0.01)
    parser.add_argument("--ctca-projection-temperature", type=float, default=0.05)
    parser.add_argument("--ctca-chunk-size", type=int, default=2500)
    parser.add_argument(
        "--ctca-projection-mode",
        choices=("exact", "sparse_topk"),
        default="sparse_topk",
    )
    parser.add_argument("--ctca-projection-top-k", type=int, default=64)
    parser.add_argument("--ctca-num-anchors", default="auto")
    parser.add_argument("--ctca-min-anchors", type=int, default=128)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    return parser.parse_args()


def print_device_summary() -> None:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print(f"cuda:{index}:", torch.cuda.get_device_name(index))


def print_import_summary() -> None:
    """Print the checkout and package locations used by the smoke test."""
    print("Working directory:", Path.cwd())
    print("dllm package:", Path(dllm.__file__).resolve())
    try:
        import dllm.pipelines.tse as tse
    except ModuleNotFoundError as error:
        print("TSE package import failed:", error)
        raise
    print("TSE package:", Path(tse.__file__).resolve())


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


def _decode_token(tokenizer, token_id: int) -> str:
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


def _decode_ids(tokenizer, token_ids) -> str:
    return " ".join(
        _decode_token(tokenizer, int(token_id)) for token_id in token_ids
    )


def make_ctca_trace_logger(
    master_id: str,
    auxiliary_id: str,
    master_tokenizer,
    auxiliary_tokenizer,
    top_k: int,
):
    from dllm.pipelines.tse.ctca.canvas import build_model_canvas_view

    labels = {
        "a": "model A (LLaDA)",
        "b": "model B (Dream)",
    }

    def top_predictions(tokenizer, logits: torch.Tensor) -> str:
        values, token_ids = torch.topk(logits.float(), k=min(top_k, logits.shape[-1]))
        return ", ".join(
            f"{_decode_token(tokenizer, int(token_id))!r}:{float(value):.3f}"
            for value, token_id in zip(values, token_ids)
        )

    def top_probabilities(tokenizer, probabilities: torch.Tensor) -> str:
        values, token_ids = torch.topk(
            probabilities.float(), k=min(top_k, probabilities.shape[-1])
        )
        return ", ".join(
            f"{_decode_token(tokenizer, int(token_id))!r}:{float(value):.6f}"
            for value, token_id in zip(values, token_ids)
        )

    def log(event: str, payload: dict) -> None:
        if event == "canvas":
            phase = payload["phase"]
            step = payload.get("step")
            prefix = f"\n[CTCA {phase} canvas"
            if step is not None:
                prefix += f" step={step[0]}.{step[1]}"
            print(prefix + "]")
            canvas = payload["canvas"]
            for sample_index, prompt_len in enumerate(payload["prompt_lens"]):
                print(
                    f"  {labels[master_id]}: "
                    f"{_decode_ids(master_tokenizer, canvas[sample_index].tolist())}"
                )
                auxiliary_prompt = payload["auxiliary_prompts"][sample_index]
                generated = canvas[sample_index, prompt_len:].tolist()
                auxiliary_view = build_model_canvas_view(
                    auxiliary_prompt,
                    generated,
                    master_tokenizer=master_tokenizer,
                    model_tokenizer=auxiliary_tokenizer,
                    master_mask_token_id=master_tokenizer.mask_token_id,
                    model_mask_token_id=auxiliary_tokenizer.mask_token_id,
                )
                print(
                    f"  {labels[auxiliary_id]}: "
                    f"{_decode_ids(auxiliary_tokenizer, auxiliary_view.input_ids.tolist())}"
                )
            return

        if event == "forward":
            step = payload["step"]
            print(f"\n[CTCA step {step[0]}.{step[1]} model predictions]")
            canvas = payload["canvas"]
            active_positions = payload["active_positions"]
            active_mask = payload["active_mask"]
            master_logits = payload["master_logits"]
            auxiliary_logits = payload["auxiliary_logits"]
            views = payload["views"]
            for sample_index, prompt_len in enumerate(payload["prompt_lens"]):
                print(
                    f"  {labels[master_id]} state: "
                    f"{_decode_ids(master_tokenizer, canvas[sample_index].tolist())}"
                )
                master_active = active_mask[sample_index]
                master_positions = torch.where(master_active)[0]
                master_values = master_logits[sample_index, master_active]
                for position, logits in zip(master_positions, master_values):
                    generation_position = int(position) - int(prompt_len)
                    print(
                        f"    abs pos {int(position)} gen pos {generation_position} -> "
                        f"{top_predictions(master_tokenizer, logits)}"
                    )

                view = views[sample_index]
                print(
                    f"  {labels[auxiliary_id]} state: "
                    f"{_decode_ids(auxiliary_tokenizer, view.input_ids.tolist())}"
                )
                auxiliary_generation = auxiliary_logits[
                    sample_index, view.generation_slice
                ]
                auxiliary_mask = view.input_ids[view.generation_slice] == (
                    auxiliary_tokenizer.mask_token_id
                )
                for position, logits in zip(
                    torch.where(auxiliary_mask)[0], auxiliary_generation[auxiliary_mask]
                ):
                    absolute_position = int(view.generation_slice.start) + int(position)
                    print(
                        f"    abs pos {absolute_position} gen pos {int(position)} -> "
                        f"{top_predictions(auxiliary_tokenizer, logits)}"
                    )

                overlap = view.overlap_matrix.to_dense().cpu()
                print("  alignment (auxiliary generation slot -> master generation slot):")
                for auxiliary_position, row in enumerate(overlap):
                    master_indices = torch.where(row > 0)[0]
                    if not len(master_indices):
                        continue
                    auxiliary_token = _decode_token(
                        auxiliary_tokenizer,
                        int(view.input_ids[view.generation_slice][auxiliary_position]),
                    )
                    mapped = ", ".join(
                        f"gen {int(master_position)} abs {int(prompt_len + master_position)} "
                        f"{_decode_token(master_tokenizer, int(canvas[sample_index, prompt_len + master_position]))!r}"
                        f" overlap={float(row[master_position]):.2f}"
                        for master_position in master_indices
                    )
                    auxiliary_absolute = int(view.generation_slice.start) + auxiliary_position
                    print(
                        f"    gen {auxiliary_position} abs {auxiliary_absolute} "
                        f"{auxiliary_token!r} -> {mapped}"
                    )
            return

        if event == "fusion":
            step = payload["step"]
            print(f"\n[CTCA step {step[0]}.{step[1]} probability fusion]")
            for index, (position, token_id, score) in enumerate(
                zip(
                    payload["active_positions"].tolist(),
                    payload["predicted_tokens"].tolist(),
                    payload["scores"].tolist(),
                )
            ):
                weight = payload["weights_a"]
                weight_text = "baseline"
                if weight is not None:
                    weight_text = f"model A weight={float(weight[index]):.3f}"
                master_probability = payload["master_probabilities"][
                    index, int(token_id)
                ]
                auxiliary_probability = payload["auxiliary_probabilities"][
                    index, int(token_id)
                ]
                fused_probabilities = payload.get("fused_probabilities")
                fused_text = "fused=baseline"
                if fused_probabilities is not None:
                    fused_probability = fused_probabilities[index, int(token_id)]
                    fused_text = f"fused={float(fused_probability):.6f}"
                print(
                    f"  batch={position[0]} pos={position[1]} -> "
                    f"{_decode_token(master_tokenizer, int(token_id))!r}, "
                    f"master={float(master_probability):.6f}, "
                    f"projected_aux={float(auxiliary_probability):.6f}, "
                    f"{fused_text}, fused_top_probability={float(score):.6f}, "
                    f"{weight_text}"
                )
                consensus_scores = payload.get("consensus_scores")
                if consensus_scores is not None:
                    print(
                        "    Jensen-Shannon consensus diagnostic: "
                        f"{float(consensus_scores[index]):.6f}"
                    )
                print(
                    f"    {labels[auxiliary_id]} projected to {labels[master_id]} top-k: "
                    f"{top_probabilities(master_tokenizer, payload['auxiliary_probabilities'][index])}"
                )
                if fused_probabilities is not None:
                    print(
                        "    fused top-k: "
                        f"{top_probabilities(master_tokenizer, fused_probabilities[index])}"
                    )

    return log


def main() -> None:
    args = parse_args()
    print_import_summary()
    from dllm.pipelines.tse import TSEConfig, TSESampler
    from dllm.pipelines.tse.loader import load_tse_models

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
        temperature=args.temperature,
        remasking=args.remasking,
        capture_logits=True,
        fusion_device=args.fusion_device or args.model_a_device,
        ctca_enabled=args.ctca,
        master_model=args.master_model,
        ctca_cache_dir=args.ctca_cache_dir,
        ctca_force_rebuild=args.ctca_force_rebuild,
        ctca_anchor_temperature=args.ctca_anchor_temperature,
        ctca_projection_temperature=args.ctca_projection_temperature,
        ctca_chunk_size=args.ctca_chunk_size,
        ctca_projection_mode=args.ctca_projection_mode,
        ctca_projection_top_k=args.ctca_projection_top_k,
        ctca_num_anchors=args.ctca_num_anchors,
        ctca_min_anchors=args.ctca_min_anchors,
    )
    models = load_tse_models(config)
    auxiliary_name = "b" if args.master_model == "a" else "a"
    tokenizer = (
        models.tokenizer_for(args.master_model) if args.ctca else models.tokenizer
    )
    auxiliary_tokenizer = (
        models.tokenizer_for(auxiliary_name) if args.ctca else None
    )

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
    auxiliary_prompt_ids = None
    if auxiliary_tokenizer is not None:
        auxiliary_prompt_ids = auxiliary_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]
        auxiliary_prompt_ids = torch.as_tensor(auxiliary_prompt_ids)
    print("\nPrompt:", args.question)
    print("Prompt token count:", prompt_ids.numel())
    print("Prompt IDs:", prompt_ids.tolist())

    if args.ctca:
        from dllm.pipelines.tse import CTCATSESampler

        master_device = getattr(args, f"model_{args.master_model}_device")
        auxiliary_device = getattr(args, f"model_{auxiliary_name}_device")
        sampler = CTCATSESampler(
            models.model_for(args.master_model),
            models.model_for(auxiliary_name),
            tokenizer,
            auxiliary_tokenizer,
            master_device,
            auxiliary_device,
            master_id=args.master_model,
            auxiliary_id=auxiliary_name,
            cache_dir=args.ctca_cache_dir,
            force_rebuild=args.ctca_force_rebuild,
            anchor_temperature=args.ctca_anchor_temperature,
            projection_temperature=args.ctca_projection_temperature,
            projection_chunk_size=args.ctca_chunk_size,
            projection_mode=args.ctca_projection_mode,
            projection_top_k=args.ctca_projection_top_k,
            num_anchors=args.ctca_num_anchors,
            min_anchors=args.ctca_min_anchors,
            master_cache_id=getattr(args, f"model_{args.master_model}"),
            auxiliary_cache_id=getattr(args, f"model_{auxiliary_name}"),
        )
        print("\nCTCA relative-anchor projection diagnostics:")
        for key, value in sampler.ctca_projection_diagnostics.items():
            print(f"  {key}: {value}")
    else:
        sampler = TSESampler(
            models.model_a,
            models.model_b,
            tokenizer,
            args.model_a_device,
            args.model_b_device,
        )
    sampler_inputs: dict[str, Any] = (
        {"auxiliary_inputs": [auxiliary_prompt_ids]}
        if auxiliary_prompt_ids is not None
        else {}
    )
    if args.ctca:
        sampler_inputs["trace_callback"] = make_ctca_trace_logger(
            args.master_model,
            auxiliary_name,
            tokenizer,
            auxiliary_tokenizer,
            args.trace_top_k,
        )
    generated = sampler.sample(
        [prompt_ids],
        **sampler_inputs,
        max_new_tokens=config.max_new_tokens,
        steps=config.steps,
        block_size=config.block_size,
        temperature=config.temperature,
        remasking=config.remasking,
        capture_logits=True,
        baseline_model=args.baseline_model,
        selection_mode=args.selection_mode,
        alpha=args.alpha,
        temperature_a=args.temperature_a,
        temperature_b=args.temperature_b,
        fusion_device=config.fusion_device,
        weighting_mode=args.weighting_mode,
        weight_temperature=args.weight_temperature,
    )

    print("Baseline commit model:", args.baseline_model)
    print("Selection mode:", args.selection_mode)
    print("Weighting mode:", args.weighting_mode)
    if args.ctca:
        aligned = getattr(sampler, "last_aligned_probabilities")
        print("Captured aligned steps:", len(aligned))
    else:
        paired_logits = getattr(sampler, "last_paired_logits")
        print("Captured forward steps:", len(paired_logits))
        if paired_logits:
            print_top_predictions(tokenizer, paired_logits, args.top_k)
    print("\nDecoded canvas:")
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
