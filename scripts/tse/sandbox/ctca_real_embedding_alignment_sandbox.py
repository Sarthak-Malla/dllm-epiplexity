"""Inspect CTCA token alignment with real tokenizers and real model embeddings.

From the repository root:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/sandbox/ctca_real_embedding_alignment_sandbox.py

For the full LLaDA/Dream checkpoints, run on a GPU allocation, for example:
    srun -p $PARTITION -q=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python /home/sarthak.malla/dllm-epiplexity/scripts/tse/sandbox/ctca_real_embedding_alignment_sandbox.py --device cuda:0

This sandbox performs token alignment only. It does not run model prediction.
It loads each model's input embedding matrix, represents tokens by their
similarities to shared anchor tokens in each model's own embedding space, and
prints the nearest master tokens for a small set of auxiliary tokens in that
relative-anchor space.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dllm.pipelines.tse.ctca.alignment import normalize_token_text


DEFAULT_LLADA = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_DREAM = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_LLADA_MASK_ID = 126336


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llada-model", default=DEFAULT_LLADA)
    parser.add_argument("--dream-model", default=DEFAULT_DREAM)
    parser.add_argument(
        "--master",
        choices=("llada", "dream"),
        default="dream",
        help="Tokenizer/model whose vocabulary is the projection target.",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=[" answer", " is", " ", "5", "6", "7", " apples", "."],
        help="Decoded token strings to inspect when available in the auxiliary tokenizer.",
    )
    parser.add_argument("--llada-mask-id", type=int, default=DEFAULT_LLADA_MASK_ID)
    parser.add_argument(
        "--num-anchors",
        default="auto",
        help="Number of shared anchors to use, or 'auto' to use all shared anchors.",
    )
    parser.add_argument("--min-anchors", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--anchor-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for token-to-anchor similarity profiles.",
    )
    parser.add_argument("--projection-temperature", type=float, default=0.05)
    parser.add_argument("--master-chunk-size", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="dtype used when loading checkpoint weights.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading and alignment, for example cpu or cuda:0.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_tokenizer(model_id: str, *, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    return tokenizer


def device_map_for(device: str):
    if device == "cpu":
        return None
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}
    return None


def load_input_embeddings(
    model_id: str,
    *,
    dtype: torch.dtype,
    device: str,
    local_files_only: bool,
) -> torch.Tensor:
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map_for(device),
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise ValueError(f"{model_id} does not expose input embeddings")
    return embeddings.weight.detach()


def mask_token_id(tokenizer, *, fallback: int | None = None) -> int | None:
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    candidates = ("[MASK]", "<mask>", "<|mask|>", "<MASK>", "<|mdm_mask|>")
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    if fallback is not None and 0 <= fallback < len(tokenizer):
        return int(fallback)
    return None


def collect_all_shared_anchors(
    auxiliary_tokenizer,
    master_tokenizer,
    *,
    auxiliary_vocab_size: int,
    master_vocab_size: int,
) -> tuple[list[int], list[int]]:
    """Collect all deterministic one-to-one shared string anchors."""

    def normalized_vocab(tokenizer, vocab_size: int) -> dict[str, int]:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        candidates: dict[str, int] = {}
        for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            token_id = int(token_id)
            normalized = normalize_token_text(token)
            if (
                not normalized
                or token_id < 0
                or token_id >= vocab_size
                or token_id in special_ids
            ):
                continue
            candidates.setdefault(normalized, token_id)
        return candidates

    auxiliary = normalized_vocab(auxiliary_tokenizer, auxiliary_vocab_size)
    master = normalized_vocab(master_tokenizer, master_vocab_size)
    shared = sorted(set(auxiliary).intersection(master))
    return [auxiliary[token] for token in shared], [master[token] for token in shared]


def select_anchor_ids(
    auxiliary_anchor_ids: Sequence[int],
    master_anchor_ids: Sequence[int],
    num_anchors: str,
    *,
    min_anchors: int,
) -> tuple[list[int], list[int]]:
    if min_anchors < 1:
        raise ValueError("min_anchors must be positive")
    if num_anchors.casefold() == "auto":
        selected_count = len(auxiliary_anchor_ids)
    else:
        try:
            selected_count = int(num_anchors)
        except ValueError as error:
            raise ValueError("--num-anchors must be an integer or 'auto'") from error
        if selected_count < 1:
            raise ValueError("--num-anchors must be positive or 'auto'")
        selected_count = min(selected_count, len(auxiliary_anchor_ids))
    if selected_count < min_anchors:
        raise ValueError(
            "relative-anchor alignment found too few shared anchors: "
            f"{selected_count} < {min_anchors}"
        )
    return (
        list(auxiliary_anchor_ids[:selected_count]),
        list(master_anchor_ids[:selected_count]),
    )


def decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(map(int, token_ids)),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def encode_single_token(tokenizer, text: str) -> int | None:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def token_label(tokenizer, token_id: int, mask_id: int | None) -> str:
    token_id = int(token_id)
    if mask_id is not None and token_id == mask_id:
        return "[MASK]"
    token = tokenizer.convert_ids_to_tokens(token_id)
    text = decode(tokenizer, [token_id])
    if text and text != token:
        return f"{token!r}/{text!r}"
    return repr(token)


def inspect_auxiliary_token_ids(
    tokenizer,
    texts: Sequence[str],
    *,
    mask_id: int | None,
) -> list[int]:
    token_ids: list[int] = []
    seen = set()
    for text in texts:
        token_id = encode_single_token(tokenizer, text)
        if token_id is None:
            print(f"  skipping {text!r}: not a single auxiliary token")
            continue
        if token_id not in seen:
            token_ids.append(token_id)
            seen.add(token_id)
    if mask_id is not None and mask_id not in seen:
        token_ids.append(mask_id)
    return token_ids


def topk_master_neighbors_by_relative_anchors(
    auxiliary_query_embeddings: torch.Tensor,
    auxiliary_anchor_embeddings: torch.Tensor,
    master_embeddings: torch.Tensor,
    master_anchor_embeddings: torch.Tensor,
    *,
    top_k: int,
    anchor_temperature: float,
    projection_temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if anchor_temperature <= 0 or projection_temperature <= 0 or chunk_size < 1:
        raise ValueError("temperatures and chunk_size must be positive")
    if (
        auxiliary_query_embeddings.ndim != 2
        or auxiliary_anchor_embeddings.ndim != 2
        or master_embeddings.ndim != 2
        or master_anchor_embeddings.ndim != 2
    ):
        raise ValueError("embedding matrices must have shape [N, hidden_size]")
    if auxiliary_query_embeddings.shape[1] != auxiliary_anchor_embeddings.shape[1]:
        raise ValueError("auxiliary queries and anchors must share a hidden size")
    if master_embeddings.shape[1] != master_anchor_embeddings.shape[1]:
        raise ValueError("master embeddings and anchors must share a hidden size")
    if auxiliary_anchor_embeddings.shape[0] != master_anchor_embeddings.shape[0]:
        raise ValueError("auxiliary and master anchor counts must match")

    device = auxiliary_query_embeddings.device
    k = min(top_k, master_embeddings.shape[0])
    best_values = torch.full(
        (auxiliary_query_embeddings.shape[0], k),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.zeros(
        (auxiliary_query_embeddings.shape[0], k), dtype=torch.long, device=device
    )

    auxiliary_queries = F.normalize(auxiliary_query_embeddings.float(), dim=-1)
    auxiliary_anchors = F.normalize(auxiliary_anchor_embeddings.float(), dim=-1)
    query_anchor_scores = auxiliary_queries @ auxiliary_anchors.transpose(0, 1)
    query_relative = torch.softmax(query_anchor_scores / anchor_temperature, dim=-1)
    query_relative = F.normalize(query_relative, dim=-1)

    master_anchors = F.normalize(master_anchor_embeddings.to(device).float(), dim=-1)

    for start in range(0, master_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, master_embeddings.shape[0])
        master = F.normalize(master_embeddings[start:end].to(device).float(), dim=-1)
        master_anchor_scores = master @ master_anchors.transpose(0, 1)
        master_relative = torch.softmax(master_anchor_scores / anchor_temperature, dim=-1)
        master_relative = F.normalize(master_relative, dim=-1)
        similarities = query_relative @ master_relative.transpose(0, 1)
        local_k = min(k, similarities.shape[1])
        local_values, local_indices = torch.topk(similarities, k=local_k, dim=-1)
        local_indices = local_indices + start
        merged_values = torch.cat([best_values, local_values], dim=-1)
        merged_indices = torch.cat([best_indices, local_indices], dim=-1)
        best_values, order = torch.topk(merged_values, k=k, dim=-1)
        best_indices = torch.gather(merged_indices, 1, order)

    weights = torch.softmax(best_values / projection_temperature, dim=-1)
    return best_indices.cpu(), weights.cpu(), best_values.cpu()


def print_parameter_guide(args: argparse.Namespace) -> None:
    print("\nParameter guide:")
    print(
        "  --num-anchors controls how many shared normalized token strings define "
        "the relative space. Use 'auto' to use every shared anchor."
    )
    print(
        "  --min-anchors is the safety floor; the script fails if fewer anchors are found."
    )
    print(
        "  --top-k controls how many nearest master-vocab tokens are printed for each "
        "auxiliary token."
    )
    print(
        "  --anchor-temperature controls the softmax over token-to-anchor cosine "
        "similarities. Lower values make each token focus on fewer anchors; higher "
        "values make the relative-anchor profile flatter."
    )
    print(
        "  --projection-temperature controls the softmax over top-k master-token "
        "similarities. Lower values make the top neighbor receive more projection "
        "mass; higher values spread mass across the top-k neighbors."
    )
    print(
        "  --master-chunk-size controls memory during nearest-neighbor search over "
        "the master vocabulary. It should not change results except for tiny numeric drift."
    )
    print(
        "  This sandbox has no generation temperature because it does not run model "
        "prediction or sample tokens."
    )
    print(
        f"  Current values: num_anchors={args.num_anchors}, "
        f"min_anchors={args.min_anchors}, top_k={args.top_k}, "
        f"anchor_temperature={args.anchor_temperature}, "
        f"projection_temperature={args.projection_temperature}, "
        f"master_chunk_size={args.master_chunk_size}"
    )


def print_neighbors(
    title: str,
    *,
    auxiliary_tokenizer,
    master_tokenizer,
    auxiliary_token_ids: Sequence[int],
    master_indices: torch.Tensor,
    weights: torch.Tensor,
    similarities: torch.Tensor,
    auxiliary_mask_id: int | None,
    master_mask_id: int | None,
) -> None:
    print(f"\n{title}")
    for row, aux_id in enumerate(auxiliary_token_ids):
        mapped = [
            (
                token_label(master_tokenizer, int(master_id), master_mask_id),
                round(float(weight), 4),
                round(float(similarity), 4),
            )
            for master_id, weight, similarity in zip(
                master_indices[row], weights[row], similarities[row]
            )
        ]
        print(
            f"  {token_label(auxiliary_tokenizer, aux_id, auxiliary_mask_id)} "
            f"-> {mapped}"
        )


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=4, sci_mode=False)

    model_ids = {"llada": args.llada_model, "dream": args.dream_model}
    auxiliary_name = "llada" if args.master == "dream" else "dream"
    master_name = args.master

    print("Loading tokenizers...")
    tokenizers = {
        name: load_tokenizer(model_id, local_files_only=args.local_files_only)
        for name, model_id in model_ids.items()
    }
    mask_ids = {
        "llada": mask_token_id(tokenizers["llada"], fallback=args.llada_mask_id),
        "dream": mask_token_id(tokenizers["dream"]),
    }

    dtype = torch_dtype(args.dtype)
    print("Loading model input embeddings...")
    embeddings = {
        name: load_input_embeddings(
            model_id,
            dtype=dtype,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        for name, model_id in model_ids.items()
    }

    master_tokenizer = tokenizers[master_name]
    auxiliary_tokenizer = tokenizers[auxiliary_name]
    master_embeddings = embeddings[master_name]
    auxiliary_embeddings = embeddings[auxiliary_name]

    print("\nSetup:")
    print(f"  master: {master_name} ({model_ids[master_name]})")
    print(f"  auxiliary: {auxiliary_name} ({model_ids[auxiliary_name]})")
    print(f"  master embeddings: {tuple(master_embeddings.shape)}")
    print(f"  auxiliary embeddings: {tuple(auxiliary_embeddings.shape)}")
    print(f"  dtype: {args.dtype}")
    print(f"  device: {args.device}")

    all_auxiliary_anchor_ids, all_master_anchor_ids = collect_all_shared_anchors(
        auxiliary_tokenizer,
        master_tokenizer,
        auxiliary_vocab_size=auxiliary_embeddings.shape[0],
        master_vocab_size=master_embeddings.shape[0],
    )
    auxiliary_anchor_ids, master_anchor_ids = select_anchor_ids(
        all_auxiliary_anchor_ids,
        all_master_anchor_ids,
        args.num_anchors,
        min_anchors=args.min_anchors,
    )
    print(f"\nTotal usable shared anchors: {len(all_auxiliary_anchor_ids)}")
    print(f"Anchors selected by --num-anchors: {len(auxiliary_anchor_ids)}")
    print("First anchors:")
    for aux_id, master_id in list(zip(all_auxiliary_anchor_ids, all_master_anchor_ids))[:10]:
        print(
            "  "
            f"{token_label(auxiliary_tokenizer, aux_id, mask_ids[auxiliary_name])} "
            "-> "
            f"{token_label(master_tokenizer, master_id, mask_ids[master_name])}"
        )

    print("\nAuxiliary tokens requested for inspection:")
    auxiliary_token_ids = inspect_auxiliary_token_ids(
        auxiliary_tokenizer,
        args.tokens,
        mask_id=mask_ids[auxiliary_name],
    )
    if not auxiliary_token_ids:
        raise ValueError("no requested tokens were single auxiliary tokens")

    query_ids = torch.tensor(
        auxiliary_token_ids, dtype=torch.long, device=auxiliary_embeddings.device
    )
    auxiliary_anchor_index = torch.tensor(
        auxiliary_anchor_ids, dtype=torch.long, device=auxiliary_embeddings.device
    )
    master_anchor_index = torch.tensor(
        master_anchor_ids, dtype=torch.long, device=master_embeddings.device
    )
    auxiliary_queries = auxiliary_embeddings.index_select(0, query_ids)
    auxiliary_anchor_embeddings = auxiliary_embeddings.index_select(
        0, auxiliary_anchor_index
    )
    master_anchor_embeddings = master_embeddings.index_select(0, master_anchor_index)

    print("\nSearching master tokens in relative-anchor space...")
    relative_indices, relative_weights, relative_similarities = (
        topk_master_neighbors_by_relative_anchors(
            auxiliary_queries,
            auxiliary_anchor_embeddings,
            master_embeddings,
            master_anchor_embeddings,
            top_k=args.top_k,
            anchor_temperature=args.anchor_temperature,
            projection_temperature=args.projection_temperature,
            chunk_size=args.master_chunk_size,
        )
    )

    print_neighbors(
        "Relative-anchor projection: active auxiliary token -> top master tokens",
        auxiliary_tokenizer=auxiliary_tokenizer,
        master_tokenizer=master_tokenizer,
        auxiliary_token_ids=auxiliary_token_ids,
        master_indices=relative_indices,
        weights=relative_weights,
        similarities=relative_similarities,
        auxiliary_mask_id=mask_ids[auxiliary_name],
        master_mask_id=mask_ids[master_name],
    )
    print_parameter_guide(args)


if __name__ == "__main__":
    main()
