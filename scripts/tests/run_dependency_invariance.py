"""
Run exact Q/K-capture logit invariance on the pinned LLaDA checkpoint.

On an allocated GPU node:

    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    DEPENDENCY_REPO=/home/sarthak.malla/dllm-selection-ensemble
    DEPENDENCY_CACHE=/home/sarthak.malla/.cache/huggingface/hub
    DEPENDENCY_MODEL=models--GSAI-ML--LLaDA-8B-Instruct
    DEPENDENCY_REVISION=08b83a6feb34df1a6011b80c3c00c7563e963b07
    python "${DEPENDENCY_REPO}/scripts/tests/run_dependency_invariance.py" \
        --checkpoint \
        "${DEPENDENCY_CACHE}/${DEPENDENCY_MODEL}/snapshots/${DEPENDENCY_REVISION}" \
        --output-path \
        "${DEPENDENCY_REPO}/eval_results/path_selection/p1_7_logit_invariance/result.json"
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

import dllm.pipelines.llada.models  # noqa: F401
from dllm.core.samplers.dependency import check_llada_capture_invariance


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Parse the reproducible checkpoint diagnostic configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Load the checkpoint, run the exact comparison, and write JSON evidence."""
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This real-checkpoint diagnostic requires an allocated CUDA GPU.")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = AutoModelForMaskedLM.from_pretrained(
        args.checkpoint,
        dtype=DTYPES[args.dtype],
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)
    if not 1 <= args.sequence_length <= int(model.config.max_sequence_length):
        raise ValueError("sequence-length is outside the model's supported range.")
    input_ids = torch.arange(
        1,
        args.sequence_length + 1,
        device=model_device,
        dtype=torch.long,
    ).remainder(vocab_size)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    result = check_llada_capture_invariance(
        model,
        input_ids,
        attention_mask=attention_mask,
        last_n_layers=args.last_n_layers,
    )
    report = asdict(result)
    report["passed"] = result.passed
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["model_class"] = type(model).__name__
    report["dtype"] = args.dtype
    report["device"] = str(model_device)
    report["gpu_name"] = torch.cuda.get_device_name(device_index)
    report["sequence_length"] = args.sequence_length
    report["seed"] = args.seed
    report["torch_version"] = torch.__version__

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    args.output_path.write_text(serialized + "\n")
    print(serialized)
    print(f"Saved invariance report to {args.output_path.resolve()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
