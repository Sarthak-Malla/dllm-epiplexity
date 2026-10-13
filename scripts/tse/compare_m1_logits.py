"""Compare native Dream and CTCA first-step logits for M1 parity.

Run from the repository root after activating the ``dllm`` environment:

    python /home/sarthak.malla/dllm-epiplexity/scripts/tse/compare_m1_logits.py \
        --native /absolute/path/to/native.pt \
        --ctca /absolute/path/to/ctca.pt
"""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse parity artifact paths and numerical tolerances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--ctca", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def load_artifact(path: Path) -> dict:
    """Load one tensor artifact on CPU."""
    if not path.is_absolute():
        raise ValueError(f"Artifact path must be absolute: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    """Align generation positions and enforce full-distribution parity."""
    args = parse_args()
    native = load_artifact(args.native)
    ctca = load_artifact(args.ctca)
    if native.get("source") != "dream_native":
        raise ValueError("--native is not a native Dream parity artifact")
    if ctca.get("source") != "ctca":
        raise ValueError("--ctca is not a CTCA parity artifact")
    if ctca.get("master_logit_adapter") != "dream_right_shift":
        raise ValueError("CTCA master did not use the Dream right-shift adapter")

    active_positions = ctca["active_positions"].long()
    prompt_lengths = ctca["master_prompt_lengths"]
    if active_positions.numel() == 0:
        raise ValueError("CTCA artifact contains no active positions")
    if torch.any(active_positions[:, 0] != 0):
        raise ValueError("M1 comparator currently expects a single-example trace")
    ctca_generation_positions = torch.tensor(
        [
            int(absolute_position) - int(prompt_lengths[int(batch_index)])
            for batch_index, absolute_position in active_positions.tolist()
        ],
        dtype=torch.long,
    )
    native_positions = native["generation_positions"].long()
    if not torch.equal(ctca_generation_positions, native_positions):
        raise AssertionError(
            "Native and CTCA generation positions differ: "
            f"{native_positions.tolist()} != {ctca_generation_positions.tolist()}"
        )

    ctca_rows = ctca["master_logits_sampling"][
        active_positions[:, 0], active_positions[:, 1]
    ].float()
    native_rows = native["sampling_logits"].float()
    if ctca_rows.shape != native_rows.shape:
        raise AssertionError(
            f"Logit shapes differ: native={tuple(native_rows.shape)}, "
            f"ctca={tuple(ctca_rows.shape)}"
        )

    difference = (native_rows - ctca_rows).abs()
    top1_match = native_rows.argmax(dim=-1) == ctca_rows.argmax(dim=-1)
    logits_match = torch.allclose(
        native_rows, ctca_rows, atol=args.atol, rtol=args.rtol
    )
    native_probabilities = torch.softmax(native_rows, dim=-1)
    ctca_probabilities = torch.softmax(ctca_rows, dim=-1)
    probabilities_match = torch.allclose(
        native_probabilities,
        ctca_probabilities,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("Compared generation positions:", native_rows.shape[0])
    print("Compared vocabulary entries per position:", native_rows.shape[1])
    print("Maximum absolute logit difference:", float(difference.max()))
    print("Mean absolute logit difference:", float(difference.mean()))
    print("Top-1 matches:", f"{int(top1_match.sum())}/{top1_match.numel()}")
    print("Logits allclose:", logits_match)
    print("Probabilities allclose:", probabilities_match)
    if not logits_match or not probabilities_match or not bool(top1_match.all()):
        raise SystemExit("M1 native/CTCA parity FAILED")
    print("M1 native/CTCA parity PASSED")


if __name__ == "__main__":
    main()
