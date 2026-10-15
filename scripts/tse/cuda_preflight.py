"""Validate CUDA visibility inside a Slurm job step.

Run from an existing allocation with:
    srun python scripts/tse/cuda_preflight.py --minimum-device-count 1
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-device-count", type=int, required=True)
    args = parser.parse_args()

    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print("===== CUDA preflight =====")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={available}")
    print(f"cuda_device_count={device_count}")
    for index in range(device_count):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
    print("==========================")

    if not available or device_count < args.minimum_device_count:
        raise RuntimeError(
            "CUDA preflight failed: required at least "
            f"{args.minimum_device_count} visible GPU(s), found {device_count}"
        )


if __name__ == "__main__":
    main()
