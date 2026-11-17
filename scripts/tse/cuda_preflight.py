"""Validate CUDA visibility inside a Slurm job step.

Run from an existing allocation with:
    srun python scripts/tse/cuda_preflight.py --minimum-device-count 1
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-device-count", type=int, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=0.0)
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be non-negative")

    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print("===== CUDA preflight =====")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={available}")
    print(f"cuda_device_count={device_count}")
    free_gib = []
    memory_query_errors = []
    for index in range(device_count):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except RuntimeError as error:
            free_gib.append(None)
            memory_query_errors.append((index, str(error).splitlines()[0]))
            print(f"cuda_device_{index}_memory_query_error={memory_query_errors[-1][1]}")
        else:
            free_gib.append(free_bytes / (1024 ** 3))
            print(f"cuda_device_{index}_free_gib={free_gib[-1]:.2f}")
            print(f"cuda_device_{index}_total_gib={total_bytes / (1024 ** 3):.2f}")
    print("==========================")

    if not available or device_count < args.minimum_device_count:
        raise RuntimeError(
            "CUDA preflight failed: required at least "
            f"{args.minimum_device_count} visible GPU(s), found {device_count}"
        )
    if memory_query_errors:
        raise RuntimeError(
            "CUDA preflight failed: could not query GPU free memory: "
            f"{memory_query_errors}"
        )
    if any(free < args.minimum_free_gib for free in free_gib[:args.minimum_device_count]):
        raise RuntimeError(
            f"CUDA preflight failed: need {args.minimum_free_gib:.1f} GiB free "
            f"on each of the first {args.minimum_device_count} devices; "
            f"found {[round(free, 2) for free in free_gib]} GiB"
        )


if __name__ == "__main__":
    main()
