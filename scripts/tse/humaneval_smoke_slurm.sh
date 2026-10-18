#!/bin/bash
#SBATCH --job-name=humaneval-smoke
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

bash scripts/tse/baseline/dream_parity_suite_slurm.sh \
    scripts/tse/configs/dream_humaneval_smoke.conf
