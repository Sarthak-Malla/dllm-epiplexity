#!/bin/bash
#SBATCH --job-name=ctca-gsm8k300-a5-b0-temp
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05,gpu-02
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:?usage: sbatch $0 CONFIG_PATH}"
exec bash scripts/tse/ctca/ctca_parity_eval.sh "${CONFIG_PATH}"
