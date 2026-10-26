#!/bin/bash
#SBATCH --job-name=ctca-gsm8k300-entropy
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
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

exec bash scripts/tse/ctca/ctca_parity_eval.sh \
    scripts/tse/configs/ctca_gsm8k300_llada5_dream0_entropy_ptb0p1.conf
