#!/bin/bash
# Submit a full GSM8K-300 alpha=1 LLaDA-master CTCA endpoint benchmark.
# Usage: sbatch /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_llada_alpha1_endpoint_slurm.sh [CONFIG_PATH]
#SBATCH --job-name=ctca-gsm8k300-llada-a1
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

CONFIG_PATH="${1:-/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada_alpha1_endpoint_ptb0p1.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

exec bash scripts/tse/ctca/ctca_parity_eval.sh "${CONFIG_PATH}"
