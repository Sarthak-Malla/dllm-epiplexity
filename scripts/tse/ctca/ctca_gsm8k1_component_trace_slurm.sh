#!/bin/bash
# Submit one-question CTCA component trace.
# Usage: sbatch /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k1_component_trace_slurm.sh [CONFIG_PATH]
#SBATCH --job-name=ctca-gsm8k1-trace
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-04
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k1_llada_master_trace.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# The existing trace runner logs probability projection, fusion candidates,
# selection ranks, and committed tokens for every denoising step.
srun bash scripts/tse/ctca/ctca_static_smoke_slurm.sh "${CONFIG_PATH}"
