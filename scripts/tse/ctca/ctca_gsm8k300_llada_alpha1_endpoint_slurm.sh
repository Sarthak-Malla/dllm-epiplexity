#!/bin/bash
# Submit a full GSM8K-300 alpha=1 LLaDA-master CTCA endpoint benchmark.
# Usage: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_llada_alpha1_endpoint_slurm.sh [CONFIG_PATH]
#SBATCH --job-name=ctca-gsm8k300-llada-a1
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-51,gpu-24,gpu-38
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -eo pipefail

cd /home/juan.guevara/naruto/dllm-epiplexity
export SLURM_SUBMIT_DIR="$PWD"
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada_alpha1_endpoint_ptb0p1.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

exec bash scripts/tse/ctca/ctca_parity_eval.sh "${CONFIG_PATH}"
