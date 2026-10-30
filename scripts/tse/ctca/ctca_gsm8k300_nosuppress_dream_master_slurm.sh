#!/bin/bash
#SBATCH --job-name=ctca-gsm8k300-dream-nosup
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

set -euo pipefail

cd /home/juan.guevara/naruto/dllm-epiplexity
export SLURM_SUBMIT_DIR="$PWD"
source /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/run_logging.sh
tse_print_run_context "$0"

bash /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_parity_eval.sh \
    /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_dream_master.conf
