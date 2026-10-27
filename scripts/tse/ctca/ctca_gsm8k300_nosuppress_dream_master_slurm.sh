#!/bin/bash
#SBATCH --job-name=ctca-gsm8k300-dream-nosup
#SBATCH --output=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05,gpu-02
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -euo pipefail

cd /home/sarthak.malla/dllm-epiplexity
source /home/sarthak.malla/dllm-epiplexity/scripts/tse/run_logging.sh
tse_print_run_context "$0"

bash /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_parity_eval.sh \
    /home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_dream_master.conf
