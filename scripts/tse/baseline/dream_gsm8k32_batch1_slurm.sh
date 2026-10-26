#!/bin/bash
#SBATCH --job-name=dream-gsm8k32-b1
#SBATCH --output=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05,gpu-02
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -euo pipefail

cd /home/sarthak.malla/dllm-epiplexity
source /home/sarthak.malla/dllm-epiplexity/scripts/tse/run_logging.sh
tse_print_run_context "$0"

bash /home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/dream_parity_suite_slurm.sh \
    /home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k32_maskgit_st0p1_pt0p1_batch1_zeroshot.conf
