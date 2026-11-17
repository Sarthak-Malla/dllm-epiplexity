#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_dreammaster_jsrank_at0p03_pt0p10_slurm.sh
#SBATCH --job-name=ctca-gsm8k300-dream-jsrank
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclude=gpu-51,gpu-24,gpu-38,gpu-41

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
export SLURM_SUBMIT_DIR="$PWD"
exec bash /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_parity_eval.sh \
    /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_dreammaster_jsrank_at0p03_pt0p10.conf
