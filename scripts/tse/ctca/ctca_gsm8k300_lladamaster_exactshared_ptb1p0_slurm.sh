#!/bin/bash
# Submit with: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_lladamaster_exactshared_ptb1p0_slurm.sh
# LLaDA-master GSM8K-300 with current exact-shared projection.
#SBATCH --job-name=ctca-gsm8k300-llada-exact
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
#SBATCH --exclude=gpu-51,gpu-24,gpu-38

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
export SLURM_SUBMIT_DIR="$PWD"
exec bash /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_parity_eval.sh \
    /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_lladamaster_exactshared_ptb1p0.conf
