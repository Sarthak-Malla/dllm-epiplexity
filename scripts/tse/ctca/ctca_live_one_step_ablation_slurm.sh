#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_live_one_step_ablation_slurm.sh
#SBATCH --job-name=ctca-one-step
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

set -eo pipefail
echo "Starting job ${SLURM_JOB_ID:-manual} on ${SLURM_NODELIST:-unknown}"
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
echo "Using $(command -v python)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec python -u /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/live_one_step_ablation.py "$@"
