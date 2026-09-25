#!/bin/bash
#SBATCH --job-name=tse-phase2-baseline
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs

source /apps/local/conda_init.sh
conda activate dllm
set -u

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python scripts/tse/smoke.py \
    --model-a "GSAI-ML/LLaDA-8B-Base" \
    --model-b "GSAI-ML/LLaDA-8B-Instruct" \
    --model-a-device cuda:0 \
    --model-b-device cuda:1 \
    --max-new-tokens 16 \
    --steps 16 \
    --block-size 16 \
    --top-k 5