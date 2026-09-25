#!/bin/bash
#SBATCH --job-name=tse-gsm8k-entropy
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-54
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
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

RESULT_PATH=".logs/tse_gsm8k_online_entropy_${SLURM_JOB_ID}.json"

accelerate launch --num_processes 1 dllm/pipelines/tse/eval.py \
    --tasks gsm8k_cot \
    --num_fewshot 5 \
    --model tse_llada \
    --apply_chat_template \
    --output_path "${RESULT_PATH}" \
    --model_args "model_a=GSAI-ML/LLaDA-8B-Base,model_b=GSAI-ML/LLaDA-8B-Instruct,model_a_device=cuda:0,model_b_device=cuda:1,fusion_device=cuda:0,max_new_tokens=512,steps=128,block_size=32,selection_mode=tse,weighting_mode=online_entropy,alpha=0.5,temperature_a=1.0,temperature_b=1.0,weight_temperature=1.0"

python scripts/tse/log_eval_to_wandb.py \
    --result-path "${RESULT_PATH}" \
    --run-name "tse-gsm8k-online-entropy-${SLURM_JOB_ID}" \
    --mode tse \
    --weighting-mode online_entropy \
    --model-a "GSAI-ML/LLaDA-8B-Base" \
    --model-b "GSAI-ML/LLaDA-8B-Instruct"
