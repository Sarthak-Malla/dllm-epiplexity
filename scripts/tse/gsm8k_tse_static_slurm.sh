#!/bin/bash
#SBATCH --job-name=tse-gsm8k-static
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

CONFIG_PATH="${1:-scripts/tse/configs/gsm8k_instruct_temperature_static.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
source "${CONFIG_PATH}"

: "${experiment_name:?experiment_name is required}"
: "${model_a:?model_a is required}"
: "${model_b:?model_b is required}"
: "${model_a_device:?model_a_device is required}"
: "${model_b_device:?model_b_device is required}"
: "${fusion_device:?fusion_device is required}"
: "${dtype:?dtype is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${block_size:?block_size is required}"
: "${selection_mode:?selection_mode is required}"
: "${weighting_mode:?weighting_mode is required}"
: "${alpha:?alpha is required}"
: "${temperature_a:?temperature_a is required}"
: "${temperature_b:?temperature_b is required}"
: "${weight_temperature:?weight_temperature is required}"
: "${temperature:?temperature is required}"
: "${remasking:?remasking is required}"
: "${num_fewshot:?num_fewshot is required}"

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

RESULT_PATH=".logs/tse_gsm8k_${experiment_name}_${SLURM_JOB_ID}.json"
RUN_NAME="tse-gsm8k-${experiment_name}-${SLURM_JOB_ID}"

MODEL_ARGS="model_a=${model_a},model_b=${model_b},model_a_device=${model_a_device},model_b_device=${model_b_device},fusion_device=${fusion_device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},selection_mode=${selection_mode},weighting_mode=${weighting_mode},alpha=${alpha},temperature_a=${temperature_a},temperature_b=${temperature_b},weight_temperature=${weight_temperature},temperature=${temperature},remasking=${remasking}"

accelerate launch --num_processes 1 dllm/pipelines/tse/eval.py \
    --tasks gsm8k_cot \
    --num_fewshot "${num_fewshot}" \
    --model tse_llada \
    --apply_chat_template \
    --output_path "${RESULT_PATH}" \
    --model_args "${MODEL_ARGS}"

python scripts/tse/log_eval_to_wandb.py \
    --result-path "${RESULT_PATH}" \
    --run-name "${RUN_NAME}" \
    --mode tse \
    --weighting-mode "${weighting_mode}" \
    --model-a "${model_a}" \
    --model-b "${model_b}" \
    --temperature-a "${temperature_a}" \
    --temperature-b "${temperature_b}" \
    --alpha "${alpha}" \
    --config-path "${CONFIG_PATH}"
