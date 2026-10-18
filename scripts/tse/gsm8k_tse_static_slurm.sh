#!/bin/bash
#SBATCH --job-name=tse-gsm8k-static
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-scripts/tse/configs/gsm8k_instruct_temperature_static.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
tse_print_config_file "${CONFIG_PATH}"
source "${CONFIG_PATH}"

# Preserve existing configurations that do not specify an evaluation batch size.
batch_size="${batch_size:-1}"
progress_wandb="${progress_wandb:-false}"
progress_wandb_project="${progress_wandb_project:-dllm-tse}"
progress_wandb_entity="${progress_wandb_entity:-}"
progress_wandb_log_interval="${progress_wandb_log_interval:-1}"
model_a_suppress_tokens="${model_a_suppress_tokens:-[]}"
model_a_begin_suppress_tokens="${model_a_begin_suppress_tokens:-[]}"
model_b_suppress_tokens="${model_b_suppress_tokens:-[]}"
model_b_begin_suppress_tokens="${model_b_begin_suppress_tokens:-[]}"

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
: "${batch_size:?batch_size is required}"
: "${progress_wandb:?progress_wandb is required}"
: "${progress_wandb_project:?progress_wandb_project is required}"
: "${progress_wandb_log_interval:?progress_wandb_log_interval is required}"
: "${model_a_suppress_tokens:?model_a_suppress_tokens is required}"
: "${model_a_begin_suppress_tokens:?model_a_begin_suppress_tokens is required}"
: "${model_b_suppress_tokens:?model_b_suppress_tokens is required}"
: "${model_b_begin_suppress_tokens:?model_b_begin_suppress_tokens is required}"

source /apps/local/conda_init.sh
conda activate dllm
set -u

echo "===== TSE GSM8K run config ====="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "experiment_name=${experiment_name}"
echo "wandb_run_name=${wandb_run_name:-}"
echo "model_a=${model_a}"
echo "model_b=${model_b}"
echo "model_a_device=${model_a_device}"
echo "model_b_device=${model_b_device}"
echo "fusion_device=${fusion_device}"
echo "dtype=${dtype}"
echo "max_new_tokens=${max_new_tokens}"
echo "steps=${steps}"
echo "block_size=${block_size}"
echo "selection_mode=${selection_mode}"
echo "weighting_mode=${weighting_mode}"
echo "alpha=${alpha}"
echo "temperature_a=${temperature_a}"
echo "temperature_b=${temperature_b}"
echo "weight_temperature=${weight_temperature}"
echo "temperature=${temperature}"
echo "remasking=${remasking}"
echo "model_a_suppress_tokens=${model_a_suppress_tokens}"
echo "model_a_begin_suppress_tokens=${model_a_begin_suppress_tokens}"
echo "model_b_suppress_tokens=${model_b_suppress_tokens}"
echo "model_b_begin_suppress_tokens=${model_b_begin_suppress_tokens}"
echo "task=gsm8k_cot"
echo "num_fewshot=${num_fewshot}"
echo "batch_size=${batch_size}"
echo "progress_wandb=${progress_wandb}"
echo "progress_wandb_project=${progress_wandb_project}"
echo "progress_wandb_entity=${progress_wandb_entity}"
echo "progress_wandb_log_interval=${progress_wandb_log_interval}"
echo "================================="

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

RESULT_PATH=".logs/tse_gsm8k_${experiment_name}_${SLURM_JOB_ID}.json"
RUN_NAME="${wandb_run_name:-tse-gsm8k-${experiment_name}-${SLURM_JOB_ID}}"

MODEL_ARGS="model_a=${model_a},model_b=${model_b},model_a_device=${model_a_device},model_b_device=${model_b_device},fusion_device=${fusion_device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},selection_mode=${selection_mode},weighting_mode=${weighting_mode},alpha=${alpha},temperature_a=${temperature_a},temperature_b=${temperature_b},weight_temperature=${weight_temperature},temperature=${temperature},remasking=${remasking},model_a_suppress_tokens=${model_a_suppress_tokens},model_a_begin_suppress_tokens=${model_a_begin_suppress_tokens},model_b_suppress_tokens=${model_b_suppress_tokens},model_b_begin_suppress_tokens=${model_b_begin_suppress_tokens}"
MODEL_ARGS="${MODEL_ARGS},progress_wandb=${progress_wandb},progress_wandb_project=${progress_wandb_project},progress_wandb_run_name=${RUN_NAME}-progress,progress_wandb_log_interval=${progress_wandb_log_interval}"
if [[ -n "${progress_wandb_entity}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},progress_wandb_entity=${progress_wandb_entity}"
fi
echo "RESULT_PATH=${RESULT_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_ARGS=${MODEL_ARGS}"

accelerate launch --num_processes 1 dllm/pipelines/tse/eval.py \
    --tasks gsm8k_cot \
    --num_fewshot "${num_fewshot}" \
    --model tse_llada \
    --batch_size "${batch_size}" \
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
