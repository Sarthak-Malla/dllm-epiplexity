#!/bin/bash
#SBATCH --job-name=tse-gsm8k-instruct
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-54
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs

CONFIG_PATH="${1:-scripts/tse/configs/gsm8k_instruct_probability_temperature_02.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
source "${CONFIG_PATH}"

: "${experiment_name:?experiment_name is required}"
: "${model_name_or_path:?model_name_or_path is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${block_size:?block_size is required}"
: "${probability_temperature:?probability_temperature is required}"
: "${temperature:?temperature is required}"
: "${cfg_scale:?cfg_scale is required}"
: "${suppress_tokens:?suppress_tokens is required}"
: "${begin_suppress_tokens:?begin_suppress_tokens is required}"
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

MODEL_ARGS="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=${temperature},probability_temperature=${probability_temperature},cfg_scale=${cfg_scale},suppress_tokens=${suppress_tokens},begin_suppress_tokens=${begin_suppress_tokens}"

accelerate launch --num_processes 1 dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot \
    --num_fewshot "${num_fewshot}" \
    --model llada \
    --apply_chat_template \
    --output_path "${RESULT_PATH}" \
    --model_args "${MODEL_ARGS}"

python scripts/tse/log_eval_to_wandb.py \
    --result-path "${RESULT_PATH}" \
    --run-name "${RUN_NAME}" \
    --mode instruct_baseline \
    --model-a "${model_name_or_path}" \
    --probability-temperature "${probability_temperature}" \
    --config-path "${CONFIG_PATH}"
