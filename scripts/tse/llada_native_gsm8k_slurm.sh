#!/bin/bash
#SBATCH --job-name=llada-native-gsm8k
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs

CONFIG_PATH="${1:-scripts/tse/configs/llada_native_gsm8k.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
source "${CONFIG_PATH}"

wandb_project="${wandb_project:-dllm-tse}"
wandb_entity="${wandb_entity:-}"

: "${experiment_name:?experiment_name is required}"
: "${model:?model is required}"
: "${device:?device is required}"
: "${dtype:?dtype is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${block_size:?block_size is required}"
: "${temperature:?temperature is required}"
: "${probability_temperature:?probability_temperature is required}"
: "${remasking:?remasking is required}"
: "${stochastic_transfer:?stochastic_transfer is required}"
: "${cfg_scale:?cfg_scale is required}"
: "${suppress_tokens:?suppress_tokens is required}"
: "${begin_suppress_tokens:?begin_suppress_tokens is required}"
: "${task:?task is required}"
: "${num_fewshot:?num_fewshot is required}"
: "${batch_size:?batch_size is required}"
: "${wandb_project:?wandb_project is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
set -u

echo "===== Native LLaDA GSM8K run config ====="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "experiment_name=${experiment_name}"
echo "model=${model}"
echo "device=${device}"
echo "dtype=${dtype}"
echo "max_new_tokens=${max_new_tokens}"
echo "steps=${steps}"
echo "block_size=${block_size}"
echo "temperature=${temperature}"
echo "probability_temperature=${probability_temperature}"
echo "remasking=${remasking}"
echo "stochastic_transfer=${stochastic_transfer}"
echo "cfg_scale=${cfg_scale}"
echo "suppress_tokens=${suppress_tokens}"
echo "begin_suppress_tokens=${begin_suppress_tokens}"
echo "task=${task}"
echo "num_fewshot=${num_fewshot}"
echo "batch_size=${batch_size}"
echo "wandb_project=${wandb_project}"
echo "wandb_entity=${wandb_entity}"
echo "=========================================="

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

RESULT_PATH=".logs/llada_native_${task}_${experiment_name}_${SLURM_JOB_ID}.json"
RUN_NAME="llada-native-${task}-${experiment_name}-${SLURM_JOB_ID}"
MODEL_ARGS="pretrained=${model},device=${device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=${temperature},probability_temperature=${probability_temperature},remasking=${remasking},stochastic_transfer=${stochastic_transfer},cfg_scale=${cfg_scale},suppress_tokens=${suppress_tokens},begin_suppress_tokens=${begin_suppress_tokens}"

echo "RESULT_PATH=${RESULT_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_ARGS=${MODEL_ARGS}"

accelerate launch --num_processes 1 dllm/pipelines/llada/eval.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --model llada \
    --batch_size "${batch_size}" \
    --apply_chat_template \
    --output_path "${RESULT_PATH}" \
    --model_args "${MODEL_ARGS}"

WANDB_ARGS=(
    --result-path "${RESULT_PATH}"
    --project "${wandb_project}"
    --run-name "${RUN_NAME}"
    --mode llada_native
    --model-a "${model}"
    --probability-temperature "${probability_temperature}"
    --config-path "${CONFIG_PATH}"
)
if [[ -n "${wandb_entity}" ]]; then
    WANDB_ARGS+=(--entity "${wandb_entity}")
fi

python scripts/tse/log_eval_to_wandb.py "${WANDB_ARGS[@]}"
