#!/bin/bash
#SBATCH --job-name=dream-native-gsm8k
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-scripts/tse/configs/dream_native_gsm8k.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
tse_print_config_file "${CONFIG_PATH}"
source "${CONFIG_PATH}"

wandb_project="${wandb_project:-dllm-tse}"
wandb_entity="${wandb_entity:-}"

: "${experiment_name:?experiment_name is required}"
: "${model:?model is required}"
: "${device:?device is required}"
: "${dtype:?dtype is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${temperature:?temperature is required}"
: "${top_p:?top_p is required}"
: "${top_k:?top_k is required}"
: "${alg:?alg is required}"
: "${alg_temp:?alg_temp is required}"
: "${stochastic_transfer:?stochastic_transfer is required}"
: "${right_shift_logits:?right_shift_logits is required}"
: "${cfg_scale:?cfg_scale is required}"
: "${task:?task is required}"
: "${num_fewshot:?num_fewshot is required}"
: "${batch_size:?batch_size is required}"
: "${wandb_project:?wandb_project is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
set -u

echo "===== Native Dream GSM8K run config ====="
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
echo "temperature=${temperature}"
echo "top_p=${top_p}"
echo "top_k=${top_k}"
echo "alg=${alg}"
echo "alg_temp=${alg_temp}"
echo "stochastic_transfer=${stochastic_transfer}"
echo "right_shift_logits=${right_shift_logits}"
echo "cfg_scale=${cfg_scale}"
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

RESULT_PATH=".logs/dream_native_${task}_${experiment_name}_${SLURM_JOB_ID}.json"
RUN_NAME="dream-native-${task}-${experiment_name}-${SLURM_JOB_ID}"
MODEL_ARGS="pretrained=${model},device=${device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},temperature=${temperature},top_p=${top_p},top_k=${top_k},alg=${alg},alg_temp=${alg_temp},stochastic_transfer=${stochastic_transfer},right_shift_logits=${right_shift_logits},cfg_scale=${cfg_scale}"

echo "MODEL_ARGS=${MODEL_ARGS}"

accelerate launch --num_processes 1 dllm/pipelines/dream/eval.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --model dream \
    --batch_size "${batch_size}" \
    --apply_chat_template \
    --output_path "${RESULT_PATH}" \
    --model_args "${MODEL_ARGS}"

WANDB_ARGS=(
    --result-path "${RESULT_PATH}"
    --project "${wandb_project}"
    --run-name "${RUN_NAME}"
    --mode dream_native
    --model-a "${model}"
    --config-path "${CONFIG_PATH}"
)
if [[ -n "${wandb_entity}" ]]; then
    WANDB_ARGS+=(--entity "${wandb_entity}")
fi

python scripts/tse/log_eval_to_wandb.py "${WANDB_ARGS[@]}"
