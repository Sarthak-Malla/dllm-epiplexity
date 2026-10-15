#!/bin/bash
# Execute one immutable CTCA parity configuration inside a Slurm allocation.

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs .cache/lm_eval_responses
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:?usage: ctca_parity_eval.sh CONFIG_PATH}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
tse_print_config_file "${CONFIG_PATH}"
source "${CONFIG_PATH}"

: "${protocol_id:?protocol_id is required}"
: "${experiment_name:?experiment_name is required}"
: "${model_a:?model_a is required}"
: "${model_b:?model_b is required}"
: "${master_model:?master_model is required}"
: "${task:?task is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${block_size:?block_size is required}"
: "${batch_size:?batch_size is required}"
: "${master_suppress_tokens:?master_suppress_tokens is required}"
: "${master_begin_suppress_tokens:?master_begin_suppress_tokens is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CTCA_CANVAS_WORKERS="${ctca_canvas_workers}"

srun nvidia-smi
srun --ntasks=1 --nodes=1 python scripts/tse/cuda_preflight.py \
    --minimum-device-count 2

ATTEMPT_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR=".logs/${protocol_id}/${experiment_name}/${ATTEMPT_ID}"
RESULT_PATH="${RUN_DIR}/results.json"
CACHE_PATH=".cache/lm_eval_responses/${protocol_id}/${experiment_name}"
RUN_NAME="${experiment_name}-${ATTEMPT_ID}"
mkdir -p "${RUN_DIR}" "$(dirname "${CACHE_PATH}")"

MODEL_ARGS="model_a=${model_a},model_b=${model_b},model_a_device=${model_a_device},model_b_device=${model_b_device},fusion_device=${fusion_device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},selection_mode=${selection_mode},weighting_mode=${weighting_mode},alpha=${alpha},temperature_a=${temperature_a},temperature_b=${temperature_b},weight_temperature=${weight_temperature},temperature=${temperature},remasking=${remasking},stochastic_transfer=${stochastic_transfer},capture_logits=${capture_logits},normalize_entropy=${normalize_entropy},epsilon=${epsilon},ctca_enabled=${ctca_enabled},master_model=${master_model},master_suppress_tokens=${master_suppress_tokens},master_begin_suppress_tokens=${master_begin_suppress_tokens},ctca_cache_dir=${ctca_cache_dir},ctca_force_rebuild=${ctca_force_rebuild},ctca_anchor_temperature=${ctca_anchor_temperature},ctca_projection_temperature=${ctca_projection_temperature},ctca_chunk_size=${ctca_chunk_size},ctca_projection_mode=${ctca_projection_mode},ctca_projection_top_k=${ctca_projection_top_k},ctca_num_anchors=${ctca_num_anchors},ctca_min_anchors=${ctca_min_anchors}"
MODEL_ARGS="${MODEL_ARGS},progress_wandb=${progress_wandb},progress_wandb_project=${progress_wandb_project},progress_wandb_group=${protocol_id},progress_wandb_run_name=${RUN_NAME}-progress,progress_wandb_log_interval=${progress_wandb_log_interval}"
if [[ -n "${progress_wandb_entity}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},progress_wandb_entity=${progress_wandb_entity}"
fi

LIMIT_ARGS=()
if [[ -n "${limit}" ]]; then
    LIMIT_ARGS+=(--limit "${limit}")
fi
UNSAFE_ARGS=()
if [[ "${confirm_run_unsafe_code}" == "true" ]]; then
    UNSAFE_ARGS+=(--confirm_run_unsafe_code)
fi

echo "===== CTCA parity run ====="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "protocol_id=${protocol_id}"
echo "experiment_name=${experiment_name}"
echo "task=${task}"
echo "master_model=${master_model}"
echo "master_suppress_tokens=${master_suppress_tokens}"
echo "master_begin_suppress_tokens=${master_begin_suppress_tokens}"
echo "RESULT_PATH=${RESULT_PATH}"
echo "CACHE_PATH=${CACHE_PATH}"
echo "MODEL_ARGS=${MODEL_ARGS}"
echo "==========================="

srun --ntasks=1 --nodes=1 accelerate launch --num_processes 1 \
    dllm/pipelines/tse/eval.py \
    --tasks "${task}" \
    --num_fewshot "${num_fewshot}" \
    --model tse_llada \
    --batch_size "${batch_size}" \
    --apply_chat_template \
    --seed 0,1234,1234,1234 \
    --cache_requests true \
    --use_cache "${CACHE_PATH}" \
    --log_samples \
    --output_path "${RESULT_PATH}" \
    --model_args "${MODEL_ARGS}" \
    "${LIMIT_ARGS[@]}" \
    "${UNSAFE_ARGS[@]}"

WANDB_ARGS=(
    --result-path "${RESULT_PATH}"
    --project "${progress_wandb_project}"
    --group "${protocol_id}"
    --run-name "${RUN_NAME}"
    --mode ctca
    --weighting-mode "${weighting_mode}"
    --model-a "${model_a}"
    --model-b "${model_b}"
    --temperature-a "${temperature_a}"
    --temperature-b "${temperature_b}"
    --alpha "${alpha}"
    --config-path "${CONFIG_PATH}"
)
if [[ -n "${progress_wandb_entity}" ]]; then
    WANDB_ARGS+=(--entity "${progress_wandb_entity}")
fi
python scripts/tse/log_eval_to_wandb.py "${WANDB_ARGS[@]}"
