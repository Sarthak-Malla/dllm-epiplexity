#!/bin/bash
#SBATCH --job-name=dream-parity-suite
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs .cache/lm_eval_responses
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

if (( $# > 0 )); then
    CONFIG_PATHS=("$@")
else
    CONFIG_PATHS=(
        "scripts/tse/configs/dream_gsm8k300_parity.conf"
        "scripts/tse/configs/dream_humaneval_parity.conf"
    )
fi
for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Configuration file not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
    tse_print_config_file "${CONFIG_PATH}"
done

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

srun nvidia-smi
srun --ntasks=1 --nodes=1 python scripts/tse/cuda_preflight.py \
    --minimum-device-count 1

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    (
        source "${CONFIG_PATH}"

        : "${protocol_id:?protocol_id is required}"
        : "${experiment_name:?experiment_name is required}"
        : "${model:?model is required}"
        : "${task:?task is required}"
        : "${max_new_tokens:?max_new_tokens is required}"
        : "${steps:?steps is required}"
        : "${batch_size:?batch_size is required}"
        : "${suppress_tokens:?suppress_tokens is required}"
        : "${begin_suppress_tokens:?begin_suppress_tokens is required}"

        ATTEMPT_ID="${SLURM_JOB_ID:-manual}"
        RUN_DIR=".logs/${protocol_id}/${experiment_name}/${ATTEMPT_ID}"
        RESULT_PATH="${RUN_DIR}/results.json"
        CACHE_PATH=".cache/lm_eval_responses/${protocol_id}/${experiment_name}"
        RUN_NAME="${experiment_name}-${ATTEMPT_ID}"
        mkdir -p "${RUN_DIR}" "$(dirname "${CACHE_PATH}")"

        MODEL_ARGS="pretrained=${model},device=${device},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},temperature=${temperature},top_p=${top_p},alg=${alg},alg_temp=${alg_temp},stochastic_transfer=${stochastic_transfer},right_shift_logits=${right_shift_logits},cfg_scale=${cfg_scale},add_bos_token=${add_bos_token},suppress_tokens=${suppress_tokens},begin_suppress_tokens=${begin_suppress_tokens}"
        if [[ -n "${top_k}" ]]; then
            MODEL_ARGS="${MODEL_ARGS},top_k=${top_k}"
        fi
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

        echo "===== Dream parity run ====="
        echo "CONFIG_PATH=${CONFIG_PATH}"
        echo "protocol_id=${protocol_id}"
        echo "experiment_name=${experiment_name}"
        echo "task=${task}"
        echo "RESULT_PATH=${RESULT_PATH}"
        echo "CACHE_PATH=${CACHE_PATH}"
        echo "MODEL_ARGS=${MODEL_ARGS}"
        echo "============================"

        srun --ntasks=1 --nodes=1 accelerate launch --num_processes 1 \
            dllm/pipelines/dream/eval.py \
            --tasks "${task}" \
            --num_fewshot "${num_fewshot}" \
            --model dream \
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
            --mode dream_control
            --model-a "${model}"
            --config-path "${CONFIG_PATH}"
        )
        if [[ -n "${progress_wandb_entity}" ]]; then
            WANDB_ARGS+=(--entity "${progress_wandb_entity}")
        fi
        python scripts/tse/log_eval_to_wandb.py "${WANDB_ARGS[@]}"
    )
done
