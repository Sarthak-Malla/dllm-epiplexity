#!/bin/bash
#SBATCH --job-name=dream-native-m1-one-question
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-scripts/tse/configs/dream_native_m1_smoke.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
tse_print_config_file "${CONFIG_PATH}"
source "${CONFIG_PATH}"

: "${model:?model is required}"
: "${device:?device is required}"
: "${dtype:?dtype is required}"
: "${prompt_style:?prompt_style is required}"
: "${max_new_tokens:?max_new_tokens is required}"
: "${steps:?steps is required}"
: "${temperature:?temperature is required}"
: "${top_p:?top_p is required}"
: "${top_k:?top_k is required}"
: "${alg:?alg is required}"
: "${alg_temp:?alg_temp is required}"
: "${right_shift_logits:?right_shift_logits is required}"
: "${seed:?seed is required}"
: "${trace_top_k:?trace_top_k is required}"
: "${trace_all_logits:?trace_all_logits is required}"
: "${trace_jsonl:?trace_jsonl is required}"
trace_tensors="${trace_tensors:-}"
: "${question:?question is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
set -u

echo "===== Native Dream M1 smoke run config ====="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "model=${model}"
echo "device=${device}"
echo "prompt_style=${prompt_style}"
echo "max_new_tokens=${max_new_tokens}"
echo "steps=${steps}"
echo "temperature=${temperature}"
echo "top_p=${top_p}"
echo "top_k=${top_k}"
echo "alg=${alg}"
echo "right_shift_logits=${right_shift_logits}"
echo "trace_jsonl=${trace_jsonl}"
echo "trace_tensors=${trace_tensors}"
echo "question=${question}"
echo "============================================="

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TSE_TIMERS=1

SHIFT_ARGS=(--right-shift-logits)
if [[ "${right_shift_logits}" != "true" ]]; then
    SHIFT_ARGS=(--no-right-shift-logits)
fi

TRACE_ARGS=()
if [[ "${trace_all_logits}" == "true" ]]; then
    TRACE_ARGS+=(--trace-all-logits)
fi
if [[ -n "${trace_tensors}" ]]; then
    TRACE_ARGS+=(--trace-tensors "${trace_tensors}")
fi

python scripts/tse/dream_native_m1_smoke.py \
    --model "${model}" \
    --device "${device}" \
    --dtype "${dtype}" \
    --prompt-style "${prompt_style}" \
    --max-new-tokens "${max_new_tokens}" \
    --steps "${steps}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --alg "${alg}" \
    --alg-temp "${alg_temp}" \
    --seed "${seed}" \
    --trace-top-k "${trace_top_k}" \
    --trace-jsonl "${trace_jsonl}" \
    --question "${question}" \
    "${SHIFT_ARGS[@]}" \
    "${TRACE_ARGS[@]}"
