#!/bin/bash
#SBATCH --job-name=ctca-m0-one-question
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-scripts/tse/configs/ctca_static_smoke.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
tse_print_config_file "${CONFIG_PATH}"
source "${CONFIG_PATH}"

ctca_canvas_workers="${ctca_canvas_workers:-8}"
trace_top_k="${trace_top_k:-5}"
trace_jsonl="${trace_jsonl:-.logs/ctca-m0-one-question_${SLURM_JOB_ID:-manual}.jsonl}"
trace_tensors="${trace_tensors:-}"
master_suppress_tokens="${master_suppress_tokens:-}"
master_begin_suppress_tokens="${master_begin_suppress_tokens:-}"
read -r -a MASTER_SUPPRESS_ARGS <<< "${master_suppress_tokens}"
read -r -a MASTER_BEGIN_SUPPRESS_ARGS <<< "${master_begin_suppress_tokens}"

: "${model_a:?model_a is required}"
: "${model_b:?model_b is required}"
: "${master_model:?master_model is required}"
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
: "${top_k:?top_k is required}"
: "${ctca_cache_dir:?ctca_cache_dir is required}"
: "${ctca_force_rebuild:?ctca_force_rebuild is required}"
: "${ctca_anchor_temperature:?ctca_anchor_temperature is required}"
: "${ctca_projection_temperature:?ctca_projection_temperature is required}"
: "${ctca_chunk_size:?ctca_chunk_size is required}"
: "${ctca_projection_mode:?ctca_projection_mode is required}"
: "${ctca_projection_top_k:?ctca_projection_top_k is required}"
: "${ctca_num_anchors:?ctca_num_anchors is required}"
: "${ctca_min_anchors:?ctca_min_anchors is required}"
: "${ctca_canvas_workers:?ctca_canvas_workers is required}"
: "${trace_top_k:?trace_top_k is required}"
: "${trace_jsonl:?trace_jsonl is required}"
: "${question:?question is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
set -u

echo "===== CTCA smoke run config ====="
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "model_a=${model_a}"
echo "model_b=${model_b}"
echo "master_model=${master_model}"
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
echo "top_k=${top_k}"
echo "ctca_cache_dir=${ctca_cache_dir}"
echo "ctca_force_rebuild=${ctca_force_rebuild}"
echo "ctca_anchor_temperature=${ctca_anchor_temperature}"
echo "ctca_projection_temperature=${ctca_projection_temperature}"
echo "ctca_chunk_size=${ctca_chunk_size}"
echo "ctca_projection_mode=${ctca_projection_mode}"
echo "ctca_projection_top_k=${ctca_projection_top_k}"
echo "ctca_num_anchors=${ctca_num_anchors}"
echo "ctca_min_anchors=${ctca_min_anchors}"
echo "ctca_canvas_workers=${ctca_canvas_workers}"
echo "trace_top_k=${trace_top_k}"
echo "trace_jsonl=${trace_jsonl}"
echo "trace_tensors=${trace_tensors}"
echo "master_suppress_tokens=${master_suppress_tokens}"
echo "master_begin_suppress_tokens=${master_begin_suppress_tokens}"
echo "question=${question}"
echo "================================="

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TSE_TIMERS=1
export CTCA_CANVAS_WORKERS="${ctca_canvas_workers}"

REBUILD_ARGS=()
if [[ "${ctca_force_rebuild}" == "true" ]]; then
    REBUILD_ARGS+=(--ctca-force-rebuild)
fi

TRACE_TENSOR_ARGS=()
if [[ -n "${trace_tensors}" ]]; then
    TRACE_TENSOR_ARGS+=(--trace-tensors "${trace_tensors}")
fi

QUESTION_METADATA_ARGS=()
if [[ -n "${question_label:-}" ]]; then
    QUESTION_METADATA_ARGS+=(--question-label "${question_label}")
fi
if [[ -n "${gold_answer:-}" ]]; then
    QUESTION_METADATA_ARGS+=(--gold-answer "${gold_answer}")
fi
if [[ -n "${source_doc_id:-}" ]]; then
    QUESTION_METADATA_ARGS+=(--source-doc-id "${source_doc_id}")
fi
if [[ -n "${source_correct:-}" ]]; then
    QUESTION_METADATA_ARGS+=(--source-correct "${source_correct}")
fi
if [[ -n "${source_job_id:-}" ]]; then
    QUESTION_METADATA_ARGS+=(--source-job-id "${source_job_id}")
fi

python scripts/tse/smoke.py \
    --model-a "${model_a}" \
    --model-b "${model_b}" \
    --model-a-device "${model_a_device}" \
    --model-b-device "${model_b_device}" \
    --fusion-device "${fusion_device}" \
    --dtype "${dtype}" \
    --max-new-tokens "${max_new_tokens}" \
    --steps "${steps}" \
    --block-size "${block_size}" \
    --selection-mode "${selection_mode}" \
    --weighting-mode "${weighting_mode}" \
    --alpha "${alpha}" \
    --temperature-a "${temperature_a}" \
    --temperature-b "${temperature_b}" \
    --weight-temperature "${weight_temperature}" \
    --temperature "${temperature}" \
    --remasking "${remasking}" \
    --top-k "${top_k}" \
    --ctca \
    --master-model "${master_model}" \
    --ctca-cache-dir "${ctca_cache_dir}" \
    --ctca-anchor-temperature "${ctca_anchor_temperature}" \
    --ctca-projection-temperature "${ctca_projection_temperature}" \
    --ctca-chunk-size "${ctca_chunk_size}" \
    --ctca-projection-mode "${ctca_projection_mode}" \
    --ctca-projection-top-k "${ctca_projection_top_k}" \
    --ctca-num-anchors "${ctca_num_anchors}" \
    --ctca-min-anchors "${ctca_min_anchors}" \
    --trace-top-k "${trace_top_k}" \
    --trace-jsonl "${trace_jsonl}" \
    --master-suppress-tokens "${MASTER_SUPPRESS_ARGS[@]}" \
    --master-begin-suppress-tokens "${MASTER_BEGIN_SUPPRESS_ARGS[@]}" \
    --question "${question}" \
    "${QUESTION_METADATA_ARGS[@]}" \
    "${REBUILD_ARGS[@]}" \
    "${TRACE_TENSOR_ARGS[@]}"
