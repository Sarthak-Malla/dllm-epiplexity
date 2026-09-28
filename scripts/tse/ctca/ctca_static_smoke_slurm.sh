#!/bin/bash
#SBATCH --job-name=ctca-static-smoke
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-54
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs

CONFIG_PATH="${1:-scripts/tse/configs/ctca_static_smoke.conf}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 1
fi
source "${CONFIG_PATH}"

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
: "${ctca_projection_temperature:?ctca_projection_temperature is required}"
: "${ctca_chunk_size:?ctca_chunk_size is required}"
: "${ctca_num_anchors:?ctca_num_anchors is required}"
: "${ctca_min_anchors:?ctca_min_anchors is required}"
: "${question:?question is required}"

source /apps/local/conda_init.sh
conda activate dllm
set -u

srun nvidia-smi

export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

REBUILD_ARGS=()
if [[ "${ctca_force_rebuild}" == "true" ]]; then
    REBUILD_ARGS+=(--ctca-force-rebuild)
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
    --ctca-projection-temperature "${ctca_projection_temperature}" \
    --ctca-chunk-size "${ctca_chunk_size}" \
    --ctca-num-anchors "${ctca_num_anchors}" \
    --ctca-min-anchors "${ctca_min_anchors}" \
    --question "${question}" \
    "${REBUILD_ARGS[@]}"
