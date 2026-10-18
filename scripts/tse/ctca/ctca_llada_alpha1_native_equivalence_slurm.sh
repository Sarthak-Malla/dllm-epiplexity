#!/bin/bash
#SBATCH --job-name=ctca-llada-alpha1-equiv
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05,gpu-02
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

CONFIG_PATH="${1:-scripts/tse/configs/ctca_llada_alpha1_native_equivalence.conf}"
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
: "${output_json:?output_json is required}"

source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
set -u

mkdir -p "$(dirname "${output_json}")"
export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TSE_TIMERS=1
export CTCA_CANVAS_WORKERS="${ctca_canvas_workers}"

srun python scripts/tse/cuda_preflight.py --minimum-device-count 2
srun python scripts/tse/experiments/verify_llada_alpha1_native_equivalence.py \
    --model-a "${model_a}" \
    --model-b "${model_b}" \
    --model-a-device "${model_a_device}" \
    --model-b-device "${model_b_device}" \
    --fusion-device "${fusion_device}" \
    --dtype "${dtype}" \
    --max-new-tokens "${max_new_tokens}" \
    --steps "${steps}" \
    --block-size "${block_size}" \
    --seed "${seed}" \
    --cache-dir "${ctca_cache_dir}" \
    --anchor-temperature "${ctca_anchor_temperature}" \
    --projection-temperature "${ctca_projection_temperature}" \
    --projection-chunk-size "${ctca_chunk_size}" \
    --projection-mode "${ctca_projection_mode}" \
    --projection-top-k "${ctca_projection_top_k}" \
    --num-anchors "${ctca_num_anchors}" \
    --min-anchors "${ctca_min_anchors}" \
    --output-json "${output_json}"
