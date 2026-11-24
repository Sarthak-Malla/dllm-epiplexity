#!/bin/bash
#SBATCH --job-name=ctca-gsm8k-greedy
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .logs
baseline_slot="${1:-}"
CONFIG_PATH="${2:-scripts/tse/configs/ctca_gsm8k_greedy.conf}"
if [[ "${baseline_slot}" == "--config" ]]; then
    CONFIG_PATH="${2:?Pass the baseline config path after --config}"
    baseline_slot=""
fi
source "${CONFIG_PATH}"

baseline_slot="${baseline_slot:-${baseline_model:-}}"
case "${baseline_slot}" in
    a)
        pretrained="${model_a}"
        probability_temperature="${temperature_a}"
        ;;
    b)
        pretrained="${model_b}"
        probability_temperature="${temperature_b}"
        ;;
    *)
        echo "Pass a for Model A, b for Model B, or --config with a baseline config." >&2
        exit 2
        ;;
esac

source /apps/local/conda_init.sh
conda activate dllm
set -u
export TOKENIZERS_PARALLELISM=false

RESULT_PATH=".logs/ctca_${task}_${experiment_name}_greedy_${baseline_slot}_${SLURM_JOB_ID}.json"
MODEL_ARGS="pretrained=${pretrained},dtype=${dtype},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature_a=${probability_temperature},temperature=${temperature:-0.0},remasking=${remasking:-low_confidence},stochastic_transfer=${stochastic_transfer:-false}"
eval_args=(
    --model tse_llada
    --model_args "${MODEL_ARGS}"
    --tasks "${task}"
    --num_fewshot "${num_fewshot}"
    --batch_size "${batch_size:-1}"
    --apply_chat_template
    --output_path "${RESULT_PATH}"
)
if [[ -n "${limit:-}" ]]; then
    eval_args+=(--limit "${limit}")
fi
python -m dllm.pipelines.tse.eval "${eval_args[@]}"
