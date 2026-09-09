#!/bin/bash
#SBATCH --job-name=abl6-marginal-confidence
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-7%2
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit the eight configurations on 128 GSM8K questions:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/6_marginal_utility_confidence/run_gsm8k_marginal_utility_confidence.slurm.sh
# Set ABLATION6_LIMIT=full to evaluate all questions. Override --array to
# select configurations; each array task uses two GPUs, with two tasks at once.

set -eo pipefail

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) confidence_exponent=0.0; utility_threshold=1.0 ;;
    1) confidence_exponent=0.0; utility_threshold=2.0 ;;
    2) confidence_exponent=0.0; utility_threshold=4.0 ;;
    3) confidence_exponent=1.0; utility_threshold=0.0 ;;
    4) confidence_exponent=1.0; utility_threshold=0.5 ;;
    5) confidence_exponent=1.0; utility_threshold=1.0 ;;
    6) confidence_exponent=1.0; utility_threshold=2.0 ;;
    7) confidence_exponent=1.0; utility_threshold=4.0 ;;
    *)
        echo "Ablation 6 requires array task 0 through 7" >&2
        exit 2
        ;;
esac

evaluation_limit=${ABLATION6_LIMIT:-128}
if [ "${evaluation_limit}" = full ]; then
    evaluation_limit=
fi

export PATH_ABLATION_NUMBER=6
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=6_marginal_utility_confidence
export PATH_ABLATION_CANDIDATE_BUDGET=4
export PATH_ABLATION_CARDINALITY_STRATEGY=marginal_utility
export PATH_ABLATION_UTILITY_THRESHOLD=${utility_threshold}
export PATH_ABLATION_CONFIDENCE_EXPONENT=${confidence_exponent}
export PATH_ABLATION_TOKEN_TEMPERATURE=0.0
export PATH_ABLATION_LIMIT=${evaluation_limit}
export PATH_ABLATION_WANDB_GROUP=ablation-6-marginal-utility-confidence
export PATH_ABLATION_WANDB_NAME_PREFIX=ablation-6-n4
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION6_RUN_TAG:-marginal_confidence_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
