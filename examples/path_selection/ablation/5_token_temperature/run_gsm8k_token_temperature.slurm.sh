#!/bin/bash
#SBATCH --job-name=abl5-temperature
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit both temperatures:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature.slurm.sh
# Array task 0 uses 0.5; task 1 uses 1.0. For a single run, use --array=0
# with --export=ALL,ABLATION5_TEMPERATURE=0.5 (or 1.0).

set -eo pipefail

if [ -n "${ABLATION5_TEMPERATURE:-}" ]; then
    token_temperature=${ABLATION5_TEMPERATURE}
else
    case "${SLURM_ARRAY_TASK_ID:-0}" in
        0) token_temperature=0.5 ;;
        1) token_temperature=1.0 ;;
        *)
            echo "Ablation 5 requires array task 0 or 1" >&2
            exit 2
            ;;
    esac
fi
case "${token_temperature}" in
    0.5) ;;
    1|1.0) token_temperature=1.0 ;;
    *)
        echo "ABLATION5_TEMPERATURE must be 0.5 or 1.0" >&2
        exit 2
        ;;
esac

export PATH_ABLATION_NUMBER=5
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=5_token_temperature
export PATH_ABLATION_CANDIDATE_BUDGET=4
export PATH_ABLATION_CARDINALITY_STRATEGY=entropy_budget
export PATH_ABLATION_TOKEN_TEMPERATURE=${token_temperature}
export PATH_ABLATION_WANDB_GROUP=ablation-5-token-temperature
export PATH_ABLATION_WANDB_NAME_PREFIX=ablation-5-n4
export ABLATION2_ENTROPY_BUDGET=2.0
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION5_RUN_TAG:-token_temperature_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
