#!/bin/bash
#SBATCH --job-name=abl5-temp0.5
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit with:
#   sbatch --exclude=gpu-51 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/5_token_temperature/run_gsm8k_token_temperature0.5.slurm.sh

set -eo pipefail

export PATH_ABLATION_NUMBER=5
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=5_token_temperature
export PATH_ABLATION_CANDIDATE_BUDGET=4
export PATH_ABLATION_CARDINALITY_STRATEGY=entropy_budget
export PATH_ABLATION_TOKEN_TEMPERATURE=0.5
export PATH_ABLATION_WANDB_GROUP=ablation-5-token-temperature
export PATH_ABLATION_WANDB_NAME_PREFIX=ablation-5-n4
export ABLATION2_ENTROPY_BUDGET=2.0
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION5_RUN_TAG:-token_temperature_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
