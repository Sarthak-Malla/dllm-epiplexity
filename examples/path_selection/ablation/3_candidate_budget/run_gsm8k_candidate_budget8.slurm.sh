#!/bin/bash
#SBATCH --job-name=abl3-cand8
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit with:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/3_candidate_budget/run_gsm8k_candidate_budget8.slurm.sh

export PATH_ABLATION_NUMBER=3
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=3_candidate_budget
export PATH_ABLATION_CANDIDATE_BUDGET=8
export PATH_ABLATION_WANDB_GROUP=ablation-3-candidate-budget
export PATH_ABLATION_WANDB_NAME_PREFIX=ablation-3-n8
export ABLATION2_ENTROPY_BUDGET=2.0
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION3_RUN_TAG:-candidate_budget8_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
