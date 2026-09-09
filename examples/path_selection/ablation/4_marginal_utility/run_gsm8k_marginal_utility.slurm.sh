#!/bin/bash
#SBATCH --job-name=abl4-marginal
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
#   sbatch --export=ALL,ABLATION4_UTILITY_THRESHOLD=0.25 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/4_marginal_utility/run_gsm8k_marginal_utility.slurm.sh

export PATH_ABLATION_NUMBER=4
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=4_marginal_utility
export PATH_ABLATION_CANDIDATE_BUDGET=4
export PATH_ABLATION_CARDINALITY_STRATEGY=marginal_utility
export PATH_ABLATION_UTILITY_THRESHOLD=${ABLATION4_UTILITY_THRESHOLD:-0.0}
export PATH_ABLATION_WANDB_GROUP=ablation-4-marginal-utility
export PATH_ABLATION_WANDB_NAME_PREFIX=ablation-4-n4
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION4_RUN_TAG:-marginal_utility_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
