#!/bin/bash
#SBATCH --job-name=abl2-ent4-uncap
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
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget4_uncapped.slurm.sh

export ABLATION2_ENTROPY_BUDGET=4.0
export ABLATION2_MAXIMUM_ACTION_SIZE=${ABLATION2_MAXIMUM_ACTION_SIZE:-64}
export ABLATION2_RUN_TAG=${ABLATION2_RUN_TAG:-vectorized_soft_full_budget4_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
