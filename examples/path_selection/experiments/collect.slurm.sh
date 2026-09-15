#!/bin/bash
#SBATCH --job-name=tf-collect
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# Submit manually: sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/collect.slurm.sh
# All arms use GSM8K documents 0–99, seed 42, BF16, and two workers.
# W&B defaults online; export WANDB_MODE=offline for local logs.
# Time is an initial allocation estimate; rerun this command to resume after timeout.
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh

# Collect 100 problems, with three shared states each (300 states total).
run_training_free collect
