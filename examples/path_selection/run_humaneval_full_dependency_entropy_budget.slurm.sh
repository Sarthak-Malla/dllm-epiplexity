#!/bin/bash
#SBATCH --job-name=path-p8-he-entropy
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

# Submit with:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_humaneval_full_dependency_entropy_budget.slurm.sh

export P8_TASK=humaneval_instruct
export P8_METHOD=dependency_entropy_budget_n4
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_p8_full_task.sh
