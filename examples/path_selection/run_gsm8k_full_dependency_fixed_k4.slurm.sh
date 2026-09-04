#!/bin/bash
#SBATCH --job-name=path-p8-gsm-k4
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
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_gsm8k_full_dependency_fixed_k4.slurm.sh

export P8_TASK=gsm8k_cot
export P8_METHOD=dependency_fixed_k4_n4
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_p8_full_task.sh
