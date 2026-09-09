#!/bin/bash
#SBATCH --job-name=abl1-dep-conf
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit with:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/run_max_confidence.slurm.sh

export ABLATION_SELECTOR=max_confidence
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/run_common.sh
