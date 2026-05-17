#!/bin/bash
#SBATCH --job-name=val_duel_sft
#SBATCH --gres=gpu:1
#SBATCH --output=.logs/val_duel_sft_%j.out

# Source environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/dllm 

PYTHONNOUSERSITE=1 python -m experiments.validation_based_duel.llada_sft_validation_duel \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5