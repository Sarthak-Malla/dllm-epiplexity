#!/bin/bash
#SBATCH --job-name=epiplexity_sft
#SBATCH --gres=gpu:1
#SBATCH --output=.logs/epiplexity_%j.out

# Source environment
source ~/.bashrc  # or ~/.zshrc depending on your shell
conda activate ~/miniconda3/envs/dllm 

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m epiplexity.llada_sft_epiplexity \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5
