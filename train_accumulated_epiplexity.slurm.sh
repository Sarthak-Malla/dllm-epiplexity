#!/bin/bash
#SBATCH --job-name=accum_epiplexity_sft
#SBATCH --gres=gpu:1
#SBATCH --exclude=ws-l5-012
#SBATCH --output=.logs/accum_epiplexity_sft_%j.out

# Source environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/dllm 

PYTHONNOUSERSITE=1 python -m experiments.epiplexity_accumulated_loss.llada_sft_accumulated_epiplexity \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5