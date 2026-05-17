#!/bin/bash
#SBATCH --job-name=experiment_without_epiplexity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=.logs/exp_without_epiplexity_%j.out

# Source environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/dllm 

PYTHONNOUSERSITE=1 python -m experiments.without_epiplexity.oracle_upper_bound \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --load_in_4bit True
