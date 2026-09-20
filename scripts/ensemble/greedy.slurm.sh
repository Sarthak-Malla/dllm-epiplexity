#!/bin/bash
#SBATCH --job-name=baseline-gsm8k-greedy
#SBATCH --output=/home/sarthak.malla/dllm-learning-decoding-path/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-learning-decoding-path/.logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

# The common launcher initializes Conda and reads the shared LLaDA config.
# Preview without submitting: bash this-script --dry-run.
exec bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh \
    greedy "$@"
