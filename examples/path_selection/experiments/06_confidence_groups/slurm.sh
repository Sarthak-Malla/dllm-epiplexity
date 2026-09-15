#!/bin/bash
#SBATCH --job-name=tf-conf-groups
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
#SBATCH --array=0-5%1

# Submit manually: sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/06_confidence_groups/slurm.sh
# All arms use GSM8K documents 0–99, seed 42, BF16, and two workers.
# W&B defaults online; export WANDB_MODE=offline for local logs.
# Time is an initial allocation estimate; rerun this command to resume after timeout.
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh

# Match confidence versus incoming ranking at block64 and all three thresholds.
# Already completed confidence baselines from stage05 resume without reevaluation.
case "${SLURM_ARRAY_TASK_ID:-}" in
    0) arm=threshold_0.80_block64_confidence ;;
    1) arm=threshold_0.80_block64_incoming ;;
    2) arm=threshold_0.90_block64_confidence ;;
    3) arm=threshold_0.90_block64_incoming ;;
    4) arm=threshold_0.95_block64_confidence ;;
    5) arm=threshold_0.95_block64_incoming ;;
    *) echo "Expected array index 0 through 5." >&2; exit 2 ;;
esac
run_training_free benchmark --arm "$arm"
