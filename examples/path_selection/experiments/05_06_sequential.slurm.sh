#!/bin/bash
#SBATCH --job-name=tf-blocks-conf
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# One queue slot: six stage-05 arms, then three new stage-06 arms.
# Submit manually:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_06_sequential.slurm.sh
# No job arrays, nested submissions, or node exclusions. The two-GPU launcher
# runs 100 documents per arm. Time is an initial estimate; completed units resume
# on resubmission with the same TF_RUN_TAG and unchanged configuration/sources.
set -eo pipefail
export TF_RUN_TAG=${TF_RUN_TAG:-training_free_v1_05_06_two_gpu}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$TF_RUN_TAG}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh

for TF_BENCHMARK_ARM in \
    threshold_0.80_block64_confidence \
    threshold_0.80_block256_confidence \
    threshold_0.90_block64_confidence \
    threshold_0.90_block256_confidence \
    threshold_0.95_block64_confidence \
    threshold_0.95_block256_confidence \
    threshold_0.80_block64_incoming \
    threshold_0.90_block64_incoming \
    threshold_0.95_block64_incoming; do
    echo "Running benchmark: $TF_BENCHMARK_ARM"
    run_training_free benchmark --arm "$TF_BENCHMARK_ARM"
done
