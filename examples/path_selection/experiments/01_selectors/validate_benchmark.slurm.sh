#!/bin/bash
#SBATCH --job-name=tf-first-action-tests
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# Submit separately from the benchmark:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/validate_benchmark.slurm.sh
# Synthetic tests use CPU tensors and a fake W&B client; no GPUs are requested.
set -eo pipefail
export TF_RUN_TAG=${TF_RUN_TAG:-first_action_validation}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export TMPDIR=/tmp/tf-first-action-tests-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

srun --ntasks=1 --kill-on-bad-exit=1 python -m pytest -q \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_first_action_benchmark.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_runner.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_analysis.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_telemetry.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_imports.py "$@"
