#!/bin/bash
#SBATCH --job-name=tf-first-action-full
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# Submit manually:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/full_benchmark.slurm.sh /absolute/path/to/task.json
# Optional arguments after the task JSON select policies; otherwise run all three.
# Repeating the same submission resumes completed documents with the same sources.
# gpu-54 also logged CUDA driver initialization failures in jobs 240498/240499;
# keep it excluded until its GPU initialization has been checked on the cluster.
set -eo pipefail
TF_BENCHMARK_CONFIG=${1:?Pass the absolute path to a benchmark task JSON configuration.}
shift
if [ "$#" -eq 0 ]; then
    set -- reference_cheap first_action_entropy reference_entropy
fi
for TF_ARM in "$@"; do
    case "$TF_ARM" in
        reference_cheap|first_action_entropy|reference_entropy) ;;
        *) echo "Unknown benchmark policy: $TF_ARM" >&2; exit 2 ;;
    esac
done
if [[ "$TF_BENCHMARK_CONFIG" != /* ]] || [ ! -f "$TF_BENCHMARK_CONFIG" ]; then
    echo "Benchmark configuration must be an existing absolute path." >&2
    exit 2
fi
TF_CONFIG_NAME=$(basename "$TF_BENCHMARK_CONFIG" .json)
export TF_RUN_TAG=${TF_RUN_TAG:-first_action_full_${TF_CONFIG_NAME}_two_gpu}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh
export TMPDIR=/tmp/tf-benchmark-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

for TF_ARM in "$@"; do
    srun --ntasks=1 --kill-on-bad-exit=1 \
        python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py \
        --benchmark-config "$TF_BENCHMARK_CONFIG" --output-root "$TF_OUTPUT_ROOT" -- \
        --resume --arm "$TF_ARM" --wandb-mode "$WANDB_MODE" \
        --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_RUN_GROUP"
done

srun --ntasks=1 --kill-on-bad-exit=1 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/analyze.py \
    --output-root "$TF_OUTPUT_ROOT"
