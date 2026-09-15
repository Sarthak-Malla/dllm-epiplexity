#!/bin/bash
# Source from a suite Slurm launcher; never execute experiment Python on a login node.
# Submit the shared collection first:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/collect.slurm.sh

set -eo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Use sbatch to run a suite launcher on a compute node." >&2
    exit 2
fi
if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:/home/sarthak.malla/dllm-selection-ensemble/lm-evaluation-harness:"${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

TF_RUN_TAG=${TF_RUN_TAG:-training_free_v1_two_gpu}
if [[ ! "$TF_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "TF_RUN_TAG must contain only letters, digits, dots, underscores, or hyphens." >&2
    exit 2
fi
TF_OUTPUT_ROOT=/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/"$TF_RUN_TAG"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-dllm-selection-ensemble}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$TF_RUN_TAG}
export WANDB_DIR="$TF_OUTPUT_ROOT"

run_training_free() {
    # One Slurm task owns both GPUs; launch.py starts two independent workers.
    # A common root is locked against concurrent stages; completed units resume.
    srun --ntasks=1 --kill-on-bad-exit=1 \
        python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py \
        --output-root "$TF_OUTPUT_ROOT" -- \
        --resume --wandb-mode "$WANDB_MODE" \
        --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_RUN_GROUP" "$@"
}
