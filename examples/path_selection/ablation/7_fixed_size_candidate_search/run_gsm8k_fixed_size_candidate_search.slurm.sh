#!/bin/bash
#SBATCH --job-name=abl7-fixed-k4
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-3%2
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-51
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# Submit the original four arms on the same 300 GSM8K test documents:
#   sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_fixed_size_candidate_search.slurm.sh
# Array mapping: 0=D4, 1=D8, 2=C4, 3=C8, 4=CD4, 5=CD8.
# The default array remains 0-3; use --array=4 or --array=5 for the new arms.
# Two tasks may run concurrently within the default array.
# Each task uses two GPUs: four GPUs total when both tasks are running.
# Optional shared controls: ABLATION7_RUN_TAG, ABLATION7_GENERATION_SEED,
# ABLATION7_WANDB_MODE. Each task always uses --limit 300.

set -eo pipefail

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) arm=D4 ;;
    1) arm=D8 ;;
    2) arm=C4 ;;
    3) arm=C8 ;;
    4) arm=CD4 ;;
    5) arm=CD8 ;;
    *)
        echo "Submit this script with sbatch; array task must be 0 through 5." >&2
        exit 2
        ;;
esac

# Match the environment setup used by the existing ablation launchers.
if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

run_tag=${ABLATION7_RUN_TAG:-fixed_k_v1}
generation_seed=${ABLATION7_GENERATION_SEED:-42}
wandb_mode=${ABLATION7_WANDB_MODE:-disabled}

echo "Ablation 7: ${arm}, two GPUs, GSM8K documents 0-299, seed ${generation_seed}, run ${run_tag}"
# The step inherits this sbatch allocation, including any submission overrides.
# Keep one Slurm task: run_ablation.py starts both evaluation workers itself.
srun --ntasks=1 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_ablation.py \
        --arm "${arm}" --limit 300 --run-tag "${run_tag}" \
        --generation-seed "${generation_seed}" --wandb-mode "${wandb_mode}"
