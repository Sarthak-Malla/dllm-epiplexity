#!/bin/bash
#SBATCH --job-name=abl7-seed
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-5%2
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

# Submit the six seed arms (two GPUs each; at most two arms concurrently):
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_corrected_seed.slurm.sh
# Mapping: 0=I4, 1=I8, 2=IE4, 3=IE8, 4=CS4, 5=CS8.
# Optional 6=CD4, 7=CD8 rerun the legacy seed under the same sampler sources.
# No pilot. All arms evaluate GSM8K documents 0-299 with four-token actions.

set -eo pipefail

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) arm=I4 ;;
    1) arm=I8 ;;
    2) arm=IE4 ;;
    3) arm=IE8 ;;
    4) arm=CS4 ;;
    5) arm=CS8 ;;
    6) arm=CD4 ;;
    7) arm=CD8 ;;
    *) echo "Submit with sbatch; array task must be 0 through 7." >&2; exit 2 ;;
esac

if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

run_tag=${ABLATION7_RUN_TAG:-fixed_k_seed_v1}
generation_seed=${ABLATION7_GENERATION_SEED:-42}
wandb_mode=${ABLATION7_WANDB_MODE:-disabled}

echo "Seed ablation: ${arm}, two GPUs, GSM8K documents 0-299, seed ${generation_seed}, run ${run_tag}"
srun --ntasks=1 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_ablation.py \
        --arm "${arm}" --limit 300 --run-tag "${run_tag}" \
        --generation-seed "${generation_seed}" --wandb-mode "${wandb_mode}"
