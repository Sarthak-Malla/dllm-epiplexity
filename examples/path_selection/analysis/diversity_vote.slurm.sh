#!/bin/bash
#SBATCH --job-name=path-diversity-vote
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# CPU only. Submit manually:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/analysis/diversity_vote.slurm.sh
set -eo pipefail
if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:${PYTHONPATH:-}
export OMP_NUM_THREADS=2
srun --ntasks=1 python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_historical_diversity_vote.py -q
srun --ntasks=1 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/analysis/diversity_vote.py \
    --output-directory /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/diversity_vote_table_job${SLURM_JOB_ID}
