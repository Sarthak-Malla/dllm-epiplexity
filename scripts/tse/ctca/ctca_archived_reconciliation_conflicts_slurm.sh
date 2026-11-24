#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_archived_reconciliation_conflicts_slurm.sh
#SBATCH --job-name=ctca-real-conflicts
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --partition=cscc-cpu-p
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
echo "Audit job ${SLURM_JOB_ID:?Submit with sbatch} on ${SLURM_NODELIST:-unknown}"
exec python -u /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/archived_reconciliation_conflicts.py "$@"
