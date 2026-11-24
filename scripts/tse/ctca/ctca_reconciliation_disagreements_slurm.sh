#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_reconciliation_disagreements_slurm.sh
#SBATCH --job-name=ctca-reconcile-disagreements
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
echo "Pilot job ${SLURM_JOB_ID:?Submit with sbatch} on ${SLURM_NODELIST:-unknown}"
exec python -u /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/reconciliation_disagreements.py "$@"
