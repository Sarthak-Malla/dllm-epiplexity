#!/bin/bash
# Submit: sbatch scripts/tse/ctca/ctca_benchmark_replay_slurm.sh STUDY_ID [--case DOC:MASTER ...] [--condition ensemble]
#SBATCH --job-name=ctca-benchmark-replay
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclude=gpu-51,gpu-24,gpu-38,gpu-41
set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || true
source /apps/local/conda_init.sh
conda activate dllm
export PYTHONPATH=".:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CTCA_CANVAS_WORKERS=16
echo "job=${SLURM_JOB_ID:-manual} node=${SLURM_NODELIST:-unknown}"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader || true
python scripts/tse/cuda_preflight.py --minimum-device-count 2 --minimum-free-gib 30
study_id="${1:?Pass the approved study ID}"
shift
python scripts/tse/ctca/benchmark_replay.py --study-id "$study_id" "$@"
