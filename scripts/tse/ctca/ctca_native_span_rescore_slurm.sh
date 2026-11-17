#!/bin/bash
# Submit: sbatch scripts/tse/ctca/ctca_native_span_rescore_slurm.sh STUDY_ID [--conditional]
#SBATCH --job-name=ctca-native-span-rescore
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
export HF_HUB_OFFLINE=1
echo "runner=$(realpath scripts/tse/ctca/native_span_rescore.py)"
sha256sum scripts/tse/ctca/native_span_rescore.py
git status --short
echo "job=${SLURM_JOB_ID:-manual} node=${SLURM_NODELIST:-unknown} gpus=${SLURM_GPUS:-unknown}"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader || true
study_id="${1:?Pass the study ID}"
shift
echo "study_id=${study_id} source_study_id=ctca-gsm8k4-benchmark-replay-20261029 top_k=8 max_span=3 extra_args=$*"
python scripts/tse/cuda_preflight.py --minimum-device-count 2 --minimum-free-gib 30
srun --ntasks=1 python scripts/tse/ctca/native_span_rescore.py --study-id "$study_id" "$@"
