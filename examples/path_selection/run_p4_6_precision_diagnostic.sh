#!/bin/bash
#SBATCH --job-name=path-p4-precision
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

# Conda CUDA hooks reference optional unset variables, so do not enable -u.
set -eo pipefail

repo=/home/sarthak.malla/dllm-selection-ensemble
cache=/home/sarthak.malla/.cache/huggingface/hub
model=models--GSAI-ML--LLaDA-8B-Instruct
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
checkpoint=${cache}/${model}/snapshots/${revision}
script=${repo}/examples/path_selection/dependency_guided/benchmark_batched_lookahead.py
output_dir=${repo}/eval_results/path_selection/dependency_guided/p4_6_precision_diagnostic

cd "${repo}"

mkdir -p "${repo}/.logs"
mkdir -p "${repo}/.tmp"
mkdir -p "${output_dir}"
mkdir -p /scratch/sarthak.malla/tmp/matplotlib

export TMPDIR=${repo}/.tmp/slurm-${SLURM_JOB_ID:-manual}
mkdir -p "${TMPDIR}"

if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

export PYTHONPATH=${repo}:${PYTHONPATH:-}
export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
export HF_HOME=${HF_HOME:-/home/sarthak.malla/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

if [ ! -d "${checkpoint}" ]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 1
fi

run_precision() {
    local dtype=$1
    local output=${output_dir}/${dtype}.json

    echo "Running controlled ${dtype} lookahead diagnostic"
    echo "Output: ${output}"
    srun --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
        python "${script}" \
            --checkpoint "${checkpoint}" \
            --output-path "${output}" \
            --device cuda:0 \
            --dtype "${dtype}" \
            --metrics entropy_drop risk_reduction \
            --prompt-length 32 \
            --response-length 64 \
            --candidate-budgets 2 4 8 \
            --chunk-sizes 2 4 \
            --warmup-runs 1 \
            --measurement-runs 3 \
            --score-atol 1e-4 \
            --seed 42
}

srun --ntasks=1 nvidia-smi
run_precision bfloat16
run_precision float32

echo "Phase 4.6 precision diagnostic completed"
echo "Results: ${output_dir}"
