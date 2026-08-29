#!/bin/bash
#SBATCH --job-name=path-gsm8k-full-entropy-drop
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

cd /home/sarthak.malla/dllm-selection-ensemble

mkdir -p /home/sarthak.malla/dllm-selection-ensemble/.logs

export TMPDIR=/home/sarthak.malla/dllm-selection-ensemble/.tmp/slurm-${SLURM_JOB_ID:-manual}
mkdir -p "${TMPDIR}"

source /apps/local/conda_init.sh
conda activate dllm

srun nvidia-smi
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/sarthak.malla/.cache/huggingface}"

model_name_or_path="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
num_gpu="${SLURM_GPUS_ON_NODE:-2}"

output_root=/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection
run_name=gsm8k_full_entropy_drop

mkdir -p "${output_root}/${run_name}"

model_args="pretrained=${model_name_or_path},max_new_tokens=256,steps=64,block_size=64,cfg_scale=0.0,sampler_type=entropy_drop,oracle_candidate_strategy=mixed"

echo "Running full GSM8K entropy drop evaluation"
echo "Model: ${model_name_or_path}"
echo "Accelerate processes: ${num_gpu}"
echo "Output: ${output_root}/${run_name}"


accelerate launch \
    --num_processes "${num_gpu}" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/eval.py \
    --model llada_path_selection \
    --apply_chat_template \
    --tasks gsm8k_cot \
    --num_fewshot 5 \
    --log_samples \
    --model_args "${model_args}" \
    --output_path "${output_root}/${run_name}" \
    --use_cache "${output_root}/${run_name}/responses.cache"

echo "Full GSM8K entropy drop evaluation completed."
