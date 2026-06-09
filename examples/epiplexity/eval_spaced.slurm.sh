#!/usr/bin/env bash
#SBATCH --job-name=dllm-epiplexity-spaced
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G

set -euo pipefail

nvidia-smi

cd /home/sarthak.malla/dllm-epiplexity
mkdir -p .logs

# If your cluster uses modules, uncomment/edit these:
# module load cuda/12.1
# module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
# Conda activation scripts may reference unset variables.
# Disable nounset only during activation.
set +u
conda activate dllm
set -u

export PYTHONPATH=/home/sarthak.malla/dllm-epiplexity:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ===== Input Arguments =====
model_name_or_path="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
sampler_type="${SAMPLER_TYPE:-spaced_0}"
num_gpu="${SLURM_GPUS_ON_NODE:-1}"
output_dir="${OUTPUT_DIR:-/home/sarthak.malla/dllm-epiplexity/eval_results/spaced}"

if [[ "${sampler_type}" != "spaced_0" && "${sampler_type}" != "spaced_1" ]]; then
    echo "Error: SAMPLER_TYPE must be spaced_0 or spaced_1, got ${sampler_type}" >&2
    exit 1
fi

mkdir -p "${output_dir}/${sampler_type}"

model_args="pretrained=${model_name_or_path},max_new_tokens=256,steps=64,block_size=64,cfg_scale=0.0,sampler_type=${sampler_type}"

echo "================================================="
echo "Evaluating gsm8k_cot with sampler=${sampler_type}"
echo "Running on node: ${SLURM_NODELIST:-unknown}"
echo "GPUs requested: ${num_gpu}"
echo "Output directory: ${output_dir}/${sampler_type}/gsm8k"
echo "================================================="

accelerate launch \
    --num_processes "${num_gpu}" \
    /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/eval.py \
    --model llada_epiplexity \
    --apply_chat_template \
    --tasks gsm8k_cot \
    --num_fewshot 5 \
    --model_args "${model_args}" \
    --output_path "${output_dir}/${sampler_type}/gsm8k" \
    --use_cache "${output_dir}/${sampler_type}/gsm8k_just_spaced.cache"

echo -e "\n\nSpaced sampler GSM8K evaluation completed!"
