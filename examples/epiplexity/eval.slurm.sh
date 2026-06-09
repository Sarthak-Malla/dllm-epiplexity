#!/usr/bin/env bash
#SBATCH --job-name=dllm-epiplexity-sampler
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G

set -euo pipefail

nvidia-smi

mkdir -p .logs

# ===== Environment =====
export PYTHONPATH=.:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

# Optional but often useful on clusters
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# If your cluster uses modules, uncomment/edit these:
# module load cuda/12.1
# module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
# Conda activation scripts may reference unset variables.
# Disable nounset only during activation.
set +u
conda activate dllm
set -u

# ===== Input Arguments =====
model_name_or_path="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
sampler_type="${SAMPLER_TYPE:-greedy}"  # greedy, oracle, guided
num_gpu="${SLURM_GPUS_ON_NODE:-1}"
output_dir="${OUTPUT_DIR:-eval_results}"

mkdir -p "${output_dir}/${sampler_type}"

declare -A ALL_TASKS=(
    # ["mmlu"]="mmlu_generative --num_fewshot 0"
    # ["mmlu_pro"]="mmlu_pro --num_fewshot 0"
    # ["arc_c"]="arc_challenge_chat --num_fewshot 0"
    # ["hellaswag"]="hellaswag_gen --num_fewshot 0"
    ["gsm8k"]="gsm8k_cot --num_fewshot 5"
    # ["math"]="minerva_math --num_fewshot 4"
    # ["gpqa"]="gpqa_diamond_generative_n_shot --num_fewshot 5"
    # ["humaneval"]="humaneval --num_fewshot 0 --confirm_run_unsafe_code"
    # ["mbpp"]="mbpp --num_fewshot 3 --confirm_run_unsafe_code"
)

base_model_args="pretrained=${model_name_or_path},max_new_tokens=256,steps=64,block_size=64,cfg_scale=0.0"
model_args="${base_model_args},sampler_type=${sampler_type}"

for task_key in "${!ALL_TASKS[@]}"; do
    task_args=${ALL_TASKS[$task_key]}

    echo "================================================="
    echo "Evaluating ${task_key} with sampler=${sampler_type}"
    echo "Running on node: ${SLURM_NODELIST:-unknown}"
    echo "GPUs requested: ${num_gpu}"
    echo "================================================="

    accelerate launch \
        --num_processes "${num_gpu}" \
        examples/epiplexity/eval.py \
        --model llada_epiplexity \
        --apply_chat_template \
        --tasks ${task_args} \
        --model_args "${model_args}" \
        --output_path "${output_dir}/${sampler_type}/${task_key}" \
        --use_cache "${output_dir}/${sampler_type}/${task_key}_without_greedy.cache"
done

echo -e "\n\nAll evaluations completed!"