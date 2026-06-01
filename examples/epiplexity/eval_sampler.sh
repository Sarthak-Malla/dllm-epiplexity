#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 
export HF_DATASETS_TRUST_REMOTE_CODE=True   

# ===== Input Arguments =====
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
sampler_type="greedy"  # greedy, oracle, guided
num_gpu=1
output_dir="eval_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --sampler_type) sampler_type="$2"; shift 2 ;;
    --num_gpu) num_gpu="$2"; shift 2 ;;
    --output_dir) output_dir="$2"; shift 2 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$output_dir/$sampler_type"

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

base_model_args="pretrained=${model_name_or_path},max_new_tokens=512,steps=64,block_size=64,cfg_scale=0.0"
model_args="${base_model_args},sampler_type=${sampler_type}"

for task_key in "${!ALL_TASKS[@]}"; do
    task_args=${ALL_TASKS[$task_key]}
    echo "================================================="
    echo "Evaluating $task_key with sampler=$sampler_type..."
    echo "================================================="
    
    accelerate launch --num_processes "${num_gpu}" examples/epiplexity/eval.py         --model llada_epiplexity         --apply_chat_template         --tasks $task_args         --model_args "${model_args}"         --output_path "${output_dir}/${sampler_type}/${task_key}"
done

echo "\n\nAll evaluations completed!"
