#!/bin/bash

# Shared implementation for ablation 1. Submit one of the three .slurm.sh
# entrypoints in this directory instead of invoking this file directly.

set -eo pipefail

repo=/home/sarthak.malla/dllm-selection-ensemble
cache=/home/sarthak.malla/.cache/huggingface/hub
model=models--GSAI-ML--LLaDA-8B-Instruct
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
checkpoint=${cache}/${model}/snapshots/${revision}
selector=${ABLATION_SELECTOR:-}
seed=42
candidate_budget=4
entropy_budget=2.0
maximum_action_size=4
num_gpu=2
task=gsm8k_cot
run_tag=${ABLATION_RUN_TAG:-entropy_budget_full_v1}
wandb_mode=${ABLATION_WANDB_MODE:-online}
wandb_project=${ABLATION_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${ABLATION_WANDB_ENTITY:-}

case "${selector}" in
    max_confidence|min_entropy|min_top2_margin)
        ;;
    *)
        echo "ABLATION_SELECTOR must be max_confidence, min_entropy, or min_top2_margin" >&2
        exit 2
        ;;
esac

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ABLATION_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [ "${wandb_mode}" != disabled ] && [ "${wandb_mode}" != offline ] && [ "${wandb_mode}" != online ]; then
    echo "ABLATION_WANDB_MODE must be disabled, offline, or online" >&2
    exit 2
fi
if [ ! -d "${checkpoint}" ]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 1
fi

cd "${repo}"
mkdir -p "${repo}/.logs" "${repo}/.wandb"
mkdir -p /scratch/sarthak.malla/tmp/matplotlib

if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

export PYTHONPATH=${repo}:${PYTHONPATH:-}
export HF_HOME=/home/sarthak.malla/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=${repo}/.wandb
export TMPDIR=/tmp/dep-nl-${SLURM_JOB_ID:-manual}
mkdir -p "${TMPDIR}"

output_root=${repo}/eval_results/path_selection/ablation/1_dependecy_with_non_lookahead_samplers
run_directory=${output_root}/${run_tag}/${task}/${selector}/seed${seed}
output_path=${run_directory}/results.json
completion_marker=${output_path}_dependency_non_lookahead_runtime.json
response_cache=${run_directory}/responses.cache
mkdir -p "${run_directory}"

if [ "${ABLATION_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
    echo "Completion marker exists; skipping ${selector}: ${completion_marker}"
    exit 0
fi

# Match the successful Phase-8 entropy-budget proposal: four candidates are
# grown independently, and each stops at cumulative entropy 2.0 or four tokens.
# Candidate selection then uses only statistics from the current base pass.
model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=256,steps=64,block_size=64,temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,diagnostic_retention=none,sampler_type=dependency_non_lookahead,proposal_strategy=dependency,candidate_budget=${candidate_budget},dependency_candidate_selector=${selector},dependency_cardinality_strategy=entropy_budget,dependency_max_action_size=${maximum_action_size},dependency_entropy_budget=${entropy_budget},dependency_size_scoring=per_token,dependency_commit_k=1,dependency_parallel_variant=soft_full,dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=${seed},dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=max,dependency_conflict_penalty=1.0,dependency_hard_conflict_threshold=0.25,dependency_anchor_support_weight=1.0,dependency_anchor_confidence_threshold=0.8,diagnostic_metadata=false"

wandb_arguments=()
wandb_group=ablation-1-entropy-budget-non-lookahead
wandb_name=ablation-1-${task}-${selector}-s${seed}
if [ "${wandb_mode}" != disabled ]; then
    wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=ablation,mode=${wandb_mode},dir=${WANDB_DIR}"
    if [ -n "${wandb_entity}" ]; then
        wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
    fi
    wandb_config_args="ablation=1,task=${task},selector=${selector},lookahead=false,proposal_strategy=dependency,candidate_budget=${candidate_budget},cardinality_strategy=entropy_budget,entropy_budget=${entropy_budget},maximum_action_size=${maximum_action_size},size_scoring=per_token,seed=${seed},max_new_tokens=256,steps=64,block_size=64,gpu_count=${num_gpu},run_tag=${run_tag}"
    wandb_arguments+=(
        --wandb_args "${wandb_init_args}"
        --wandb_config_args "${wandb_config_args}"
    )
fi

echo "Starting dependency/non-lookahead ablation"
echo "Task/selector: ${task}/${selector}"
echo "Candidate proposal/count: dependency/${candidate_budget}"
echo "Cardinality/budget/max size: entropy_budget/${entropy_budget}/${maximum_action_size}"
echo "Lookahead forwards: 0"
echo "Output: ${run_directory}"

if [ "${ABLATION_DRY_RUN:-0}" = 1 ]; then
    echo "ABLATION_DRY_RUN=1; launch skipped"
    exit 0
fi

main_process_port=$((24000 + ${SLURM_JOB_ID:-0} % 16000))
accelerate launch \
    --num_processes "${num_gpu}" \
    --num_machines 1 \
    --main_process_port "${main_process_port}" \
    --mixed_precision no \
    --dynamo_backend no \
    "${repo}/examples/path_selection/eval.py" \
        --model llada_path_selection \
        --model_args "${model_args}" \
        --tasks "${task}" \
        --batch_size 1 \
        --device cuda \
        --seed 0,1234,1234,1234 \
        --apply_chat_template \
        --num_fewshot 5 \
        --log_samples \
        "${wandb_arguments[@]}" \
        --output_path "${output_path}" \
        --use_cache "${response_cache}"

echo "Ablation completed: ${task}/${selector}"
