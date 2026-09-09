#!/bin/bash

# Shared implementation for the four reduced Phase-8 full-evaluation jobs.
# Submit one of the run_{gsm8k,humaneval}_full_dependency_*.slurm.sh
# entrypoints instead of invoking this file directly.

# Conda CUDA hooks reference optional unset variables, so do not enable -u.
set -eo pipefail

repo=/home/sarthak.malla/dllm-selection-ensemble
cache=/home/sarthak.malla/.cache/huggingface/hub
model=models--GSAI-ML--LLaDA-8B-Instruct
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
checkpoint=${cache}/${model}/snapshots/${revision}
method=${P8_METHOD:-}
task=${P8_TASK:-}
seed=42
candidate_budget=4
candidate_chunk_size=1
num_gpu=2
run_tag=${P8_RUN_TAG:-reduced_full_v3}
wandb_mode=${P8_WANDB_MODE:-online}
wandb_project=${P8_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${P8_WANDB_ENTITY:-}
output_root=${repo}/eval_results/path_selection/dependency_guided/p8/${run_tag}/full

case "${method}" in
    dependency_fixed_k4_n4)
        commit_k=4
        cardinality_strategy=fixed
        size_scoring=raw
        action_size_label=4
        ;;
    dependency_entropy_budget_n4)
        commit_k=1
        cardinality_strategy=entropy_budget
        size_scoring=per_token
        action_size_label='1-4'
        ;;
    *)
        echo "P8_METHOD must be dependency_fixed_k4_n4 or dependency_entropy_budget_n4" >&2
        exit 2
        ;;
esac

case "${task}" in
    gsm8k_cot)
        max_new_tokens=256
        steps=64
        block_size=64
        num_fewshot=5
        unsafe_code=0
        ;;
    humaneval_instruct)
        max_new_tokens=1024
        steps=256
        block_size=256
        num_fewshot=0
        unsafe_code=1
        ;;
    *)
        echo "P8_TASK must be gsm8k_cot or humaneval_instruct" >&2
        exit 2
        ;;
esac

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "P8_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [ "${wandb_mode}" != disabled ] && [ "${wandb_mode}" != offline ] && [ "${wandb_mode}" != online ]; then
    echo "P8_WANDB_MODE must be disabled, offline, or online" >&2
    exit 2
fi
for value_name in wandb_project wandb_entity; do
    value=${!value_name}
    if [[ "${value}" == *,* ]]; then
        echo "${value_name} cannot contain a comma" >&2
        exit 2
    fi
done
if [ ! -d "${checkpoint}" ]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 1
fi

cd "${repo}"
mkdir -p "${repo}/.logs"
mkdir -p "${repo}/.tmp"
mkdir -p "${repo}/.wandb"
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
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=${repo}/.wandb
if [ "${task}" = humaneval_instruct ]; then
    # Python multiprocessing appends its own socket name below TMPDIR. Keep the
    # base path short enough for Linux's 108-byte AF_UNIX address limit.
    export TMPDIR=/tmp/p8-${SLURM_JOB_ID:-manual}
else
    export TMPDIR=${repo}/.tmp/p8-full-${SLURM_JOB_ID:-manual}-${method}-${task}
fi
mkdir -p "${TMPDIR}"

run_directory=${output_root}/${task}/${method}/seed${seed}
output_path=${run_directory}/results.json
completion_marker=${output_path}_entropy_drop_runtime.json
response_cache=${run_directory}/responses.cache
mkdir -p "${run_directory}"

if [ "${P8_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
    echo "Completion marker exists; skipping ${task}: ${completion_marker}"
    exit 0
fi

model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,diagnostic_retention=compact,sampler_type=entropy_drop,proposal_strategy=dependency,candidate_budget=${candidate_budget},candidate_chunk_size=${candidate_chunk_size},dependency_commit_k=${commit_k},dependency_parallel_variant=soft_full,dependency_cardinality_strategy=${cardinality_strategy},dependency_max_action_size=4,dependency_utility_threshold=0.0,dependency_entropy_budget=2.0,dependency_size_scoring=${size_scoring},dependency_immediate_cost_weight=1.0,dependency_size_penalty=0.0,dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=${seed},dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=max,dependency_conflict_penalty=1.0,dependency_hard_conflict_threshold=0.25,dependency_anchor_support_weight=1.0,dependency_anchor_confidence_threshold=0.8,diagnostic_metadata=true"

wandb_arguments=()
wandb_group=p8-${run_tag}-full-${method}
wandb_name=p8-${run_tag}-full-${task}-${method}-s${seed}
if [ "${wandb_mode}" != disabled ]; then
    wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=p8_full,mode=${wandb_mode},dir=${WANDB_DIR}"
    if [ -n "${wandb_entity}" ]; then
        wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
    fi
    wandb_config_args="phase=8,freeze=p8_reduced_full_v3,stage=full,task=${task},method=${method},proposal_strategy=dependency,candidate_budget=${candidate_budget},action_sizes=${action_size_label},cardinality_strategy=${cardinality_strategy},entropy_budget=2.0,size_scoring=${size_scoring},seed=${seed},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},candidate_chunk_size=${candidate_chunk_size},diagnostic_retention=compact,gpu_count=${num_gpu},run_tag=${run_tag}"
    wandb_arguments+=(
        --wandb_args "${wandb_init_args}"
        --wandb_config_args "${wandb_config_args}"
    )
fi

task_arguments=(
    --tasks "${task}"
    --batch_size 1
    --device cuda
    --seed 0,1234,1234,1234
    --apply_chat_template
    --num_fewshot "${num_fewshot}"
    --log_samples
)
if [ "${unsafe_code}" = 1 ]; then
    task_arguments+=(--confirm_run_unsafe_code)
fi

echo "Starting full Phase-8 task"
echo "Task/method: ${task}/${method}"
echo "Generation length/steps/block: ${max_new_tokens}/${steps}/${block_size}"
echo "Proposal/N/action sizes/cardinality: dependency/${candidate_budget}/${action_size_label}/${cardinality_strategy}"
echo "Accelerate processes/GPUs: ${num_gpu}/${num_gpu}"
echo "W&B mode/group/run: ${wandb_mode}/${wandb_group}/${wandb_name}"
echo "Output: ${run_directory}"

if [ "${P8_DRY_RUN:-0}" = 1 ]; then
    echo "P8_DRY_RUN=1; launch skipped"
    exit 0
fi

main_process_port=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
srun --ntasks=1 nvidia-smi
accelerate launch \
    --num_processes "${num_gpu}" \
    --num_machines 1 \
    --main_process_port "${main_process_port}" \
    --mixed_precision no \
    --dynamo_backend no \
    "${repo}/examples/path_selection/eval.py" \
        --model llada_path_selection \
        --model_args "${model_args}" \
        "${task_arguments[@]}" \
        "${wandb_arguments[@]}" \
        --output_path "${output_path}" \
        --use_cache "${response_cache}"

echo "Full Phase-8 task completed: ${task}/${method}"
