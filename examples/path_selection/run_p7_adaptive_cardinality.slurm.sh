#!/bin/bash
#SBATCH --job-name=path-p7-cardinality
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-21%4
#SBATCH --time=03:00:00
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
stage=${P7_STAGE:-calibration}
verifier=${P7_VERIFIER:-entropy_drop}
max_new_tokens=${P7_MAX_NEW_TOKENS:-64}
candidate_budget=${P7_CANDIDATE_BUDGET:-4}
candidate_chunk_size=${P7_CANDIDATE_CHUNK_SIZE:-1}
maximum_action_size=${P7_MAX_ACTION_SIZE:-4}
action_sizes=${P7_ACTION_SIZES:-1|2|4}
wandb_mode=${P7_WANDB_MODE:-disabled}
wandb_project=${P7_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${P7_WANDB_ENTITY:-}
output_root=${repo}/eval_results/path_selection/dependency_guided/p7
run_tag=${P7_RUN_TAG:-}
if [ -n "${run_tag}" ]; then
    if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "P7_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
        exit 2
    fi
    output_root=${output_root}/${run_tag}
fi
output_root=${output_root}/${stage}
wandb_group=${P7_WANDB_GROUP:-p7-${run_tag:-default}-${stage}-${verifier}}

tasks=(gsm8k_cot humaneval_instruct)
if [ "${stage}" = calibration ]; then
    policies=(
        fixed_k2
        scheduler
        marginal_tau0
        marginal_tau0.25
        marginal_tau0.5
        joint_per_token
        joint_immediate_cost
        joint_size_penalty
        entropy_budget0.5
        entropy_budget1.0
        entropy_budget2.0
    )
    sample_manifest=${P7_SAMPLE_MANIFEST:-${repo}/examples/path_selection/dependency_guided/p7_calibration_samples.json}
elif [ "${stage}" = comparison ]; then
    policies=(fixed_k2 scheduler marginal_utility joint_k entropy_budget)
    sample_manifest=${P7_SAMPLE_MANIFEST:-${repo}/examples/path_selection/dependency_guided/p7_evaluation_samples.json}
elif [ "${stage}" = confirmation ]; then
    policies=(fixed_k2 fixed_k4 entropy_budget)
    sample_manifest=${P7_SAMPLE_MANIFEST:-${repo}/examples/path_selection/dependency_guided/p7_confirmation_samples.json}
else
    echo "P7_STAGE must be calibration, comparison, or confirmation, got ${stage}" >&2
    exit 2
fi
policy_count=${#policies[@]}
logical_cell_count=$((2 * policy_count))

if [ "${verifier}" != entropy_drop ]; then
    echo "P7 freezes entropy_drop verification, got ${verifier}" >&2
    exit 2
fi
if [ "${wandb_mode}" != disabled ] && [ "${wandb_mode}" != offline ] && [ "${wandb_mode}" != online ]; then
    echo "P7_WANDB_MODE must be disabled, offline, or online, got ${wandb_mode}" >&2
    exit 2
fi
for value_name in wandb_project wandb_entity wandb_group; do
    value=${!value_name}
    if [[ "${value}" == *,* ]]; then
        echo "${value_name} cannot contain a comma" >&2
        exit 2
    fi
done
for value_name in max_new_tokens candidate_budget candidate_chunk_size maximum_action_size; do
    value=${!value_name}
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done
if ! [[ "${action_sizes}" =~ ^[1-9][0-9]*(\|[1-9][0-9]*)*$ ]]; then
    echo "P7_ACTION_SIZES must use a form such as 1|2|4, got ${action_sizes}" >&2
    exit 2
fi
if [ ! -f "${sample_manifest}" ]; then
    echo "Sample manifest not found: ${sample_manifest}" >&2
    exit 1
fi
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
export HF_HOME=${HF_HOME:-/home/sarthak.malla/.cache/huggingface}
export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=${repo}/.wandb

logical_indices=()
worker_count=${P7_WORKER_COUNT:-}
if [ -n "${worker_count}" ]; then
    if ! [[ "${worker_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "P7_WORKER_COUNT must be a positive integer, got ${worker_count}" >&2
        exit 2
    fi
    worker_index=${SLURM_ARRAY_TASK_ID:-0}
    if [ "${worker_index}" -ge "${worker_count}" ]; then
        echo "Array task ${worker_index} is outside P7_WORKER_COUNT=${worker_count}" >&2
        exit 2
    fi
    for ((logical_index=worker_index; logical_index<logical_cell_count; logical_index+=worker_count)); do
        logical_indices+=("${logical_index}")
    done
else
    logical_indices+=("${SLURM_ARRAY_TASK_ID:-0}")
fi

if [ "${P7_DRY_RUN:-0}" != 1 ]; then
    srun --ntasks=1 nvidia-smi
fi

for array_index in "${logical_indices[@]}"; do
    if [ "${array_index}" -lt 0 ] || [ "${array_index}" -ge "${logical_cell_count}" ]; then
        echo "Logical matrix index must be between 0 and $((logical_cell_count - 1)), got ${array_index}" >&2
        exit 2
    fi
    task_index=$((array_index / policy_count))
    policy_index=$((array_index % policy_count))
    task=${tasks[${task_index}]}
    policy=${policies[${policy_index}]}
    cardinality_strategy=fixed
    commit_k=2
    steps=$((max_new_tokens / 2))
    utility_threshold=0.0
    entropy_budget=1.0
    size_scoring=raw
    immediate_cost_weight=1.0
    size_penalty=0.0

    case "${policy}" in
        fixed_k2)
            ;;
        fixed_k4)
            if [ $((max_new_tokens % 4)) -ne 0 ]; then
                echo "fixed_k4 requires P7_MAX_NEW_TOKENS divisible by 4" >&2
                exit 2
            fi
            commit_k=4
            steps=$((max_new_tokens / 4))
            ;;
        scheduler)
            cardinality_strategy=scheduler
            ;;
        marginal_tau0)
            cardinality_strategy=marginal_utility
            steps=${max_new_tokens}
            size_scoring=per_token
            ;;
        marginal_tau0.25)
            cardinality_strategy=marginal_utility
            steps=${max_new_tokens}
            utility_threshold=0.25
            size_scoring=per_token
            ;;
        marginal_tau0.5)
            cardinality_strategy=marginal_utility
            steps=${max_new_tokens}
            utility_threshold=0.5
            size_scoring=per_token
            ;;
        marginal_utility)
            cardinality_strategy=marginal_utility
            steps=${max_new_tokens}
            utility_threshold=${P7_UTILITY_THRESHOLD:-0.25}
            size_scoring=${P7_MARGINAL_SIZE_SCORING:-per_token}
            ;;
        joint_per_token)
            cardinality_strategy=joint_k
            steps=${max_new_tokens}
            size_scoring=per_token
            ;;
        joint_immediate_cost)
            cardinality_strategy=joint_k
            steps=${max_new_tokens}
            size_scoring=immediate_cost
            ;;
        joint_size_penalty)
            cardinality_strategy=joint_k
            steps=${max_new_tokens}
            size_scoring=size_penalty
            size_penalty=1.0
            ;;
        joint_k)
            cardinality_strategy=joint_k
            steps=${max_new_tokens}
            size_scoring=${P7_JOINT_SIZE_SCORING:-per_token}
            immediate_cost_weight=${P7_IMMEDIATE_COST_WEIGHT:-1.0}
            size_penalty=${P7_SIZE_PENALTY:-1.0}
            ;;
        entropy_budget0.5)
            cardinality_strategy=entropy_budget
            steps=${max_new_tokens}
            entropy_budget=0.5
            size_scoring=per_token
            ;;
        entropy_budget1.0)
            cardinality_strategy=entropy_budget
            steps=${max_new_tokens}
            entropy_budget=1.0
            size_scoring=per_token
            ;;
        entropy_budget2.0)
            cardinality_strategy=entropy_budget
            steps=${max_new_tokens}
            entropy_budget=2.0
            size_scoring=per_token
            ;;
        entropy_budget)
            cardinality_strategy=entropy_budget
            steps=${max_new_tokens}
            entropy_budget=${P7_ENTROPY_BUDGET:-1.0}
            size_scoring=${P7_ENTROPY_SIZE_SCORING:-per_token}
            ;;
        *)
            echo "Unsupported Phase-7 policy: ${policy}" >&2
            exit 2
            ;;
    esac

    run_directory=${output_root}/${verifier}/${task}/${policy}/n${candidate_budget}
    output_path=${run_directory}/results.json
    completion_marker=${output_path}_${verifier}_runtime.json
    mkdir -p "${run_directory}"
    export TMPDIR=${repo}/.tmp/slurm-${SLURM_ARRAY_JOB_ID:-manual}-${array_index}
    mkdir -p "${TMPDIR}"

    model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${steps},block_size=${max_new_tokens},temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,sampler_type=${verifier},proposal_strategy=dependency,candidate_budget=${candidate_budget},candidate_chunk_size=${candidate_chunk_size},dependency_commit_k=${commit_k},dependency_parallel_variant=soft_full,dependency_cardinality_strategy=${cardinality_strategy},dependency_max_action_size=${maximum_action_size},dependency_action_sizes=${action_sizes},dependency_utility_threshold=${utility_threshold},dependency_entropy_budget=${entropy_budget},dependency_size_scoring=${size_scoring},dependency_immediate_cost_weight=${immediate_cost_weight},dependency_size_penalty=${size_penalty},dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=42,dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=max,dependency_conflict_penalty=1.0,dependency_hard_conflict_threshold=0.25,dependency_anchor_support_weight=1.0,dependency_anchor_confidence_threshold=0.8,diagnostic_metadata=true"
    wandb_arguments=()
    wandb_name=p7-${run_tag:-default}-${stage}-${task}-${policy}-n${candidate_budget}
    if [ "${wandb_mode}" != disabled ]; then
        wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=p7_${stage},mode=${wandb_mode},dir=${WANDB_DIR}"
        if [ -n "${wandb_entity}" ]; then
            wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
        fi
        wandb_config_args="phase=7,stage=${stage},task=${task},policy=${policy},verifier=${verifier},candidate_budget_per_size=${candidate_budget},maximum_action_size=${maximum_action_size},action_sizes=${action_sizes},utility_threshold=${utility_threshold},entropy_budget=${entropy_budget},size_scoring=${size_scoring},sample_manifest=${sample_manifest},run_tag=${run_tag:-default},slurm_logical_index=${array_index}"
        wandb_arguments+=(
            --wandb_args "${wandb_init_args}"
            --wandb_config_args "${wandb_config_args}"
        )
    fi

    task_arguments=(
        --tasks "${task}"
        --batch_size 1
        --device cuda
        --samples "${sample_manifest}"
        --seed 42
        --apply_chat_template
        --log_samples
    )
    if [ "${task}" = gsm8k_cot ]; then
        task_arguments+=(--num_fewshot 5)
    else
        task_arguments+=(--num_fewshot 0 --confirm_run_unsafe_code)
    fi

    echo "Starting Phase-7 adaptive-cardinality cell"
    echo "Logical matrix index: ${array_index}/${logical_cell_count}"
    echo "Stage/task/policy: ${stage}/${task}/${policy}"
    echo "Cardinality strategy: ${cardinality_strategy}"
    echo "Configured commit k / requested steps: ${commit_k}/${steps}"
    echo "Candidate budget: ${candidate_budget} per action size"
    echo "Maximum/allowed action sizes: ${maximum_action_size}/${action_sizes}"
    echo "Utility threshold / entropy budget: ${utility_threshold}/${entropy_budget}"
    echo "Size scoring: ${size_scoring}"
    echo "Sample manifest: ${sample_manifest}"
    echo "W&B mode: ${wandb_mode}"
    echo "Output: ${run_directory}"

    if [ "${P7_DRY_RUN:-0}" = 1 ]; then
        echo "P7_DRY_RUN=1; launch skipped"
        continue
    fi
    if [ "${P7_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
        echo "Completion marker exists; skipping: ${completion_marker}"
        continue
    fi

    srun --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
        python "${repo}/examples/path_selection/eval.py" \
            --model llada_path_selection \
            --model_args "${model_args}" \
            "${task_arguments[@]}" \
            "${wandb_arguments[@]}" \
            --output_path "${output_path}"

    echo "Phase-7 adaptive-cardinality cell completed: ${run_directory}"
done
