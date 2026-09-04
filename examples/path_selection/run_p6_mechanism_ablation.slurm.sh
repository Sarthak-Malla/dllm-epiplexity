#!/bin/bash
#SBATCH --job-name=path-p6-ablation
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-23%4
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
verifier=${P6_VERIFIER:-entropy_drop}
limit=${P6_LIMIT:-8}
max_new_tokens=${P6_MAX_NEW_TOKENS:-64}
candidate_budget=${P6_CANDIDATE_BUDGET:-4}
candidate_chunk_size=${P6_CANDIDATE_CHUNK_SIZE:-1}
conflict_normalization=${P6_CONFLICT_NORMALIZATION:-max}
conflict_penalty=${P6_CONFLICT_PENALTY:-1.0}
hard_conflict_threshold=${P6_HARD_CONFLICT_THRESHOLD:-0.25}
anchor_support_weight=${P6_ANCHOR_SUPPORT_WEIGHT:-1.0}
anchor_confidence_threshold=${P6_ANCHOR_CONFIDENCE_THRESHOLD:-0.8}
wandb_mode=${P6_WANDB_MODE:-disabled}
wandb_project=${P6_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${P6_WANDB_ENTITY:-}
output_root=${repo}/eval_results/path_selection/dependency_guided/p6_7
run_tag=${P6_RUN_TAG:-}
if [ -n "${run_tag}" ]; then
    if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "P6_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
        exit 2
    fi
    output_root=${output_root}/${run_tag}
fi
wandb_group=${P6_WANDB_GROUP:-p6-${run_tag:-default}-${verifier}}

tasks=(gsm8k_cot humaneval_instruct)
variants=(
    correlated_together
    top_confidence
    hard_low_conflict
    anchor_support_only
    soft_no_anchor
    soft_full
)
commit_sizes=(2 4)
logical_cell_count=24

if [ "${verifier}" != entropy_drop ] && [ "${verifier}" != risk_reduction ]; then
    echo "P6_VERIFIER must be entropy_drop or risk_reduction, got ${verifier}" >&2
    exit 2
fi
if [ "${wandb_mode}" != disabled ] && [ "${wandb_mode}" != offline ] && [ "${wandb_mode}" != online ]; then
    echo "P6_WANDB_MODE must be disabled, offline, or online, got ${wandb_mode}" >&2
    exit 2
fi
for value_name in wandb_project wandb_entity wandb_group; do
    value=${!value_name}
    if [[ "${value}" == *,* ]]; then
        echo "${value_name} cannot contain a comma" >&2
        exit 2
    fi
done
for value_name in limit max_new_tokens candidate_budget candidate_chunk_size; do
    value=${!value_name}
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
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
export HF_HOME=${HF_HOME:-/home/sarthak.malla/.cache/huggingface}
export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=${repo}/.wandb

logical_indices=()
worker_count=${P6_WORKER_COUNT:-}
if [ -n "${worker_count}" ]; then
    if ! [[ "${worker_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "P6_WORKER_COUNT must be a positive integer, got ${worker_count}" >&2
        exit 2
    fi
    worker_index=${SLURM_ARRAY_TASK_ID:-0}
    if [ "${worker_index}" -ge "${worker_count}" ]; then
        echo "Array task ${worker_index} is outside P6_WORKER_COUNT=${worker_count}" >&2
        exit 2
    fi
    for ((logical_index=worker_index; logical_index<logical_cell_count; logical_index+=worker_count)); do
        logical_indices+=("${logical_index}")
    done
else
    logical_indices+=("${SLURM_ARRAY_TASK_ID:-0}")
fi

if [ "${P6_DRY_RUN:-0}" != 1 ]; then
    srun --ntasks=1 nvidia-smi
fi

for array_index in "${logical_indices[@]}"; do
    if [ "${array_index}" -lt 0 ] || [ "${array_index}" -ge "${logical_cell_count}" ]; then
        echo "Logical matrix index must be between 0 and 23, got ${array_index}" >&2
        exit 2
    fi
    task_index=$((array_index / 12))
    combination_index=$((array_index % 12))
    variant_index=$((combination_index / 2))
    commit_index=$((combination_index % 2))
    task=${tasks[${task_index}]}
    variant=${variants[${variant_index}]}
    commit_k=${commit_sizes[${commit_index}]}
    if [ $((max_new_tokens % commit_k)) -ne 0 ]; then
        echo "P6_MAX_NEW_TOKENS=${max_new_tokens} must be divisible by k=${commit_k}" >&2
        exit 2
    fi
    steps=$((max_new_tokens / commit_k))
    sampler_type=${verifier}

    run_directory=${output_root}/${verifier}/${task}/${variant}/k${commit_k}/n${candidate_budget}
    output_path=${run_directory}/results.json
    completion_marker=${output_path}_${sampler_type}_runtime.json
    mkdir -p "${run_directory}"
    export TMPDIR=${repo}/.tmp/slurm-${SLURM_ARRAY_JOB_ID:-manual}-${array_index}
    mkdir -p "${TMPDIR}"

    model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${steps},block_size=${max_new_tokens},temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,sampler_type=${sampler_type},proposal_strategy=dependency,candidate_budget=${candidate_budget},candidate_chunk_size=${candidate_chunk_size},dependency_commit_k=${commit_k},dependency_parallel_variant=${variant},dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=42,dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=${conflict_normalization},dependency_conflict_penalty=${conflict_penalty},dependency_hard_conflict_threshold=${hard_conflict_threshold},dependency_anchor_support_weight=${anchor_support_weight},dependency_anchor_confidence_threshold=${anchor_confidence_threshold},diagnostic_metadata=true"
    wandb_arguments=()
    wandb_name=p6-${run_tag:-default}-${verifier}-${task}-${variant}-k${commit_k}-n${candidate_budget}
    if [ "${wandb_mode}" != disabled ]; then
        wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=p6_ablation,mode=${wandb_mode},dir=${WANDB_DIR}"
        if [ -n "${wandb_entity}" ]; then
            wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
        fi
        wandb_config_args="phase=6,task=${task},verifier=${verifier},parallel_variant=${variant},commit_k=${commit_k},candidate_budget=${candidate_budget},limit=${limit},max_new_tokens=${max_new_tokens},candidate_chunk_size=${candidate_chunk_size},run_tag=${run_tag:-default},slurm_logical_index=${array_index}"
        wandb_arguments+=(
            --wandb_args "${wandb_init_args}"
            --wandb_config_args "${wandb_config_args}"
        )
    fi

    task_arguments=(
        --tasks "${task}"
        --batch_size 1
        --device cuda
        --limit "${limit}"
        --seed 42
        --apply_chat_template
        --log_samples
    )
    if [ "${task}" = gsm8k_cot ]; then
        task_arguments+=(--num_fewshot 5)
    else
        task_arguments+=(--num_fewshot 0 --confirm_run_unsafe_code)
    fi

    echo "Starting Phase-6 mechanism ablation cell"
    echo "Logical matrix index: ${array_index}"
    echo "Verifier: ${verifier}"
    echo "Task: ${task}"
    echo "Parallel variant: ${variant}"
    echo "Candidate budget N: ${candidate_budget}"
    echo "Commit size k: ${commit_k}"
    echo "Steps: ${steps}"
    echo "Examples: ${limit}"
    echo "W&B mode: ${wandb_mode}"
    if [ "${wandb_mode}" != disabled ]; then
        echo "W&B project/group/run: ${wandb_project}/${wandb_group}/${wandb_name}"
    fi
    echo "Output: ${run_directory}"

    if [ "${P6_DRY_RUN:-0}" = 1 ]; then
        echo "P6_DRY_RUN=1; launch skipped"
        continue
    fi
    if [ "${P6_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
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

    echo "Phase-6 mechanism ablation cell completed: ${run_directory}"
done
