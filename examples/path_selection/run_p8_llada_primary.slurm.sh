#!/bin/bash
#SBATCH --job-name=path-p8-llada
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-3%4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

# Run the frozen one-example smoke first:
#   sbatch --export=ALL,P8_STAGE=smoke,P8_WANDB_MODE=online \
#     /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_p8_llada_primary.slurm.sh
#
# After the smoke is accepted, run one primary shard (all seven methods):
#   sbatch --export=ALL,P8_STAGE=primary,P8_TASK=gsm8k_cot,P8_SHARD_INDEX=0,P8_WANDB_MODE=online \
#     /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_p8_llada_primary.slurm.sh

# Conda CUDA hooks reference optional unset variables, so do not enable -u.
set -eo pipefail

repo=/home/sarthak.malla/dllm-selection-ensemble
cache=/home/sarthak.malla/.cache/huggingface/hub
model=models--GSAI-ML--LLaDA-8B-Instruct
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
checkpoint=${cache}/${model}/snapshots/${revision}
plan=${repo}/examples/path_selection/dependency_guided/p8_evaluation_plan.json
manifest_tool=${repo}/examples/path_selection/dependency_guided/prepare_p8_manifest.py
stage=${P8_STAGE:-smoke}
seed=${P8_SEED:-42}
candidate_chunk_size=${P8_CANDIDATE_CHUNK_SIZE:-1}
run_tag=${P8_RUN_TAG:-frozen_v1}
wandb_mode=${P8_WANDB_MODE:-online}
wandb_project=${P8_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${P8_WANDB_ENTITY:-}
output_root=${repo}/eval_results/path_selection/dependency_guided/p8/${run_tag}/${stage}

tasks=(
    gsm8k_cot
    hendrycks_math500
    humaneval_instruct_llada
    mbpp_instruct_llada
)
methods=(
    native_confidence
    current_mixed_n4
    confidence_gumbel_n8
    dependency_fixed_k1_n4
    dependency_fixed_k2_n4
    dependency_fixed_k4_n4
    dependency_entropy_budget_n4
)
method_count=${#methods[@]}

if [ "${stage}" = smoke ]; then
    logical_cell_count=$((${#tasks[@]} * method_count))
    shard_index=0
    diagnostic_retention=full
elif [ "${stage}" = primary ]; then
    task=${P8_TASK:-}
    shard_index=${P8_SHARD_INDEX:-}
    if [ -z "${task}" ]; then
        echo "P8_TASK is required for P8_STAGE=primary" >&2
        exit 2
    fi
    if [[ ! " ${tasks[*]} " =~ " ${task} " ]]; then
        echo "Unsupported P8_TASK: ${task}" >&2
        exit 2
    fi
    if ! [[ "${shard_index}" =~ ^[0-9]+$ ]]; then
        echo "P8_SHARD_INDEX must be a nonnegative integer" >&2
        exit 2
    fi
    logical_cell_count=${method_count}
    diagnostic_retention=compact
else
    echo "P8_STAGE must be smoke or primary, got ${stage}" >&2
    exit 2
fi

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "P8_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "P8_SEED must be a nonnegative integer, got ${seed}" >&2
    exit 2
fi
if [ "${seed}" -ne 42 ]; then
    echo "The frozen primary configuration uses P8_SEED=42, got ${seed}" >&2
    exit 2
fi
if ! [[ "${candidate_chunk_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "P8_CANDIDATE_CHUNK_SIZE must be a positive integer" >&2
    exit 2
fi
if [ "${candidate_chunk_size}" -ne 1 ]; then
    echo "The frozen correctness reference requires candidate chunk size 1" >&2
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
if [ ! -f "${plan}" ] || [ ! -f "${manifest_tool}" ]; then
    echo "The frozen Phase-8 plan or manifest tool is missing" >&2
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
export HF_HOME=/home/sarthak.malla/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR=/scratch/sarthak.malla/tmp/matplotlib
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=${repo}/.wandb

worker_count=${P8_WORKER_COUNT:-4}
if ! [[ "${worker_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "P8_WORKER_COUNT must be a positive integer" >&2
    exit 2
fi
worker_index=${SLURM_ARRAY_TASK_ID:-0}
if [ "${worker_index}" -ge "${worker_count}" ]; then
    echo "Array task ${worker_index} is outside P8_WORKER_COUNT=${worker_count}" >&2
    exit 2
fi
logical_indices=()
for ((logical_index=worker_index; logical_index<logical_cell_count; logical_index+=worker_count)); do
    logical_indices+=("${logical_index}")
done

if [ "${P8_DRY_RUN:-0}" != 1 ]; then
    srun --ntasks=1 nvidia-smi
fi

for logical_index in "${logical_indices[@]}"; do
    if [ "${stage}" = smoke ]; then
        task_index=$((logical_index / method_count))
        method_index=$((logical_index % method_count))
        task=${tasks[${task_index}]}
    else
        method_index=${logical_index}
    fi
    method=${methods[${method_index}]}

    case "${task}" in
        gsm8k_cot)
            max_new_tokens=512
            num_fewshot=5
            suppress_tokens='[]'
            begin_suppress_tokens='[126081;126348]'
            unsafe_code=0
            ;;
        hendrycks_math500)
            max_new_tokens=512
            num_fewshot=0
            suppress_tokens='[]'
            begin_suppress_tokens='[126081;126348]'
            unsafe_code=0
            ;;
        humaneval_instruct_llada)
            max_new_tokens=512
            num_fewshot=0
            suppress_tokens='[126081]'
            begin_suppress_tokens='[]'
            unsafe_code=1
            ;;
        mbpp_instruct_llada)
            max_new_tokens=256
            num_fewshot=0
            suppress_tokens='[]'
            begin_suppress_tokens='[126081;126348]'
            unsafe_code=1
            ;;
        *)
            echo "Unsupported task: ${task}" >&2
            exit 2
            ;;
    esac

    sampler_type=entropy_drop
    proposal_strategy=dependency
    candidate_budget=4
    commit_k=1
    cardinality_strategy=fixed
    entropy_budget=2.0
    size_scoring=raw
    steps=${max_new_tokens}

    case "${method}" in
        native_confidence)
            sampler_type=greedy
            proposal_strategy=native
            candidate_budget=1
            ;;
        current_mixed_n4)
            proposal_strategy=baseline_current_mixed
            ;;
        confidence_gumbel_n8)
            proposal_strategy=baseline_confidence_gumbel
            candidate_budget=8
            ;;
        dependency_fixed_k1_n4)
            ;;
        dependency_fixed_k2_n4)
            commit_k=2
            steps=$((max_new_tokens / 2))
            ;;
        dependency_fixed_k4_n4)
            commit_k=4
            steps=$((max_new_tokens / 4))
            ;;
        dependency_entropy_budget_n4)
            cardinality_strategy=entropy_budget
            size_scoring=per_token
            ;;
        *)
            echo "Unsupported Phase-8 method: ${method}" >&2
            exit 2
            ;;
    esac

    run_directory=${output_root}/${task}/${method}/seed${seed}/shard$(printf '%04d' "${shard_index}")
    output_path=${run_directory}/results.json
    completion_marker=${output_path}_${sampler_type}_runtime.json
    mkdir -p "${run_directory}"
    export TMPDIR=${repo}/.tmp/p8-${SLURM_ARRAY_JOB_ID:-manual}-${worker_index}-${logical_index}
    mkdir -p "${TMPDIR}"
    sample_manifest=${TMPDIR}/samples.json
    python "${manifest_tool}" \
        --plan "${plan}" \
        --stage "${stage}" \
        --task "${task}" \
        --shard-index "${shard_index}" \
        --output "${sample_manifest}"
    cp "${sample_manifest}" "${run_directory}/samples.json"

    common_model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${steps},block_size=64,temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,suppress_tokens=${suppress_tokens},begin_suppress_tokens=${begin_suppress_tokens},diagnostic_retention=${diagnostic_retention}"
    if [ "${method}" = native_confidence ]; then
        model_args="${common_model_args},sampler_type=greedy,remasking=low_confidence"
    else
        model_args="${common_model_args},sampler_type=entropy_drop,proposal_strategy=${proposal_strategy},candidate_budget=${candidate_budget},candidate_chunk_size=${candidate_chunk_size},dependency_commit_k=${commit_k},dependency_parallel_variant=soft_full,dependency_cardinality_strategy=${cardinality_strategy},dependency_max_action_size=4,dependency_action_sizes=1|2|4,dependency_utility_threshold=0.0,dependency_entropy_budget=${entropy_budget},dependency_size_scoring=${size_scoring},dependency_immediate_cost_weight=1.0,dependency_size_penalty=0.0,dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=${seed},dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=max,dependency_conflict_penalty=1.0,dependency_hard_conflict_threshold=0.25,dependency_anchor_support_weight=1.0,dependency_anchor_confidence_threshold=0.8,diagnostic_metadata=true"
    fi

    wandb_arguments=()
    wandb_group=p8-${run_tag}-${stage}-${task}
    wandb_name=p8-${run_tag}-${stage}-${task}-${method}-s${seed}-shard$(printf '%04d' "${shard_index}")
    if [ "${wandb_mode}" != disabled ]; then
        wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=p8_${stage},mode=${wandb_mode},dir=${WANDB_DIR}"
        if [ -n "${wandb_entity}" ]; then
            wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
        fi
        wandb_config_args="phase=8,freeze=p8_llada_frozen_v1,stage=${stage},task=${task},method=${method},proposal_strategy=${proposal_strategy},candidate_budget=${candidate_budget},commit_k=${commit_k},cardinality_strategy=${cardinality_strategy},entropy_budget=${entropy_budget},seed=${seed},shard_index=${shard_index},max_new_tokens=${max_new_tokens},block_size=64,candidate_chunk_size=${candidate_chunk_size},diagnostic_retention=${diagnostic_retention},run_tag=${run_tag}"
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
        --seed "${seed}"
        --apply_chat_template
        --num_fewshot "${num_fewshot}"
        --log_samples
    )
    if [ "${unsafe_code}" = 1 ]; then
        task_arguments+=(--confirm_run_unsafe_code)
    fi

    echo "Starting frozen Phase-8 LLaDA cell"
    echo "Logical index: ${logical_index}/${logical_cell_count}"
    echo "Stage/task/method: ${stage}/${task}/${method}"
    echo "Shard/seed: ${shard_index}/${seed}"
    echo "Generation length/steps/block: ${max_new_tokens}/${steps}/64"
    echo "Proposal/N/k/cardinality: ${proposal_strategy}/${candidate_budget}/${commit_k}/${cardinality_strategy}"
    echo "Diagnostic retention: ${diagnostic_retention}"
    echo "W&B mode/group/run: ${wandb_mode}/${wandb_group}/${wandb_name}"
    echo "Output: ${run_directory}"

    if [ "${P8_DRY_RUN:-0}" = 1 ]; then
        echo "P8_DRY_RUN=1; launch skipped"
        continue
    fi
    if [ "${P8_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
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

    echo "Frozen Phase-8 LLaDA cell completed: ${run_directory}"
done
