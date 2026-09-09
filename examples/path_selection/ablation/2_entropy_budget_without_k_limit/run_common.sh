#!/bin/bash

# Shared implementation for Ablations 2--6. Submit a .slurm.sh entrypoint.

set -eo pipefail

repo=/home/sarthak.malla/dllm-selection-ensemble
cache=/home/sarthak.malla/.cache/huggingface/hub
model=models--GSAI-ML--LLaDA-8B-Instruct
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
checkpoint=${cache}/${model}/snapshots/${revision}
task=gsm8k_cot
seed=42
ablation_number=${PATH_ABLATION_NUMBER:-2}
output_subdirectory=${PATH_ABLATION_OUTPUT_SUBDIRECTORY:-2_entropy_budget_without_k_limit}
candidate_budget=${PATH_ABLATION_CANDIDATE_BUDGET:-4}
cardinality_strategy=${PATH_ABLATION_CARDINALITY_STRATEGY:-entropy_budget}
utility_threshold=${PATH_ABLATION_UTILITY_THRESHOLD:-0.0}
token_temperature=${PATH_ABLATION_TOKEN_TEMPERATURE:-0.0}
confidence_exponent=${PATH_ABLATION_CONFIDENCE_EXPONENT:-0.0}
evaluation_limit=${PATH_ABLATION_LIMIT:-}
entropy_budget=${ABLATION2_ENTROPY_BUDGET:-}
block_size=64
maximum_action_size=${ABLATION2_MAXIMUM_ACTION_SIZE:-${block_size}}
num_gpu=2
run_tag=${ABLATION2_RUN_TAG:-full_v1}
wandb_mode=${ABLATION2_WANDB_MODE:-online}
wandb_project=${ABLATION2_WANDB_PROJECT:-dllm-selection-ensemble}
wandb_entity=${ABLATION2_WANDB_ENTITY:-}
wandb_group=${PATH_ABLATION_WANDB_GROUP:-ablation-2-entropy-budget-cap-comparison}
wandb_name_prefix=${PATH_ABLATION_WANDB_NAME_PREFIX:-ablation-2}

case "${confidence_exponent}" in
    0|0.0) confidence_exponent=0.0 ;;
    1|1.0) confidence_exponent=1.0 ;;
    *)
        echo "PATH_ABLATION_CONFIDENCE_EXPONENT must be 0.0 or 1.0" >&2
        exit 2
        ;;
esac
confidence_path=
confidence_name=
if [ -n "${PATH_ABLATION_CONFIDENCE_EXPONENT:-}" ]; then
    confidence_path=/confidence_exponent${confidence_exponent}
    confidence_name=-confidence${confidence_exponent}
fi

evaluation_arguments=()
limit_path=
limit_name=
if [ -n "${evaluation_limit}" ]; then
    if [[ ! "${evaluation_limit}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PATH_ABLATION_LIMIT must be a positive integer" >&2
        exit 2
    fi
    evaluation_arguments+=(--limit "${evaluation_limit}")
    limit_path=/limit${evaluation_limit}
    limit_name=-limit${evaluation_limit}
fi

case "${token_temperature}" in
    0|0.0) token_temperature=0.0 ;;
    0.5) ;;
    1|1.0) token_temperature=1.0 ;;
    *)
        echo "PATH_ABLATION_TOKEN_TEMPERATURE must be 0.0, 0.5, or 1.0" >&2
        exit 2
        ;;
esac
# Keep the existing temperature-zero result and cache locations unchanged.
temperature_path=
temperature_name=
if [ "${token_temperature}" != 0.0 ]; then
    temperature_path=/temperature${token_temperature}
    temperature_name=-temperature${token_temperature}
fi

if [[ ! "${ablation_number}" =~ ^[0-9]+$ ]]; then
    echo "PATH_ABLATION_NUMBER must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! "${output_subdirectory}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "PATH_ABLATION_OUTPUT_SUBDIRECTORY must be one directory name" >&2
    exit 2
fi
if [[ ! "${candidate_budget}" =~ ^[0-9]+$ ]] \
    || [ "${candidate_budget}" -lt 1 ] \
    || [ "${candidate_budget}" -gt 64 ]; then
    echo "PATH_ABLATION_CANDIDATE_BUDGET must be an integer from 1 to 64" >&2
    exit 2
fi

case "${cardinality_strategy}" in
    entropy_budget)
        case "${entropy_budget}" in
            1.0|2.0|4.0) ;;
            *)
                echo "ABLATION2_ENTROPY_BUDGET must be 1.0, 2.0, or 4.0" >&2
                exit 2
                ;;
        esac
        stopping_args="dependency_entropy_budget=${entropy_budget}"
        stopping_config="entropy_budget=${entropy_budget}"
        stopping_label=entropy_budget${entropy_budget}
        stopping_name=entropy-budget${entropy_budget}
        ;;
    marginal_utility)
        case "${utility_threshold}" in
            0|0.0) utility_threshold=0.0 ;;
            0.25|0.5) ;;
            1|1.0) utility_threshold=1.0 ;;
            2|2.0) utility_threshold=2.0 ;;
            4|4.0) utility_threshold=4.0 ;;
            *)
                echo "PATH_ABLATION_UTILITY_THRESHOLD must be 0, 0.25, 0.5, 1, 2, or 4" >&2
                exit 2
                ;;
        esac
        stopping_args="dependency_utility_threshold=${utility_threshold}"
        stopping_config="utility_threshold=${utility_threshold}"
        stopping_label=marginal_utility_tau${utility_threshold}
        stopping_name=marginal-utility-tau${utility_threshold}
        ;;
    *)
        echo "PATH_ABLATION_CARDINALITY_STRATEGY must be entropy_budget or marginal_utility" >&2
        exit 2
        ;;
esac

if [[ ! "${maximum_action_size}" =~ ^[0-9]+$ ]] \
    || [ "${maximum_action_size}" -lt 1 ] \
    || [ "${maximum_action_size}" -gt "${block_size}" ]; then
    echo "ABLATION2_MAXIMUM_ACTION_SIZE must be an integer from 1 to ${block_size}" >&2
    exit 2
fi

if [ "${maximum_action_size}" -eq "${block_size}" ]; then
    action_cap_label=uncapped
else
    action_cap_label=max${maximum_action_size}
fi

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ABLATION2_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if [ "${wandb_mode}" != disabled ] && [ "${wandb_mode}" != offline ] && [ "${wandb_mode}" != online ]; then
    echo "ABLATION2_WANDB_MODE must be disabled, offline, or online" >&2
    exit 2
fi
export WANDB_DIR=${repo}/.wandb

output_root=${repo}/eval_results/path_selection/ablation/${output_subdirectory}
if [ "${action_cap_label}" = uncapped ]; then
    # Preserve the original uncapped output layout for backwards compatibility.
    run_directory=${output_root}/${run_tag}/${task}${confidence_path}/${stopping_label}${temperature_path}${limit_path}/seed${seed}
else
    run_directory=${output_root}/${run_tag}/${task}${confidence_path}/${action_cap_label}/${stopping_label}${temperature_path}${limit_path}/seed${seed}
fi
output_path=${run_directory}/results.json
completion_marker=${output_path}_entropy_drop_runtime.json
response_cache=${run_directory}/responses.cache

# With maximum_action_size equal to the complete decoding block, there is no
# separate k cap. Smaller values provide a hard safety cap in addition to the
# configured stopping rule.
model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=256,steps=64,block_size=${block_size},temperature=${token_temperature},cfg_scale=0.0,stochastic_transfer=false,return_dict=true,diagnostic_retention=compact,sampler_type=entropy_drop,proposal_strategy=dependency,candidate_budget=${candidate_budget},candidate_chunk_size=1,dependency_commit_k=1,dependency_parallel_variant=soft_full,dependency_cardinality_strategy=${cardinality_strategy},dependency_max_action_size=${maximum_action_size},${stopping_args},dependency_size_scoring=per_token,dependency_immediate_cost_weight=1.0,dependency_size_penalty=0.0,dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=${confidence_exponent},dependency_generation_seed=${seed},dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,dependency_conflict_normalization=max,dependency_conflict_penalty=1.0,dependency_hard_conflict_threshold=0.25,dependency_anchor_support_weight=1.0,dependency_anchor_confidence_threshold=0.8,diagnostic_metadata=true"

wandb_arguments=()
wandb_name=${wandb_name_prefix}-${task}-${action_cap_label}${confidence_name}-${stopping_name}${temperature_name}${limit_name}-s${seed}
if [ "${wandb_mode}" != disabled ]; then
    wandb_init_args="project=${wandb_project},name=${wandb_name},group=${wandb_group},job_type=ablation,mode=${wandb_mode},dir=${WANDB_DIR}"
    if [ -n "${wandb_entity}" ]; then
        wandb_init_args="${wandb_init_args},entity=${wandb_entity}"
    fi
    wandb_config_args="ablation=${ablation_number},task=${task},selector=entropy_drop,lookahead=true,proposal_strategy=dependency,candidate_budget=${candidate_budget},cardinality_strategy=${cardinality_strategy},${stopping_config},maximum_action_size=${maximum_action_size},size_scoring=per_token,seed=${seed},max_new_tokens=256,steps=64,block_size=${block_size},gpu_count=${num_gpu},run_tag=${run_tag}"
    wandb_config_args+=",temperature=${token_temperature},dependency_position_temperature=1.0"
    wandb_config_args+=",dependency_confidence_exponent=${confidence_exponent},evaluation_limit=${evaluation_limit:-full}"
    wandb_arguments+=(
        --wandb_args "${wandb_init_args}"
        --wandb_config_args "${wandb_config_args}"
    )
fi

echo "Starting ${cardinality_strategy}/entropy-drop action-cap ablation"
echo "Ablation: ${ablation_number}"
echo "Task: ${task}"
echo "Candidate count: ${candidate_budget}"
echo "Stopping configuration: ${stopping_config}"
echo "Token generation temperature: ${token_temperature}"
echo "Confidence exponent: ${confidence_exponent}"
echo "Evaluation limit: ${evaluation_limit:-full}"
echo "Maximum action size/block size: ${maximum_action_size}/${block_size}"
echo "Output: ${run_directory}"

if [ "${ABLATION2_DRY_RUN:-0}" = 1 ]; then
    echo "Model arguments: ${model_args}"
    echo "Output path: ${output_path}"
    echo "Response cache: ${response_cache}"
    echo "W&B name: ${wandb_name}"
    echo "W&B configuration: ${wandb_config_args:-disabled}"
    echo "ABLATION2_DRY_RUN=1; launch skipped"
    exit 0
fi

if [ ! -d "${checkpoint}" ]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 1
fi

cd "${repo}"
mkdir -p "${repo}/.logs" "${repo}/.wandb" "${run_directory}"
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
export TMPDIR=/tmp/dep-uncap-${SLURM_JOB_ID:-manual}
mkdir -p "${TMPDIR}"

if [ "${ABLATION2_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
    echo "Completion marker exists; skipping: ${completion_marker}"
    exit 0
fi

main_process_port=$((26000 + ${SLURM_JOB_ID:-0} % 14000))
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
        "${evaluation_arguments[@]}" \
        "${wandb_arguments[@]}" \
        --output_path "${output_path}" \
        --use_cache "${response_cache}"

echo "Ablation completed: ${task}/${action_cap_label}/${stopping_label}"
