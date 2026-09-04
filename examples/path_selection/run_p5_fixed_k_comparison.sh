#!/bin/bash
#SBATCH --job-name=path-p5-fixed-k
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%A_%a.err
#SBATCH --array=0-25%4
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
verifier=${P5_VERIFIER:-entropy_drop}
limit=${P5_LIMIT:-8}
max_new_tokens=${P5_MAX_NEW_TOKENS:-64}
candidate_chunk_size=${P5_CANDIDATE_CHUNK_SIZE:-1}
output_root=${repo}/eval_results/path_selection/dependency_guided/p5_6
run_tag=${P5_RUN_TAG:-}
if [ -n "${run_tag}" ]; then
    if [[ ! "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "P5_RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2
        exit 2
    fi
    output_root=${output_root}/${run_tag}
fi

tasks=(gsm8k_cot humaneval_instruct)
strategies=(baseline_random baseline_current_mixed baseline_confidence_gumbel dependency)
budgets=(2 4 8)

if [ "${verifier}" != entropy_drop ] && [ "${verifier}" != risk_reduction ]; then
    echo "P5_VERIFIER must be entropy_drop or risk_reduction, got ${verifier}" >&2
    exit 2
fi
if [ ! -d "${checkpoint}" ]; then
    echo "Checkpoint directory not found: ${checkpoint}" >&2
    exit 1
fi

cd "${repo}"
mkdir -p "${repo}/.logs"
mkdir -p "${repo}/.tmp"
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

logical_indices=()
worker_count=${P5_WORKER_COUNT:-}
if [ -n "${worker_count}" ]; then
    if ! [[ "${worker_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "P5_WORKER_COUNT must be a positive integer, got ${worker_count}" >&2
        exit 2
    fi
    worker_index=${SLURM_ARRAY_TASK_ID:-0}
    if [ "${worker_index}" -ge "${worker_count}" ]; then
        echo "Array task ${worker_index} is outside P5_WORKER_COUNT=${worker_count}" >&2
        exit 2
    fi
    for ((logical_index=worker_index; logical_index<26; logical_index+=worker_count)); do
        logical_indices+=("${logical_index}")
    done
else
    logical_indices+=("${SLURM_ARRAY_TASK_ID:-0}")
fi

if [ "${P5_DRY_RUN:-0}" != 1 ]; then
    srun --ntasks=1 nvidia-smi
fi

for array_index in "${logical_indices[@]}"; do
    if [ "${array_index}" -lt 0 ] || [ "${array_index}" -gt 25 ]; then
        echo "Logical matrix index must be between 0 and 25, got ${array_index}" >&2
        exit 2
    fi
    if [ "${array_index}" -lt 24 ]; then
        task_index=$((array_index / 12))
        combination_index=$((array_index % 12))
        strategy_index=$((combination_index / 3))
        budget_index=$((combination_index % 3))
        task=${tasks[${task_index}]}
        strategy=${strategies[${strategy_index}]}
        candidate_budget=${budgets[${budget_index}]}
        sampler_type=${verifier}
    else
        task_index=$((array_index - 24))
        task=${tasks[${task_index}]}
        strategy=native_confidence
        candidate_budget=1
        sampler_type=greedy
    fi

    run_directory=${output_root}/${verifier}/${task}/${strategy}/n${candidate_budget}
    output_path=${run_directory}/results.json
    completion_marker=${output_path}_${sampler_type}_runtime.json
    mkdir -p "${run_directory}"
    export TMPDIR=${repo}/.tmp/slurm-${SLURM_ARRAY_JOB_ID:-manual}-${array_index}
    mkdir -p "${TMPDIR}"

    if [ "${strategy}" = native_confidence ]; then
        model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${max_new_tokens},temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,sampler_type=greedy,remasking=low_confidence"
    else
        model_args="pretrained=${checkpoint},dtype=bfloat16,load_in_4bit=false,max_length=4096,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${max_new_tokens},temperature=0.0,cfg_scale=0.0,stochastic_transfer=false,return_dict=true,sampler_type=${sampler_type},proposal_strategy=${strategy},candidate_budget=${candidate_budget},candidate_chunk_size=${candidate_chunk_size},dependency_last_n_layers=4,dependency_direction=outgoing,dependency_target_weighting=entropy,dependency_position_temperature=1.0,dependency_confidence_exponent=0.0,dependency_generation_seed=42,dependency_sink_filter_enabled=true,dependency_sink_quantile=0.99,dependency_zero_diagonal=true,dependency_renormalize_selected_keys=true,dependency_fallback_strategy=dependency_only,diagnostic_metadata=true"
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

    echo "Starting Phase-5 fixed-k comparison cell"
    echo "Logical matrix index: ${array_index}"
    echo "Verifier: ${verifier}"
    echo "Task: ${task}"
    echo "Proposal: ${strategy}"
    echo "Candidate budget N: ${candidate_budget}"
    echo "Commit size k: 1"
    echo "Examples: ${limit}"
    echo "Output: ${run_directory}"

    if [ "${P5_DRY_RUN:-0}" = 1 ]; then
        echo "P5_DRY_RUN=1; launch skipped"
        continue
    fi
    if [ "${P5_SKIP_COMPLETED:-1}" = 1 ] && [ -s "${completion_marker}" ]; then
        echo "Completion marker exists; skipping: ${completion_marker}"
        continue
    fi

    srun --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
        python "${repo}/examples/path_selection/eval.py" \
            --model llada_path_selection \
            --model_args "${model_args}" \
            "${task_arguments[@]}" \
            --output_path "${output_path}"

    echo "Phase-5 comparison cell completed: ${run_directory}"
done
