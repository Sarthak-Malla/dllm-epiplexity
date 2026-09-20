#!/usr/bin/env bash
#SBATCH --job-name=llada-eval
#SBATCH --output=/home/sarthak.malla/dllm-learning-decoding-path/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-learning-decoding-path/.logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

set -eo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh [--dry-run] [--limit N] MODE' \
        'Modes: greedy, min_entropy, max_top2_prob, candidate_expansion, majority_voting' \
        'Shared settings and environment overrides: /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/config.sh' \
        '--dry-run prints the resolved command without environment setup or filesystem writes.' \
        '--limit overrides LIMIT with a positive sample count or a fraction between 0 and 1.'
}

dry_run=false
sampler_type=''
limit_override=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true ;;
        --limit)
            if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
                printf -- '--limit requires a positive sample count or a fraction between 0 and 1.\n' >&2
                exit 2
            fi
            limit_override="$2"
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        greedy|min_entropy|max_top2_prob|candidate_expansion|majority_voting)
            if [[ -n "$sampler_type" ]]; then
                printf 'Specify exactly one sampler mode.\n' >&2
                exit 2
            fi
            sampler_type="$1"
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if [[ -z "$sampler_type" ]]; then
    usage >&2
    exit 2
fi

source /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/config.sh
if [[ -n "$limit_override" ]]; then
    LIMIT="$limit_override"
fi
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ && ! "$LIMIT" =~ ^0?\.[0-9]*[1-9][0-9]*$ ]]; then
    printf 'LIMIT must be a positive integer count or a fraction between 0 and 1; received: %s\n' "$LIMIT" >&2
    exit 2
fi

if [[ ! "$NUM_GPU" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NUM_GPU must be a positive integer; received: %s\n' "$NUM_GPU" >&2
    exit 2
fi
if [[ -n "$RUN_SUFFIX" && ! "$RUN_SUFFIX" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    printf 'RUN_SUFFIX may contain only letters, numbers, underscores, dots, and hyphens.\n' >&2
    exit 2
fi
for boolean_setting in APPLY_CHAT_TEMPLATE CONFIRM_RUN_UNSAFE_CODE WANDB_ENABLED; do
    case "${!boolean_setting}" in
        true|false) ;;
        *) printf '%s must be true or false.\n' "$boolean_setting" >&2; exit 2 ;;
    esac
done
if [[ ! "$WANDB_LOG_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    printf 'WANDB_LOG_EVERY must be a positive integer; received: %s\n' "$WANDB_LOG_EVERY" >&2
    exit 2
fi

model_args="pretrained=${MODEL_NAME_OR_PATH},max_new_tokens=${MAX_NEW_TOKENS},block_size=${BLOCK_SIZE},cfg_scale=${CFG_SCALE},temperature=${TEMPERATURE},suppress_tokens=${SUPPRESS_TOKENS},begin_suppress_tokens=${BEGIN_SUPPRESS_TOKENS},sampler_type=${sampler_type}"
case "$sampler_type" in
    greedy|min_entropy|max_top2_prob)
        model_args+=",steps=${STEPS}"
        ;;
    candidate_expansion|majority_voting)
        model_args+=",candidate_fraction=${CANDIDATE_FRACTION},strategies=${STRATEGIES}"
        ;;
esac

command=(
    srun
    --nodes=1
    --ntasks=1
    --gres="gpu:${NUM_GPU}"
    accelerate launch
    --num_processes "$NUM_GPU"
    --num_machines 1
    --mixed_precision no
    --dynamo_backend no
)
if (( NUM_GPU > 1 )); then
    command+=(--multi_gpu --main_process_port "$MAIN_PROCESS_PORT")
fi
command+=(
    "${PROJECT_ROOT}/ensemble/pipelines/llada/eval.py"
    --model llada_ensemble
    --tasks "$TASKS"
    --num_fewshot "$NUM_FEWSHOT"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --log_samples
    --model_args "$model_args"
)
if [[ "$APPLY_CHAT_TEMPLATE" == true ]]; then
    command+=(--apply_chat_template)
fi
if [[ -n "$LIMIT" ]]; then
    command+=(--limit "$LIMIT")
fi
if [[ "$CONFIRM_RUN_UNSAFE_CODE" == true ]]; then
    command+=(--confirm_run_unsafe_code)
fi

# Hash evaluation arguments, including sampling settings and seeds.
# Logging settings do not affect predictions or their cache namespace.
# Identical configurations can resume their cache; changed configurations cannot collide.
run_digest=$(printf '%s\0' "${command[@]}" | sha256sum)
run_digest="${run_digest:0:16}"
task_label="${TASKS//[^a-zA-Z0-9_.-]/_}"
run_name="${task_label}_${sampler_type}_${run_digest}"
if [[ -n "$RUN_SUFFIX" ]]; then
    run_name+="_${RUN_SUFFIX}"
fi
output_path="${OUTPUT_ROOT}/${run_name}"
command+=(--output_path "$output_path" --use_cache "${output_path}/responses.cache")
if [[ "$WANDB_ENABLED" == true ]]; then
    wandb_args="project=${WANDB_PROJECT},name=${run_name},job_type=${sampler_type}"
    if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_args+=",entity=${WANDB_ENTITY}"
    fi
    command+=(--wandb_args "$wandb_args")
fi

gpu_info_command=(srun --nodes=1 --ntasks=1 --gres="gpu:${NUM_GPU}" nvidia-smi)

printf 'Sampler: %s\nModel: %s\nProcesses: %s\nOutput: %s\n' \
    "$sampler_type" "$MODEL_NAME_OR_PATH" "$NUM_GPU" "$output_path"
printf 'W&B: %s; decoding log interval: %s steps\n' "$WANDB_ENABLED" "$WANDB_LOG_EVERY"
printf 'GPU information:'
printf ' %q' "${gpu_info_command[@]}"
printf '\n'
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$dry_run" == true ]]; then
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" ]]; then
    printf 'Evaluation requires an allocated Slurm compute node. Submit this script with sbatch, use scripts/ensemble/slurm.sh, or use --dry-run to preview.\n' >&2
    exit 2
fi

cd "$PROJECT_ROOT"
if [[ -f "$HOME/.zshrc" ]]; then
    source "$HOME/.zshrc"
fi
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

mkdir -p "${PROJECT_ROOT}/.logs" "$output_path"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp/slurm-${SLURM_JOB_ID:-manual}/${run_name}}"
mkdir -p "$TMPDIR"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export WANDB_LOG_EVERY
export HF_HOME="${HF_HOME:-/home/sarthak.malla/.cache/huggingface}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-warn}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

"${gpu_info_command[@]}"
"${command[@]}"
printf 'Evaluation completed: %s\n' "$output_path"
