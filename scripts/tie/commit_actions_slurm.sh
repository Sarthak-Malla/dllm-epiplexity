#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh SPLIT ARM [POLICY]
#SBATCH --job-name=mdlm-commit
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
split="${1:?Specify pilot, fit, validation, fresh, support, replication or heldout}"
arm="${2:?Specify the experiment arm}"
policy="${3:-}"
case "$arm" in
    llada|dream|native_llada|native_dream|relay|post_generation|diagnostic|exact|llada_verifier|text|schedule_diagnostic|positions|joint|ctca|multi|shared_audit_midblock|action_fusion|action_fusion_wide|action_fusion_short|action_fusion_short_fresh|final_answer_fusion|final_answer_fusion_extract|peer_block_proposals|exact_text_fusion|commit_support|fused_position_replication) ;;
    *) echo "Unknown arm $arm" >&2; exit 2 ;;
esac
case "$split" in
    pilot|fit|validation|fresh|support|replication|heldout) ;;
    *) echo "Unknown split $split" >&2; exit 2 ;;
esac
task_args=(--tasks gsm8k_cot)
if [[ "$arm" == action_fusion_short_fresh || "$split" == fresh ]]; then
    if [[ "$arm" != action_fusion_short_fresh || "$split" != fresh ]]; then
        echo "Use fresh and action_fusion_short_fresh together" >&2
        exit 2
    fi
    task_args=(--tasks gsm8k_cot_train_dev --include_path "$PWD/scripts/tie/tasks")
fi
if [[ "$arm" == commit_support || "$split" == support ]]; then
    if [[ "$arm" != commit_support || "$split" != support ]]; then
        echo "Use support and commit_support together" >&2
        exit 2
    fi
    task_args=(--tasks gsm8k_cot_train_dev --include_path "$PWD/scripts/tie/tasks")
fi
if [[ "$arm" == fused_position_replication || "$split" == replication ]]; then
    if [[ "$arm" != fused_position_replication || "$split" != replication ]]; then
        echo "Use replication and fused_position_replication together" >&2
        exit 2
    fi
    task_args=(--tasks gsm8k_cot_train_dev --include_path "$PWD/scripts/tie/tasks")
fi
if [[ "$arm" == final_answer_fusion* && "$split" != fit && "$split" != validation ]]; then
    echo "Saved final-answer fusion uses the fit and validation traces" >&2
    exit 2
fi
if [[ "$arm" == action_fusion_short && "$split" != fit && "$split" != validation ]]; then
    echo "Short action fusion uses the fit and validation full-rollout traces" >&2
    exit 2
fi
if [[ ( "$arm" == peer_block_proposals || "$arm" == exact_text_fusion ) && "$split" != pilot ]]; then
    echo "$arm uses the fixed pilot split" >&2
    exit 2
fi
if [[ "$split" == heldout ]]; then
    if [[ "$arm" == diagnostic || "$arm" == schedule_diagnostic || "$arm" == shared_audit_midblock || "$arm" == action_fusion* ]]; then
        echo "Paired calibration diagnostics are development-only" >&2
        exit 2
    fi
    python -c 'import json, sys; assert "selected" in json.load(open(sys.argv[1])), "A frozen policy is required"' "$policy"
    shard="${SHARD:?Set SHARD to a value from 0 to 3}"
    [[ "$shard" =~ ^[0-3]$ ]]
    split_args=(--shard "$shard")
    output_split="heldout-$shard"
else
    split_args=()
    output_split="$split"
fi
run="/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/$output_split/$arm/${SLURM_JOB_ID:?Submit with sbatch}"
mkdir -p "$run"
echo "Experiment $split/$arm on ${SLURM_NODELIST:-unknown}, writing $run"
git rev-parse HEAD > "$run/git-commit.txt"
git status --short > "$run/git-status.txt"
python scripts/tie/commit_study.py split "$split" "${split_args[@]}" > "$run/split.json"
mode="$arm"
if [[ "$arm" == shared_audit_midblock ]]; then
    mode=shared_audit
fi
args="mode=$mode,trace_path=$run/trajectory.jsonl,doc_ids_path=$run/split.json"
if [[ "$arm" == shared_audit_midblock ]]; then
    args="$args,audit_checkpoints=12:44:76"
elif [[ "$arm" == action_fusion* ]]; then
    args="$args,audit_checkpoints=12:60:108"
elif [[ "$arm" == peer_block_proposals ]]; then
    args="$args,audit_checkpoints=17:65"
fi
if [[ "$arm" == final_answer_fusion* ]]; then
    case "$split" in
        fit) source_job=424098 ;;
        validation) source_job=424099 ;;
    esac
    source_run="/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/$split/shared_audit_midblock/$source_job"
    source_samples=("$source_run"/*/samples_gsm8k_cot_*.jsonl)
    if [[ ! -f "$source_run/trajectory.jsonl" || ${#source_samples[@]} -ne 1 || ! -f "${source_samples[0]}" ]]; then
        echo "Expected one source trace and one lm-eval samples file in $source_run" >&2
        exit 2
    fi
    args="$args,source_trace_path=$source_run/trajectory.jsonl,source_samples_path=${source_samples[0]}"
elif [[ "$arm" == action_fusion_short ]]; then
    case "$split" in
        fit) source_job=424501 ;;
        validation) source_job=424502 ;;
    esac
    source_trace="/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/$split/action_fusion_wide/$source_job/trajectory.jsonl"
    if [[ ! -f "$source_trace" ]]; then
        echo "Missing full-rollout trace $source_trace" >&2
        exit 2
    fi
    args="$args,source_trace_path=$source_trace"
fi
args="$args,max_new_tokens=256,steps=128,block_size=16,answer_tokens=16,relay_interval=16"
llada="/home/juan.guevara/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
dream="/home/juan.guevara/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"
args="$args,llada=$llada,dream=$dream"
args="$args,llada_device=${LLADA_DEVICE:-cuda:0},dream_device=${DREAM_DEVICE:-cuda:0}"
progress_interval=10
if [[ "$split" == pilot || "$split" == support ]]; then
    progress_interval=1
fi
args="$args,progress_log_interval=$progress_interval"
if [[ "$arm" == native_llada || "$arm" == native_dream ]]; then
    native="${arm#native_}"
    checkpoint="$llada"
    native_settings="block_size=16,remasking=low_confidence"
    if [[ "$native" == dream ]]; then
        checkpoint="$dream"
        native_settings="alg=maskgit_plus"
    fi
    exec python -u -m "dllm.pipelines.$native.eval" --model "$native" \
        --tasks gsm8k_cot --samples "$run/split.json" --batch_size 1 --num_fewshot 0 \
        --apply_chat_template --log_samples --output_path "$run" \
        --model_args "pretrained=$checkpoint,max_new_tokens=256,steps=128,temperature=0.0,dtype=bfloat16,$native_settings"
fi
if [[ -n "$policy" ]]; then
    cp "$policy" "$run/policy.json"
    args="$args,policy_path=$run/policy.json"
fi
if [[ "$arm" == multi ]]; then
    args="$args,multi_model=${MULTI_MODEL:-llada},sample_count=${SAMPLE_COUNT:-2}"
fi
exec python -u -c 'import runpy; print("Python started. Importing dLLM.", flush=True); runpy.run_module("dllm.pipelines.tie.eval", run_name="__main__", alter_sys=True)' \
    --model tie "${task_args[@]}" \
    --samples "$run/split.json" --batch_size 1 --num_fewshot 0 --log_samples \
    --output_path "$run" --model_args "$args"
