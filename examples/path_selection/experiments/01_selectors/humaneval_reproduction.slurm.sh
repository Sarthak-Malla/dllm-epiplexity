#!/bin/bash
#SBATCH --job-name=tf-humaneval-reproduction
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/home/sarthak.malla/dllm-selection-ensemble

# Submit manually with sbatch and this file's absolute path.
# Native sampler check for examples/llada/eval.sh's HumanEval reproduction settings.
set -eo pipefail
export TF_RUN_TAG=${TF_RUN_TAG:-humaneval_llada_native_reproduction}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/run_common.sh
export TMPDIR=/tmp/tf-reproduction-${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$TF_OUTPUT_ROOT"
export HF_ALLOW_CODE_EVAL=1

# W&B records aggregate results; omit sample uploads and diagnostic telemetry.
srun --ntasks=1 --kill-on-bad-exit=1 \
    python /home/sarthak.malla/dllm-selection-ensemble/dllm/pipelines/llada/eval.py \
    --model llada --device cuda:0 --batch_size 1 --apply_chat_template \
    --tasks humaneval_instruct_llada --num_fewshot 0 --confirm_run_unsafe_code \
    --model_args 'pretrained=GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[126081],begin_suppress_tokens=[]' \
    --output_path "$TF_OUTPUT_ROOT" \
    --wandb_args "project=${WANDB_PROJECT},group=${WANDB_RUN_GROUP},name=native-llada-humaneval,job_type=reproduction"
