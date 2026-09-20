#!/usr/bin/env bash
# Shared defaults, sourced by /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh.
# Override settings through environment variables before invoking the launcher.

PROJECT_ROOT="${PROJECT_ROOT:-/home/sarthak.malla/dllm-learning-decoding-path}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
CFG_SCALE="${CFG_SCALE:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SUPPRESS_TOKENS="${SUPPRESS_TOKENS:-[]}"
BEGIN_SUPPRESS_TOKENS="${BEGIN_SUPPRESS_TOKENS:-[]}"

# Used only by the scheduled, single-strategy baselines.
STEPS="${STEPS:-64}"
# Used only by scheduler-free ensembles. List items use the harness's semicolon syntax.
STRATEGIES="${STRATEGIES:-[low_confidence;min_entropy;max_top2_prob]}"
# Proportion of currently remaining masks proposed at each decoding step.
CANDIDATE_FRACTION="${CANDIDATE_FRACTION:-0.10}"

TASKS="${TASKS:-gsm8k_cot}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
# lm-eval seeds: Python, NumPy, PyTorch, and few-shot example selection.
SEED="${SEED:-0,1234,1234,1234}"
LIMIT="${LIMIT:-}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-false}"

# W&B shares one rank-zero run for final evaluation results and decoding metrics.
# Set WANDB_MODE=offline externally to save locally without uploading.
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-dllm-ensemble}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-25}"

NUM_GPU="${NUM_GPU:-${SLURM_GPUS_ON_NODE:-2}}"
# Accelerate chooses a free local port for each single-node distributed job.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
CONDA_ENV="${CONDA_ENV:-dllm}"
CONDA_INIT="${CONDA_INIT:-/apps/local/conda_init.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_results/ensemble/llada}"
# A suffix requests a separate output/cache namespace for otherwise identical runs.
RUN_SUFFIX="${RUN_SUFFIX:-}"
