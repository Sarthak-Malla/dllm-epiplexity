#!/bin/bash
# epiplexity/eval_epiplexity.sh
# 
# Script to evaluate the LLaDA model customized via Epiplexity SFT.
# Usage:
#   bash epiplexity/eval_epiplexity.sh [NUM_GPUS] [PEFT_DIR] [TASKS]
#
# Example:
#   bash epiplexity/eval_epiplexity.sh 1 .models/LLaDA-8B-Base-Epiplexity mmlu_pro

NUM_GPUS=${1:-1}
PEFT_MODEL=${2:-".models/LLaDA-8B-Base-Epiplexity"}
TASKS=${3:-"mmlu_pro"}
BASE_MODEL="GSAI-ML/LLaDA-8B-Base"

echo "=================================================================="
echo " Starting Epiplexity Evaluation"
echo " Base Model : $BASE_MODEL"
echo " PEFT Adapter: $PEFT_MODEL"
echo " Tasks      : $TASKS"
echo " GPUs       : $NUM_GPUS"
echo "=================================================================="

# Export PYTHONPATH so dllm internal imports work correctly
export PYTHONPATH=.

accelerate launch --num_processes $NUM_GPUS \
    dllm/pipelines/llada/eval.py \
    --tasks "$TASKS" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0"
