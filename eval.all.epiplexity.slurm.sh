#!/bin/bash
#SBATCH --job-name=epiplexity_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=.logs/epiplexity_eval_%j.out

# Source environment
cd /home/sarthak.malla/dllm-epiplexity

source /home/sarthak.malla/miniconda3/etc/profile.d/conda.sh
conda activate dllm


# ===== Mandatory for proper import and evaluation =====
export PYTHONNOUSERSITE=1 # Ensure we use the correct Python environment without interference from user site packages
export PYTHONPATH=/home/sarthak.malla/dllm-epiplexity:/home/sarthak.malla/dllm-epiplexity/lm-evaluation-harness:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHONPATH=$PYTHONPATH"
echo "which python=$(which python)"
echo "which accelerate=$(which accelerate)"

python -c "import lm_eval, pathlib; print('lm_eval=', pathlib.Path(lm_eval.__file__).resolve())"

# ===== Input Arguments =====
BASE_MODEL="GSAI-ML/LLaDA-8B-Base"
PEFT_MODEL=".models/LLaDA-8B-Base-Epiplexity"
NUM_GPU=1
COMMON_ARGS="--model llada --apply_chat_template"

echo "=================================================================="
echo " Starting Epiplexity Instruct Evaluation Suite"
echo " Base Model : $BASE_MODEL"
echo " PEFT Adapter: $PEFT_MODEL"
echo " GPUs       : $NUM_GPU"
echo "=================================================================="


# 5. Coding (HumanEval)
echo ">>> Running HumanEval Instruct..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[126081],begin_suppress_tokens=[]" \
    --confirm_run_unsafe_code

# 6. Coding (MBPP)
echo ">>> Running MBPP Instruct..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# 2. Hard Sciences
echo ">>> Running GPQA Diamond..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=64,steps=64,block_size=64,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# 3. Math (GSM8K)
echo ">>> Running GSM8K CoT..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot --num_fewshot 5 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# 4. Harder Math (Minerva)
echo ">>> Running Minerva Math..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# 1. General Knowledge & Reasoning
echo ">>> Running MMLU Pro..."
accelerate launch --num_processes $NUM_GPU dllm/pipelines/llada/eval.py \
    --tasks mmlu_pro --num_fewshot 0 $COMMON_ARGS \
    --model_args "pretrained=$BASE_MODEL,peft=$PEFT_MODEL,max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

echo "=================================================================="
echo " Evaluation Suite Completed!"
echo " Results should be printed above and saved in the output directories."
echo "=================================================================="