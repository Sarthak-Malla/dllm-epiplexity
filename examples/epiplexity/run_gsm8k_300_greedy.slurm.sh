#!/bin/bash
#SBATCH --job-name=epipath-gsm8k300-greedy
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64

set -euo pipefail

cd /home/sarthak.malla/dllm-epiplexity
mkdir -p /home/sarthak.malla/dllm-epiplexity/.logs

if [[ -f /home/sarthak.malla/.zshrc ]]; then
    source /home/sarthak.malla/.zshrc
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate /home/sarthak.malla/miniconda3/envs/dllm
set -u

export PYTHONPATH=/home/sarthak.malla/dllm-epiplexity:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/sarthak.malla/.cache/huggingface}"

model_name_or_path="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
num_gpu="${SLURM_GPUS_ON_NODE:-4}"
output_root=/home/sarthak.malla/dllm-epiplexity/eval_results/epipath
run_name=gsm8k_300_greedy
sample_file="${output_root}/gsm8k_300_indices.json"

mkdir -p "${output_root}/${run_name}"
python -c "import json, pathlib; path = pathlib.Path('${sample_file}'); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({'gsm8k_cot': list(range(300))}) + '\n')"

model_args="pretrained=${model_name_or_path},max_new_tokens=256,steps=64,block_size=64,cfg_scale=0.0,sampler_type=greedy"

echo "Running greedy GSM8K subset evaluation"
echo "Subset file: ${sample_file}"
echo "Output: ${output_root}/${run_name}"

accelerate launch \
    --num_processes "${num_gpu}" \
    /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/eval.py \
    --model llada_epiplexity \
    --apply_chat_template \
    --tasks gsm8k_cot \
    --num_fewshot 5 \
    --samples "${sample_file}" \
    --log_samples \
    --model_args "${model_args}" \
    --output_path "${output_root}/${run_name}" \
    --use_cache "${output_root}/${run_name}/responses.cache"

echo "Greedy GSM8K-300 evaluation completed."
