#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/tie_gsm8k32_pilot_slurm.sh
#SBATCH --job-name=tie-gsm8k32
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

set -eo pipefail
echo "Starting job ${SLURM_JOB_ID:-manual} on ${SLURM_NODELIST:-unknown}"
cd /home/juan.guevara/naruto/dllm-epiplexity
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
echo "Using $(command -v python)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

tie_output="/home/juan.guevara/naruto/dllm-epiplexity/.logs/tie-gsm8k32/${SLURM_JOB_ID:?Submit this launcher with sbatch}"
mkdir -p "$tie_output"
echo "Writing TIE pilot results to $tie_output"
exec python -u -c 'import runpy; print("Python started. Importing dLLM.", flush=True); runpy.run_module("dllm.pipelines.tie.eval", run_name="__main__", alter_sys=True)' \
    --model tie --tasks gsm8k_cot --limit 32 --batch_size 1 --num_fewshot 0 \
    --log_samples --output_path "$tie_output" \
    --model_args "mode=relay,trace_path=$tie_output/relays.jsonl,progress_log_interval=1"
