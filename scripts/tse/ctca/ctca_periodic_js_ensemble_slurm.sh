#!/bin/bash
# Submit: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh
# Selective: sbatch --job-name=ctca-selective-js /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh --selective
# Dream main: sbatch --job-name=ctca-selective-js-dream /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh --selective-dream
#SBATCH --job-name=ctca-periodic-js
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
root="/home/juan.guevara/naruto/dllm-epiplexity"
study=ctca-gsm8k-llada-dream-periodic-js-ensemble-pilot-20261124
split=periodic_js
report_args=()
routing_arg=""
case "${1:-}" in
    --selective|--selective-dream)
        study=ctca-gsm8k-llada-dream-selective-js-ensemble-pilot-20261124
        split=selective_js
        report_args=(--selective)
        routing_arg=",routing_confidence_threshold=0.80"
        if [[ "$1" == --selective-dream ]]; then
            study=ctca-gsm8k-dream-llada-selective-js-ensemble-pilot-20261124
            report_args+=(--owner dream)
            routing_arg+=",js_owner=dream"
        fi
        ;;
    "") ;;
    *) echo "Usage: $0 [--selective|--selective-dream]" >&2; exit 2 ;;
esac
cd "$root"
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
run="$root/.logs/$study/${SLURM_JOB_ID:?Submit with sbatch}"
mkdir -p "$run"
git rev-parse HEAD > "$run/git-commit.txt"
git status --short > "$run/git-status.txt"
python "$root/scripts/tie/commit_study.py" split "$split" > "$run/split.json"
llada="/home/juan.guevara/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
dream="/home/juan.guevara/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"
model_args="mode=${split}_ensemble,llada=$llada,dream=$dream,trace_path=$run/trajectory.jsonl,doc_ids_path=$run/split.json"
model_args="$model_args,max_new_tokens=256,steps=128,block_size=16,answer_tokens=16,progress_log_interval=1$routing_arg"
echo "Pilot job $SLURM_JOB_ID on ${SLURM_NODELIST:-unknown}, writing $run"
python -u -m dllm.pipelines.tie.eval \
    --model tie --tasks gsm8k_cot_train_dev --include_path "$root/scripts/tie/tasks" \
    --samples "$run/split.json" --batch_size 1 --num_fewshot 0 --log_samples \
    --output_path "$run" --model_args "$model_args"
exec python "$root/scripts/tse/ctca/periodic_js_ensemble_report.py" "$run/trajectory.jsonl" "$run/split.json" \
    --output "$run/report.json" "${report_args[@]}"
