#!/bin/bash
# Submit with: sbatch /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_fulltrace_slurm.sh
# Runs both previously correct and wrong GSM8K prompts under one scheduler job.
#SBATCH --job-name=ctca-gsm8k2-fulltrace
#SBATCH --output=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

set -eo pipefail
cd /home/sarthak.malla/dllm-epiplexity
for case_name in correct wrong; do
    bash /home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_static_smoke_slurm.sh \
        "/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_fulltrace_${case_name}.conf"
done

experiment_name="ctca-gsm8k2-llada8bi-dream7bi-st0p0-pta1p0-ptb0p1-w0p5-probmean-confrank-fulltrace-20261027"
job_id="${SLURM_JOB_ID:-manual}"
manifest_dir=".logs/ctca-gsm8k2-fulltrace"
run_url="/.logs/ctca-gsm8k2-fulltrace/${experiment_name}/${job_id}"
test -s "${manifest_dir}/${experiment_name}/${job_id}/correct.jsonl"
test -s "${manifest_dir}/${experiment_name}/${job_id}/wrong.jsonl"
printf '{"job_id":"%s","traces":["%s/correct.jsonl","%s/wrong.jsonl"]}\n' \
    "${job_id}" "${run_url}" "${run_url}" > "${manifest_dir}/latest.json.tmp"
mv "${manifest_dir}/latest.json.tmp" "${manifest_dir}/latest.json"
echo "CTCA viewer manifest: ${manifest_dir}/latest.json"
