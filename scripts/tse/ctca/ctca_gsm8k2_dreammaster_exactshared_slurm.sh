#!/bin/bash
# Submit with: sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_dreammaster_exactshared_slurm.sh
# Runs matched Dream-master relative and exact-shared traces for both questions.
#SBATCH --job-name=ctca-gsm8k2-dreammap
#SBATCH --output=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.out
#SBATCH --error=/home/juan.guevara/naruto/dllm-epiplexity/.logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclude=gpu-51,gpu-24,gpu-38

set -eo pipefail
cd /home/juan.guevara/naruto/dllm-epiplexity
export SLURM_SUBMIT_DIR="$PWD"
job_id="${SLURM_JOB_ID:-manual}"
root=".logs/ctca-gsm8k2-dreammaster-exactshared"

for variant in relative exactshared; do
    for case_name in correct wrong; do
        bash /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_static_smoke_slurm.sh \
            "/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_dreammaster_${case_name}_${variant}.conf"
    done
    experiment_name="ctca-gsm8k2-dreammaster-st0p0-pta1p0-ptb1p0-w0p5-probmean-confrank-${variant}-fulltrace-20261029"
    manifest_dir="${root}/${variant}"
    run_url="/${root}/${variant}/${experiment_name}/${job_id}"
    test -s "${manifest_dir}/${experiment_name}/${job_id}/correct.jsonl"
    test -s "${manifest_dir}/${experiment_name}/${job_id}/wrong.jsonl"
    printf '{"job_id":"%s","traces":["%s/correct.jsonl","%s/wrong.jsonl"]}\n' \
        "${job_id}" "${run_url}" "${run_url}" > "${manifest_dir}/latest.json.tmp"
    mv "${manifest_dir}/latest.json.tmp" "${manifest_dir}/latest.json"
    echo "CTCA viewer manifest: ${manifest_dir}/latest.json"
done
