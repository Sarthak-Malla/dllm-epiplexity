#!/bin/bash
#SBATCH --job-name=ctca-native-entropy-controls
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --exclude=gpu-55,gpu-51,gpu-52,gpu-05,gpu-02
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
source scripts/tse/run_logging.sh
tse_print_run_context "$0"

exit_status=0
bash scripts/tse/baseline/llada_parity_suite_slurm.sh \
    scripts/tse/configs/llada_gsm8k300_entropy_fiveshot.conf \
    || exit_status=$?
bash scripts/tse/baseline/dream_parity_suite_slurm.sh \
    scripts/tse/configs/dream_gsm8k300_entropy_st0p1_pt0p1_zeroshot.conf \
    || exit_status=$?
exit "${exit_status}"
