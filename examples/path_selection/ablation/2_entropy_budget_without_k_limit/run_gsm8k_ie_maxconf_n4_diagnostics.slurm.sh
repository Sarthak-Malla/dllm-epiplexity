#!/bin/bash
#SBATCH --job-name=abl2-ie-maxconf-n4-diag
#SBATCH --output=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.out
#SBATCH --error=/home/sarthak.malla/dllm-selection-ensemble/.logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Submit with:
# sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_ie_maxconf_n4_diagnostics.slurm.sh
# Optional eight-example check: sbatch --export=ALL,PATH_ABLATION_LIMIT=8 <absolute path above>

export PATH_ABLATION_NUMBER=2
export PATH_ABLATION_OUTPUT_SUBDIRECTORY=2_entropy_budget_without_k_limit
export PATH_ABLATION_SAMPLER_TYPE=max_confidence
export PATH_ABLATION_CARDINALITY_STRATEGY=entropy_budget
export PATH_ABLATION_CANDIDATE_BUDGET=4
export PATH_ABLATION_SEED_STRATEGY=incoming
export PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0
export PATH_ABLATION_TOKEN_TEMPERATURE=0.0
# The shared runner defaults to exponent zero; avoid an extra path component.
unset PATH_ABLATION_CONFIDENCE_EXPONENT
export PATH_ABLATION_NON_LOOKAHEAD_DIAGNOSTICS=1
export ABLATION2_ENTROPY_BUDGET=2.0
export ABLATION2_MAXIMUM_ACTION_SIZE=64
export ABLATION2_RUN_TAG=${ABLATION2_RUN_TAG:-max_confidence_ie_diagnostics_v1}
source /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_common.sh
