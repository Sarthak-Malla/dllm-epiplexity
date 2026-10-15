#!/bin/bash
# Shared, source-only logging helpers for TSE experiment launchers.

tse_print_run_context() {
    local runner_path
    runner_path="$(realpath "$1")"

    echo "===== Run context ====="
    echo "runner=${runner_path}"
    echo "runner_sha256=$(sha256sum "${runner_path}" | awk '{print $1}')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_dirty_files=$(git status --short 2>/dev/null | wc -l)"
    echo "date=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "slurm_job_name=${SLURM_JOB_NAME:-}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-}"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-}"
    echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
    echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-}"
    echo "----- Embedded runner configuration -----"
    awk '
        /^#SBATCH/ ||
        /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*=/ ||
        /^[[:space:]]*(accelerate|python scripts\/tse\/|--[A-Za-z0-9_-]+)/ {
            print
        }
    ' "${runner_path}"
    echo "----- End embedded configuration -----"
    echo "======================="
}

tse_print_config_file() {
    local config_path
    config_path="$(realpath "$1")"

    echo "===== Configuration file ====="
    echo "config_path=${config_path}"
    echo "config_sha256=$(sha256sum "${config_path}" | awk '{print $1}')"
    sed -n '1,240p' "${config_path}"
    echo "===== End configuration ====="
}
