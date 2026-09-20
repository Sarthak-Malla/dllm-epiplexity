#!/bin/bash
# Submit ensemble or baseline jobs from a login or compute node, or preview.
# Usage: bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh [--dry-run] [--limit N] [all|baselines|MODE]

set -eo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh [--dry-run] [--limit N] [all|baselines|MODE]' \
        'all (default): submit candidate_expansion and majority_voting as separate jobs.' \
        'baselines: submit greedy, min_entropy, and max_top2_prob as separate jobs.' \
        'MODE: greedy, min_entropy, max_top2_prob, candidate_expansion, or majority_voting.' \
        '--limit N evaluates N examples per task and policy (or a fraction between 0 and 1).' \
        '--dry-run prints submission commands without creating directories or submitting jobs.'
}

dry_run=false
selection=''
evaluation_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true ;;
        --limit)
            if [[ $# -lt 2 || ! "$2" =~ ^([1-9][0-9]*|0?\.[0-9]*[1-9][0-9]*)$ ]]; then
                printf '%s\n' '--limit requires a positive integer or a fraction between 0 and 1.' >&2
                exit 2
            fi
            evaluation_args=(--limit "$2")
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        all|baselines|greedy|min_entropy|max_top2_prob|candidate_expansion|majority_voting)
            if [[ -n "$selection" ]]; then
                printf 'Specify one mode or one group (all or baselines).\n' >&2
                exit 2
            fi
            selection="$1"
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "${selection:-all}" in
    all) policies=(candidate_expansion majority_voting) ;;
    baselines) policies=(greedy min_entropy max_top2_prob) ;;
    *) policies=("$selection") ;;
esac

if [[ "$dry_run" == false ]]; then
    # Slurm opens logs before the job script starts, so prepare this directory now.
    mkdir -p /home/sarthak.malla/dllm-learning-decoding-path/.logs
fi

for policy in "${policies[@]}"; do
    job_script="/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/${policy}.slurm.sh"
    submit_command=(sbatch "$job_script" "${evaluation_args[@]}")
    printf '%q ' "${submit_command[@]}"
    printf '\n'
    if [[ "$dry_run" == false ]]; then
        "${submit_command[@]}"
    fi
done
