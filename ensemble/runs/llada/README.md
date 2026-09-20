# Shared LLaDA evaluations

Every mode uses [config.sh](/home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/config.sh) and the same [launcher](/home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh). The launcher runs [the ensemble lm-evaluation-harness pipeline](/home/sarthak.malla/dllm-learning-decoding-path/ensemble/pipelines/llada/eval.py), registered as `llada_ensemble`.

The defaults preserve the supplied full GSM8K setup: `GSAI-ML/LLaDA-8B-Instruct`, `gsm8k_cot`, five few-shot examples, chat template enabled, 256 generated tokens, block size 64, guidance 0.0, and temperature 0.0. Batch size is 1 per process, and the harness receives fixed Python, NumPy, PyTorch, and few-shot seeds (`0,1234,1234,1234`). Sample logging is enabled. Both suppression lists are explicitly empty, following the supplied run rather than the task-specific suppression settings in [the original examples](/home/sarthak.malla/dllm-learning-decoding-path/examples/llada/eval.sh).

| Mode | Position selection | Scheduling |
| --- | --- | --- |
| `greedy` | Proposed-token probability (`low_confidence`) | 64 configured steps |
| `min_entropy` | Negative entropy | 64 configured steps |
| `max_top2_prob` | Top-1 minus top-2 probability | 64 configured steps |
| `candidate_expansion` | Unanimous agreement, expanding proposals if needed | Dynamic; no `steps` argument |
| `majority_voting` | Strict majority agreement, expanding proposals if needed | Dynamic; no `steps` argument |

The ensemble defaults use `low_confidence`, `min_entropy`, and `max_top2_prob`. Like MDLM, decoding completes one block before moving to the next; only masked positions in the current block are eligible, while previously revealed tokens supply context. At each decoding step, each strategy proposes `k = ceil(0.10 * remaining_masks)` positions for each active sequence. The candidate count therefore shrinks as the block fills. Candidate expansion still raises `k` for that step if the selected agreement policy finds no positions to commit; the next step recalculates `k` from the new remaining-mask count.

With 64, 48, 32, and 16 masks remaining, the proposed counts are 7, 5, 4, and 2 respectively. Agreement determines the number committed. If expansion reaches all remaining masks, every strategy proposes all of them and the block finishes. `CANDIDATE_FRACTION=0.10` controls the fraction at every step. These rules use the current remaining masks without a fixed step schedule. The number of model evaluations can differ from the scheduled baselines; this setup matches generation and evaluation settings, not a fixed compute budget.

The ensemble automatically excludes the mask token from predictions to ensure progress; the baselines retain native MDLM behavior. The inherited evaluation harness requires a numeric batch size dividing `mc_num` (128 by default), including for generation tasks; the shared batch size of 1 meets that requirement.

Preview any mode without loading Conda, downloading models, creating directories, starting a job, or initializing W&B:

```bash
bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run greedy
bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run candidate_expansion
bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run majority_voting
```

Submit from a login or compute node; Slurm runs evaluation on the allocated compute node. The batch defaults are one node, two GPUs, 64 CPUs, 64 GB RAM, 1 hour, partition `cscc-gpu-p`, and QoS `cscc-gpu-qos`. All batch scripts exclude nodes `gpu-54` and `gpu-05`. Create the log directory before submission because Slurm opens its logs before the script starts:

```bash
mkdir -p /home/sarthak.malla/dllm-learning-decoding-path/.logs
sbatch --job-name=llada-greedy /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh greedy
sbatch --job-name=llada-candidate-expansion /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh candidate_expansion
sbatch --job-name=llada-majority-voting /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh majority_voting
```

The launcher runs `srun --nodes=1 --ntasks=1 --gres=gpu:2 accelerate launch --num_processes 2 ...` by default. Slurm starts one Accelerate launcher, which starts one worker per allocated GPU. The `srun` GPU count and Accelerate process count both follow `NUM_GPU`. This creates a step within the existing allocation. For an existing allocation, invoke the script with `bash` from the allocated compute node and set `NUM_GPU` to that allocation's GPU count if needed. Actual evaluation requires both `SLURM_JOB_ID` and `SLURMD_NODENAME`; dry runs work without an allocation. Do not run actual evaluations on a login node.

Before Accelerate starts, the launcher runs `srun --nodes=1 --ntasks=1 --gres=gpu:2 nvidia-smi` to record GPU and driver information in the job log; the GPU count follows `NUM_GPU`. Dry runs print this command without executing it.

Dedicated job scripts are available for the [greedy](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/greedy.slurm.sh), [minimum entropy](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/min_entropy.slurm.sh), and [top-two probability margin](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/max_top2_prob.slurm.sh) baselines, plus [candidate expansion](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/candidate_expansion.slurm.sh) and [majority voting](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/majority_voting.slurm.sh). Every job uses the one-hour allocation above and inherits the shared configuration, Conda initialization, GPU diagnostics, and W&B logging. The [submission helper](/home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh) creates the log directory before submission. `baselines` selects the three scheduled baselines; `all` (the default) continues to select only the two ensembles. Any individual mode can also be selected:

```bash
# Preview the two submission commands without submitting jobs.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --dry-run

# From a login or compute node, submit both ensemble jobs.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh all

# Submit both ensemble policies on 300 GSM8K examples from a login or compute node.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --limit 300 all

# Preview the 300-example submissions without allocating GPUs or starting jobs.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --dry-run --limit 300 all

# Preview a job's resolved evaluation parameters without submitting it.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/candidate_expansion.slurm.sh --dry-run

# Preview the three baseline submissions without starting jobs.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --dry-run --limit 300 baselines

# From a login or compute node, submit all three baselines on 300 examples each.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --limit 300 baselines

# Submit one baseline; min_entropy and max_top2_prob work the same way.
bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --limit 300 greedy

# Use an optional suffix for fresh baseline decoding and new decoding metrics.
RUN_SUFFIX=baseline_v1 bash /home/sarthak.malla/dllm-learning-decoding-path/scripts/ensemble/slurm.sh --limit 300 baselines
```

Each job requests two GPUs independently: the two ensemble jobs request four GPUs in total, and the three baseline jobs request six GPUs in total if run concurrently. `STEPS` remains in the shared configuration for scheduled baselines; ensemble jobs use dynamic decoding.

Distributed runs use `MAIN_PROCESS_PORT=0` by default, allowing Accelerate to select a free port for each job when both policies run on the same node. Set `MAIN_PROCESS_PORT` explicitly if a fixed port is required.

Override any default through the environment. Keep the same overrides across modes for comparisons; edit the shared config if they should become defaults. For example, preview a small evaluation:

```bash
LIMIT=10 NUM_GPU=1 RUN_SUFFIX=smoke \
    bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run candidate_expansion
```

`--limit 300` overrides `LIMIT` from the environment or shared config. It accepts a positive integer count per task or a fraction strictly between 0 and 1; omitting both the argument and `LIMIT` evaluates the full dataset. The limit is part of the evaluation cache hash. For the full resolved Accelerate command on 300 examples:

```bash
bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run --limit 300 candidate_expansion
```

Common overrides include `MODEL_NAME_OR_PATH`, `MAX_NEW_TOKENS`, `BLOCK_SIZE`, `CFG_SCALE`, `TEMPERATURE`, `SUPPRESS_TOKENS`, `BEGIN_SUPPRESS_TOKENS`, `TASKS`, `NUM_FEWSHOT`, `BATCH_SIZE`, `SEED`, `APPLY_CHAT_TEMPLATE`, `LIMIT`, and `OUTPUT_ROOT`. `STEPS` affects only baselines; `CANDIDATE_FRACTION` and `STRATEGIES` affect only ensembles. Candidate settings are included in each ensemble's cache hash. Removing the former `candidate_growth` argument gives these ensemble runs a different cache namespace from runs with growing candidate counts. Suppression lists and strategy lists use semicolons between values, for example `STRATEGIES='[low_confidence;min_entropy;max_top2_prob]'`.

W&B is enabled by default with project `dllm-ensemble`. `WANDB_PROJECT` and optional `WANDB_ENTITY` choose the destination; each evaluation creates one run on rank 0, named after its output directory. Use an existing `wandb login` or provide `WANDB_API_KEY` through the environment. The launch command never includes the API key. Set `WANDB_ENABLED=false` to disable logging, `WANDB_MODE=offline` to save W&B data locally without uploading, or `WANDB_MODE=disabled` to use W&B's disabled mode. The launcher preserves the supplied `WANDB_MODE`.

W&B receives the final harness scores across all ranks, evaluation configuration, result table, and result JSON artifact. Per-example generations remain in the local `--log_samples` output and are not uploaded to W&B. During decoding, `WANDB_LOG_EVERY=25` logs metrics every 25 decoding events, with a final flush for a shorter last window. Token and expansion metrics summarize the window; remaining masks reports its latest value:

| Metric | Meaning |
| --- | --- |
| `decoding/tokens_per_sequence_step` | Mean number of committed tokens per active sequence per decoding step. |
| `decoding/remaining_masks_in_block` | Latest total remaining masked positions across the batch in the current block after committing. |
| `decoding/expansion_rate` | Fraction of active sequence steps that needed candidate expansion; ensembles only. |

Ensembles also measure position overlap on every decoding step, for every active
sequence in the logged batch. If each strategy proposes a set `C_s` of `k`
eligible positions, all-strategy overlap is `|intersection(C_s)|` and pairwise
overlap is `|C_a intersect C_b|`. These are counts of shared positions. Each pair
needs only one symmetric metric. Proposal counts are also recorded separately
for every strategy, before applying the ensemble's agreement rule.

Both measurements are recorded before candidate expansion, at the candidate
count determined by remaining masks, and after expansion, at the final candidate
count used for voting. Both phases precede committing tokens. Expansion can
raise agreement; once `k` equals the remaining masks, every overlap count equals
the remaining-mask count. Majority voting can commit positions even when the
all-strategy intersection is empty.

In W&B, `decoding/overlap/{phase}/{comparison}_count` is the logging-window mean
number of shared positions per active sequence-step. The phases are
`before_expansion` and `after_expansion`; comparisons are `all` and the configured
strategy pairs, such as `low_confidence__min_entropy`.
`decoding/proposals/{phase}/{strategy}_count` records the corresponding mean
number of positions proposed by each strategy, such as `min_entropy`. Currently,
all strategies use the same `k`, so their proposal-count curves coincide.

The overlap `_count_distribution` charts accumulate counts from **every**
observed sequence-step, including zero overlap and intervening events between
W&B uploads. Histograms use at most 64 bins with integer-aligned boundaries;
wider bins cover large counts without clipping values above 100. The final run
summary retains overlap histograms and exact `_mean_count` and `_observations`
for both overlaps and proposals. Each active sequence-step has equal weight;
finished rows contribute no observation. Only count frequencies and running
sums are retained in memory. Percentage metrics are no longer emitted.

With `BATCH_SIZE=1` and `WANDB_LOG_EVERY=1`, each count chart point gives the exact
number for one sequence at one decoding step. With larger batches or logging
intervals, these counts are averaged over active sequence-steps and can be
fractional. Use a fresh `RUN_SUFFIX` to collect these metrics instead of reusing
cached responses; existing W&B runs retain their original percentage metrics.

These are intersections between strategies **within each step**, summarized
across decoding steps and blocks; committed positions are removed from later
candidate sets, so intersecting commits across time would not measure agreement.

The run summary also records the observed decoding step count and cumulative metric means. These decoding metrics cover rank 0's actually decoded, uncached requests; they are not an aggregate across GPUs. A fully cached rerun has final evaluation scores but no new decoding events. Change `RUN_SUFFIX` to decode a fresh run. Logging settings do not affect the evaluation cache namespace.

For example, preview an offline run or disable W&B while retaining local results:

```bash
WANDB_MODE=offline WANDB_LOG_EVERY=10 \
    bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run candidate_expansion
WANDB_ENABLED=false \
    bash /home/sarthak.malla/dllm-learning-decoding-path/ensemble/runs/llada/eval.sh --dry-run majority_voting
```

Dry runs only print the command and settings; they never initialize W&B or contact its service.

Results and response caches are separated by task, sampler mode, and a hash of resolved evaluation arguments beneath `/home/sarthak.malla/dllm-learning-decoding-path/eval_results/ensemble/llada`. Changing relevant parameters or seeds creates a different directory. Identical settings intentionally reuse the same cache; use a new `RUN_SUFFIX` for an independent rerun, including after model or code changes that leave the launch arguments unchanged. The complete shell-escaped command and output path are printed before execution.

Environment initialization sources `/home/sarthak.malla/.zshrc` when available, then `/apps/local/conda_init.sh`, and activates `dllm`. Set `CONDA_INIT` or `CONDA_ENV` if the cluster environment differs.

The environment must have this repository's dependencies and a compatible lm-evaluation-harness installed. The existing `dllm` environment already provides harness 0.4.9.1, including `gsm8k_cot`, through `/home/sarthak.malla/dllm-selection-ensemble/lm-evaluation-harness`; that installation is sufficient. For a fresh environment, initialize this repository's `/home/sarthak.malla/dllm-learning-decoding-path/lm-evaluation-harness` submodule and install it following [the repository setup instructions](/home/sarthak.malla/dllm-learning-decoding-path/README.md).
