# Training-free decoding experiments

The first-action selector follow-up now supports full lm-eval generation tasks,
with configurations for GSM8K and HumanEval and W&B logging enabled.
See the
[full-task benchmark instructions](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/README.md#full-task-selector-benchmarks).
The shared diagnostic protocol described below remains the 100-problem development
corpus; the full-test benchmark uses its own run directory and needs no collection.

This suite tests whether decoding is limited by proposals, selection, or committing
dependent predictions simultaneously. It uses **GSM8K test documents 0–99**, not
300 documents. The shared corpus contains **three states per problem: 300 states**.
No experiment has been executed as part of setup.

The eventual objective is improved accuracy, or fewer total model evaluations
with at most one percentage point of accuracy loss. These 100 previously explored
development examples select a candidate for larger confirmation; their point
estimates, nonsignificant differences, or zero observed disagreements cannot
certify that margin. No learned scheduler or predictor is included.

## Fixed protocol

- Pinned LLaDA-8B-Instruct revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`,
  BF16, batch size one, existing GSM8K five-shot chat prompts and strict/flexible filters.
- 256 generated positions, block size 64, deterministic token predictions,
  proposal generation seed 42, four candidates, verifier chunk size one.
- Original adaptive reference: entropy budget 2, maximum action size four,
  legacy outgoing utility, confidence exponent zero, existing conflict/anchor settings.
- The fixed-four control changes cardinality while retaining those original seed
  and companion mechanics. It does not use the later corrected-seed arms.
- Capture the first pre-commit state at or beyond 0, 85, and 170 revealed response
  positions on the adaptive entropy-selector trajectory. Build adaptive and
  fixed-four pools from the same model predictions.
- Every diagnostic branch continues with the same **adaptive maximum-confidence
  policy**. Complete-policy benchmarks are separate, without counterfactual forks.
- Cheap selection means maximum mean confidence. Entropy selection retains the
  original entropy-drop-per-action-size rule.

## Stages

| Stage | Question | Consequence of a useful result |
|---|---|---|
| [01: selectors](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/README.md) | Does entropy lookahead choose better actions on identical states? | Retain lookahead only if its paired benefit justifies its cost. |
| [02: precedence](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/02_precedence/README.md) | Does low conflict predict stable values, and does refreshing companions help? | Distinguish a useful stability proxy from harmful simultaneous commitment. |
| [03: proposals](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/03_proposals/README.md) | Are better actions present in confidence-generated pools? | Separate missing useful proposals from a selector overlooking them. |
| [04: attention mass](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/04_attention_mass/README.md) | Does retaining absolute mass improve prediction or decoding? | Change attention representation only if it improves a measured tradeoff. |
| [05: blocks](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_block_scope/README.md) | Does full-response eligibility improve confidence-only decoding? | Decide whether the block restriction is useful without attention confounds. |
| [06: groups](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/06_confidence_groups/README.md) | Does attention ranking add value after a confidence gate? | Keep the least expensive useful group-selection rule. |
| [07: growth](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/07_affordable_companions/README.md) | Does searching for affordable companions reduce evaluations safely? | Improve budget use without attributing gains to changed seeds or scores. |

Run 01 and 02 first and inspect their report. Later stages are independently
selected commands. The suite never automatically runs all arms, combines changes,
submits jobs, or makes a ground-truth-based choice during inference.

## Environment and execution

**The user submits all jobs.** Each GPU launcher requests **two GPUs**, 24 CPUs,
and 64 GB RAM, following the existing ablation launchers' partition and QoS.
One Slurm task starts two independent model workers, each processing 50 documents
at batch size one. The launcher explicitly assigns one model to each GPU.
The analyzer merges the workers into one 100-problem report.

The Slurm scripts prepare the environment automatically. For manual compute-node
commands, including tests and analysis, prepare it first:

```bash
if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:/home/sarthak.malla/dllm-selection-ensemble/lm-evaluation-harness:"${PYTHONPATH:-}"
```

The fallback matches the existing launchers because `.zshrc` is absent in this
workspace. The inspected environment is `/home/sarthak.malla/.conda/envs/dllm`.

Validate the implementation on a compute node first:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 \
    python -m pytest \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_replay.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_mechanisms.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_artifacts.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_experiment_telemetry.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_eval_device_routing.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_analysis.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_runner.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_training_free_imports.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_adaptive_cardinality.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_aggregation.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_sinks.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_parallel_candidates.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_non_lookahead.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_batched_lookahead.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_counterfactual_oracle.py
```

Then collect the shared states. W&B logging defaults to online; use your existing
W&B credentials. Set `WANDB_PROJECT` and optionally `WANDB_ENTITY` before submission
to select the destination. No prompts, responses, or token vectors are uploaded:

```bash
export WANDB_PROJECT=dllm-selection-ensemble
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/collect.slurm.sh
```

After collection completes, submit stage 01, inspect its report, and submit stage
02. Each experiment directory has its own `slurm.sh`; later stages are separate
manual submissions. Do not submit different stages concurrently to the same run.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/slurm.sh
```

| Launcher | Allocation | Work |
|---|---|---|
| Collection | 4 hours | 100 carrier trajectories and 300 snapshots |
| 01 selectors | 24 hours | Paired diagnostic, then separate entropy and cheap benchmarks |
| 02 precedence | 24 hours | Seed probes, refreshed winners, and both pair orders |
| 03 proposals | 24 hours | All unique candidates from both pools |
| 04 attention mass | 24 hours | Frozen-group diagnostic and full-policy benchmark |
| 05 block scope | 4 hours per task | Six-task array: three thresholds × two block sizes |
| 06 confidence groups | 4 hours per task | Six-task array: three thresholds × two rankings |
| 07 affordable companions | 4 hours | Complete cheap-selector benchmark |
| Optional seed-first / fixed-four controls | 6 hours per task | Separate launchers under stages 02 / 03 |

These are conservative initial allocation estimates, not measured runtimes. The
diagnostics can require many continuations. Resubmit the same command after a
timeout to resume completed units. Arrays use `%1`, so each array has one active
task and requests two GPUs in total. The partition and QoS can be overridden with
normal `sbatch` arguments.

The default run directory is
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_two_gpu`.
Export `TF_RUN_TAG` consistently for all stages to create a separate run. Worker
artifacts live in its `workers/worker0` and `workers/worker1` subdirectories; the
parent launcher locks the run against simultaneous writers. Configuration/source
changes require a new run tag. Completed units resume automatically.

For a direct two-GPU allocation, after environment preparation:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:2 --cpus-per-task=24 --time=04:00:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py \
    --output-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_two_gpu \
    -- --resume --wandb-mode online --wandb-project dllm-selection-ensemble \
    --wandb-group training_free_v1_two_gpu collect
```

Global runner flags after the separator and before the stage, such as
`--doc-start 0 --doc-stop 10`, limit one invocation without changing the immutable
100-document protocol. Both workers retain the same document-to-GPU assignment
on resume. The original single-worker runner remains available for separate runs;
do not mix its artifacts with a two-worker corpus.

Inside an existing one-GPU interactive allocation, use an `srun` job step so
Slurm applies the allocated GPU assignment. `--overlap` lets the step run alongside
the interactive shell within that allocation. After the same environment preparation,
this command tries two documents in a separate one-GPU run directory:

```bash
srun --overlap --ntasks=1 --gres=gpu:1 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/runner.py \
    --output-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_one_gpu \
    --device cuda:0 --resume --doc-start 0 --doc-stop 2 collect
```

Remove the document-range flags to continue through all 100 documents. The one-GPU
and two-GPU corpora remain separate because snapshots record the CUDA RNG layout.

Analyze completed units between stages:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/analyze.py \
    --output-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_two_gpu
```

## W&B logging

The default project is `dllm-selection-ensemble`, with one group per run tag and
one persistent run per stage and worker. Logs include progress, flexible/strict
accuracy, actual model calls and evaluated rows, latency, action sizes, selector
agreement and paired wins/losses, and seed-induced flips and probability loss.
Counters rebuild from completed local artifacts on resume and do not count cached
work twice. W&B is a monitoring view; the merged local analysis supplies the
problem-clustered intervals and cross-worker comparisons.

Use `export WANDB_MODE=offline` before submission to retain W&B logs locally for
later synchronization, or `disabled` to opt out. Logging failures are reported
explicitly. W&B does not upload datasets, model weights, or replay snapshots.

## Artifacts and interpretation

The immutable manifest records source/configuration hashes, checkpoint identity,
library versions and GPU type, the exact document IDs, and document/prompt/token/target hashes. A mismatch rejects
resume. Response/request caches are disabled; W&B follows the settings above. Individually
completed documents, states, probes, and branches can be reused; snapshots contain
the full replay state, including RNG and anchors, with payload-integrity hashes.
Do not load snapshot pickles from untrusted sources.

`analysis/report.md` and `analysis/summary.json` summarize coverage and conclusions;
`paired_comparisons.csv`, `benchmarks.csv`, `stability_strata.csv`, and
`companion_stability.csv` retain numerical results. Diagnostic intervals resample
problems, not tokens or states. Flexible extraction is primary; strict extraction
is always reported as well. Unfinished stages remain untested, never failed.

Low attention conflict, stable token values, and final correctness are separate
measurements. Stability analysis keeps attention direction, confidence weighting,
group size, stage, marginal uncertainty, and absolute attention mass visible.
It does not fit a learned predictor or assert conditional independence.

Every actual outer model invocation is counted, along with evaluated sequence rows,
input-token work, and synchronized model time. Four candidate rows in one call are
four evaluations. A disjoint transaction ledger accounts for collection, replay,
verification, probes, branches, and benchmarks; reused artifacts add no fictitious
model cost. Branch/probe summaries are not added again to ledger totals. Completed
or caught-failure transactions are durable; forced termination can leave an
unrecorded partial transaction, which the report explicitly acknowledges.

Complete benchmarks include their deployed proposal/attention/refresh work and
minimal action logging, but no diagnostic forks. Generation time excludes answer
extraction; transaction wall time includes artifact processing. Summed worker time
is aggregate work, while launcher elapsed time measures concurrent job duration.
Model loading is
outside per-document generation measurements. Attention timings with asynchronous
GPU work are descriptive component timings; synchronized total generation time is
the end-to-end latency comparison.

The development nomination minimizes measured evaluations among completed arms
whose point accuracy is within one point of the entropy reference, using latency
to break ties. It is a nomination for confirmation, not a preservation claim.
If extra mechanisms do not improve the tradeoff, retain the inexpensive reference.
