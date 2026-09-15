# 01 — Same-state selector disagreements

Use the environment preparation and shared collection command in the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** entropy lookahead adds little action-selection value beyond mean
confidence. On each of the 300 frozen states, score exactly the same adaptive pool
both ways, then repeat with fixed-four candidates. Keep original token values and
the common cheap continuation fixed. Identical branches are reused.

Measure exact selected-set agreement, Jaccard overlap, duplicate candidates, action
sizes, original token IDs, group confidence/entropy, and paired correctness. The
fixed-four comparison separates selector behavior from variable action size.

Frequent agreement indicates redundant selections. Frequent disagreements with
balanced continuation wins/losses indicate unresolved or weak scoring value.
Cheap continuation wins support a scoring problem in this corpus; they do not by
themselves establish that the candidate generator is good.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/slurm.sh
```

The two-GPU diagnostic job requests 24 hours. It runs the paired diagnostic, then
separate complete entropy and cheap policy benchmarks. Resubmission resumes
completed work; W&B logging follows the suite settings.

## Full-task selector benchmarks

The follow-up compares three complete policies:

- `reference_cheap`: mean confidence at every action.
- `first_action_entropy`: entropy lookahead at the first action of each response,
  then mean confidence throughout the remaining response, including later blocks.
- `reference_entropy`: entropy lookahead at every action.

The generic worker reads a JSON task configuration and discovers the evaluation
documents from lm-eval at runtime. It uses lm-eval's prompts, filters, and metrics.
There are no hardcoded dataset sizes or task-name branches in the new worker.
The shared dependency candidate-generation settings are retained across arms.

| Configuration | lm-eval task | Few-shot | Generated positions / block / steps | Primary metric and filter |
|---|---|---:|---|---|
| [GSM8K](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/benchmarks/gsm8k.json) | `gsm8k_cot` | 5 | 256 / 64 / 64 | `exact_match,flexible-extract` |
| [HumanEval](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/benchmarks/humaneval.json) | `humaneval_instruct_llada` | 0 | 512 / 512 / 512 | `pass@1,create_test` |

HumanEval follows the local
[lm-eval task configuration](/home/sarthak.malla/dllm-selection-ensemble/lm-evaluation-harness/lm_eval/tasks/humaneval/humaneval_instruct_llada.yaml),
including its code-completion prompt, one completion per problem, prediction filter, and
code-test evaluation. Its JSON configuration enables the required code-evaluation
flag. Both tasks use the complete test split; training data are not evaluation cases.

HumanEval also sets `suppress_tokens: [126081]` (`<|endoftext|>`) and
`begin_suppress_tokens: []`, following the native
[reproduction command](/home/sarthak.malla/dllm-selection-ensemble/examples/llada/eval.sh).
For the dependency policies, action sizes remain adaptive: `steps: 512` does not
force 512 actions. Their confidence-only arm is not the native LLaDA sampler.

### Native HumanEval reproduction check

Before interpreting differences against the README's reproduced 47.0%, submit:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/humaneval_reproduction.slurm.sh
```

This separate one-GPU job uses the native `llada` evaluator and the README's
HumanEval task and sampling arguments. It logs aggregate results to W&B without
uploading samples or collecting diagnostic traces. It adds one benchmark and one
W&B run to the totals below. It has a four-hour allocation and no response cache.
The README supplies configuration guidance, not a guarantee of identical scores
across software revisions and hardware.

The old greedy run scored 23.78% on `humaneval_instruct`, using 1024 positions,
256 steps, 256-position blocks, and no explicit token suppression. That task uses
a different prompt, assistant prefix, and code extraction, and stops at `\ndef`;
the LLaDA-specific task preserves function definitions and sanitizes extracted
code. Although the old launcher passed `--num_fewshot 5`, its saved effective
configuration was zero-shot. These are mismatched protocols, so the old score
does not measure reproduction of the 47.0% result. The contribution of each
difference requires a controlled run.

### Runtime planning

The [old full entropy-drop result](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/humaneval_full_entropy_drop/GSAI-ML__LLaDA-8B-Instruct/results_2026-08-31T19-49-03.066184.json)
records 28,843 seconds (8 h 00 m 43 s) for 164 problems. Its generation progress
finished in 7 h 58 m 13 s, about 175 seconds/problem, using one evaluation process
on an A100 node despite requesting two GPUs. Old greedy evaluation took 5,539
seconds (1 h 32 m 19 s).

Splitting identical work across two independent workers would suggest about four
hours for the old entropy policy. The new policy uses dependency candidates and
a changed prompt, block size, and generation budget, so this is only a reference.
Use 4–8 hours as a rough planning range for the three-policy HumanEval job, keep
the 24-hour allocation, and refine the estimate from W&B progress. It is not a
measured runtime prediction for the corrected configuration.

### Submit selector benchmarks

Submit either or both jobs yourself:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/full_benchmark.slurm.sh \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/benchmarks/gsm8k.json \
    first_action_entropy

sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/full_benchmark.slurm.sh \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/benchmarks/humaneval.json
```

Each job requests two GPUs, 24 CPUs, 64 GB RAM, and 24 hours. It prepares dllm,
benchmarks the selected arms, and writes a merged report. Validation tests run
only through the separate launcher below. Each arm uses two independent workers with disjoint
document assignments. No snapshot collection or diagnostic forks are required.
Resubmitting the same command resumes completed documents. Source or configuration
changes require a new run tag.

Jobs 240498 and 240499 stopped during the validation tests with
`ModuleNotFoundError: No module named 'scripts.tests'`, before model loading or
benchmark generation. The bundled lm-eval harness has its own `scripts` package;
the benchmark and replay tests now import sibling helpers through the existing
test-directory setup in
[conftest.py](/home/sarthak.malla/dllm-selection-ensemble/scripts/tests/conftest.py).
Validation now runs separately from benchmark jobs. These failed attempts wrote no
benchmark records, so the same submissions can be retried without a new run tag.
Both jobs also logged CUDA driver initialization warnings on `gpu-54`; the
launcher excludes that node pending a GPU health check on the cluster.

The commands above run only the new hybrid on GSM8K and all three policies on
HumanEval: two jobs, four new policy benchmarks, and eight W&B worker runs.
Optional policy names after the JSON path select arms for any task; omitting them
runs all three. This choice does not depend on the dataset name.

GSM8K already has completed full-test baselines (flexible exact match):

- Confidence-only: **69.83%**, from the
  [non-lookahead ablation](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/entropy_budget_full_v1/gsm8k_cot/max_confidence/seed42/results_2026-09-04T18-02-09.965457.json).
- Entropy lookahead: **71.19%**, from the
  [dependency entropy reference](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/dependency_guided/p8/reduced_full_v3/full/gsm8k_cot/dependency_entropy_budget_n4/seed42/results_2026-09-04T04-53-11.654760.json).

Both have matching recorded candidate and generation settings. Each reproduces
experiment 01's responses and flexible correctness on every one of the 100
overlapping problems. Reuse these as historical accuracy baselines. Their
execution and timing instrumentation differ, so they do not establish a controlled
latency comparison against the new hybrid. Historical samples remain alongside
the linked results for a later paired comparison; the automatic report currently
compares only policies present in the new output root. No matching completed
HumanEval pair was found in these ablation directories.

Default output roots are:

- `/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/first_action_full_gsm8k_two_gpu`
- `/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/first_action_full_humaneval_two_gpu`

An exported `TF_RUN_TAG` overrides the default; use distinct tags for different
tasks. Reports are written under each root's `analysis` directory as `report.md`,
`summary.json`, `benchmarks.csv`, `metrics.csv`, and `paired_comparisons.csv`.
The report includes task-native metrics, paired primary-accuracy comparisons,
model evaluations, and generation latency. The optional `report_split_at` setting
also reports a document prefix and remainder separately; the GSM8K configuration
uses it to separate the previously explored development prefix.

### Run validation separately

Submit the synthetic validation suite when checking implementation changes:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/validate_benchmark.slurm.sh
```

This independent compute-node job requests two CPUs, 16 GB RAM, and 20 minutes,
with no GPUs. It prepares the dllm environment and runs the five benchmark,
runner, analysis, telemetry, and import test files formerly bundled with benchmark
submission. Tests use synthetic data, a tiny CPU model, and a fake W&B client.
Benchmark submissions neither invoke this launcher nor depend on its completion.

## W&B and compact artifacts

W&B defaults to online, with one persistent run per arm and worker. Set
`WANDB_PROJECT` before submission to select the project; `WANDB_ENTITY` follows
the standard W&B environment setting. Logs include primary accuracy, all native
metric/filter scores, progress, actual model calls/evaluated rows, latency, and
aggregate action sizes and selector counts. Completed units are reused on resume.

Local artifacts retain responses, scalar per-problem results, model-call/timing
accounting, completion records, and a manifest with configuration and document
hashes. They omit replay snapshots, candidate/probe dumps, per-action traces, and
generated-token arrays. Prompts and generated responses are not uploaded to W&B.

## Adding another task or running a smoke check

Copy a task JSON configuration and change `task`, `primary_metric`, `primary_filter`,
and the desired generation settings. `num_fewshot` can be omitted to retain the
task's own value. The current benchmark supports one concrete `generate_until`
task with `repeats: 1`, mean-aggregated scalar metrics, and a binary per-problem
primary metric. Unsupported task types fail explicitly.

For a smoke check inside an existing two-GPU allocation, after preparing the
environment from the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md):

```bash
srun --ntasks=1 --gres=gpu:2 --cpus-per-task=24 --time=00:30:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/launch.py \
    --benchmark-config /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/01_selectors/benchmarks/gsm8k.json \
    --output-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/first_action_smoke \
    -- --resume --arm first_action_entropy --doc-start 0 --doc-stop 2
```

Work-range flags only limit an invocation; the manifest still identifies the full
task. Remove those flags to continue across all documents.
