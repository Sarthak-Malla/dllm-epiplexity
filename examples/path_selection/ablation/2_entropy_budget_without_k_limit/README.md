# Ablation 2: Entropy Budget Without a Four-Token Cap

This experiment compares two entropy budgets at two maximum action sizes:

- entropy budget `1.0`, maximum action size `8`;
- entropy budget `2.0`, maximum action size `8`;
- entropy budget `1.0`, maximum action size `64`;
- entropy budget `2.0`, maximum action size `64`.

A follow-up full GSM8K job tests entropy budget `4.0` with maximum action size
`64`, completing the log-spaced budget sequence 1.0, 2.0, and 4.0 in the
uncapped setting. It writes to the fresh run tag
`vectorized_soft_full_budget4_v1` by default.

The completed Phase-8 reference used entropy budget `2.0` and maximum action
size `4`. Therefore, the `2.0` job isolates removal of the hard cap, while the
`1.0` job tests whether a stricter entropy bound is safer when actions can grow.

Because decoding is blockwise with `block_size=64`, setting the maximum action
size to 64 removes the separate `k=4` restriction. A candidate now stops when
adding the next token would exceed the entropy budget, or when no masked token
remains in the current block.

The completed reference diagnostics contain 110,101 selected actions. Of these,
66.31% reached size four because of the cap. For those capped actions, cumulative
entropy was 0.0037 at the median, 0.709 at p90, and 1.746 at p99. This shows that
the four-token cap often stopped growth well before the entropy budget of 2.0.

These diagnostics cannot reproduce the uncapped result exactly: they do not
retain the per-position logits, dependency matrices, or the candidate ordering
beyond the fourth token. The earlier budget calibration used only eight GSM8K
examples, so its accuracy tie across budgets 0.5, 1.0, and 2.0 is not enough to
retune the budget reliably.

Submit all four full GSM8K jobs with a fresh run tag:

```bash
sbatch --job-name=abl2-ent1-k8 --export=ALL,ABLATION2_RUN_TAG=vectorized_soft_full_v1,ABLATION2_MAXIMUM_ACTION_SIZE=8 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget1_uncapped.slurm.sh
sbatch --job-name=abl2-ent2-k8 --export=ALL,ABLATION2_RUN_TAG=vectorized_soft_full_v1,ABLATION2_MAXIMUM_ACTION_SIZE=8 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget2_uncapped.slurm.sh
sbatch --job-name=abl2-ent1-uncap --export=ALL,ABLATION2_RUN_TAG=vectorized_soft_full_v1,ABLATION2_MAXIMUM_ACTION_SIZE=64 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget1_uncapped.slurm.sh
sbatch --job-name=abl2-ent2-uncap --export=ALL,ABLATION2_RUN_TAG=vectorized_soft_full_v1,ABLATION2_MAXIMUM_ACTION_SIZE=64 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget2_uncapped.slurm.sh
```

Submit the budget-4 follow-up separately:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget4_uncapped.slurm.sh
```

## Maximum confidence without lookahead: entropy budget 2.0, N=4 or N=8

Set `PATH_ABLATION_SAMPLER_TYPE=max_confidence` to use
`sampler_type=max_confidence`. This reuses the dependency sampler with
`dependency_candidate_selector=max_confidence`: select the candidate group
with the highest mean probability of its proposed tokens from the current
base pass. At temperature zero these are the maximum token probabilities.
No candidate lookahead forward is performed. N still controls the number of
candidate groups, while `candidate_chunk_size` is unused.

The following runs use the corrected IE seed policy, entropy budget `2.0`,
maximum action size and block size `64`, generation seed `42`, temperature
`0.0`, full five-shot GSM8K, and the same LLaDA-8B-Instruct checkpoint. Companion
scoring retains outgoing attention and confidence exponent `0.0`.

```bash
source /home/sarthak.malla/.zshrc
conda activate /home/sarthak.malla/.conda/envs/dllm

# N=4, corrected IE seed, maximum confidence, no lookahead.
sbatch --job-name=abl2-ie-maxconf-n4 \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=max_confidence,PATH_ABLATION_CANDIDATE_BUDGET=4,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,ABLATION2_MAXIMUM_ACTION_SIZE=64,ABLATION2_RUN_TAG=max_confidence_ie_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget2_uncapped.slurm.sh

# N=8, corrected IE seed, maximum confidence, no lookahead.
sbatch --job-name=abl3-ie-maxconf-n8 \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=max_confidence,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,ABLATION3_RUN_TAG=max_confidence_ie_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/3_candidate_budget/run_gsm8k_candidate_budget8.slurm.sh
```

Each job requests two GPUs for data-parallel evaluation. Results and caches are
isolated from entropy-lookahead runs:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/max_confidence_ie_v1/gsm8k_cot/max_confidence/seed_incoming_entropy1.0/entropy_budget2.0/candidates4/seed42
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/3_candidate_budget/max_confidence_ie_v1/gsm8k_cot/max_confidence/seed_incoming_entropy1.0/entropy_budget2.0/candidates8/seed42
```

The launcher records `selector=max_confidence` and `lookahead=false` in W&B.
The original no-diagnostics launches set `diagnostic_metadata=false` and
`diagnostic_retention=none`. Evaluation accuracy,
selected candidate names, generation timing, and peak GPU memory are still
saved. There are no per-step lookahead diagnostics. Artifacts and the completion
marker use the `max_confidence` suffix.

## N4 rerun with per-step measurements and calls per example

The dedicated launcher reproduces the algorithm settings of job `245619`: IE
seeds, N4, temperature zero, entropy budget 2, action/block size 64, companion
confidence exponent zero, seed 42, and full five-shot GSM8K on two GPU ranks.
It enables `diagnostic_metadata=true` and retains all lightweight candidate
records with `diagnostic_retention=full`. It performs no lookahead forwards.
Logging adds transfers and timing synchronization, so its wall time includes
diagnostic overhead. Prediction parity tests are provided below but have not
been executed here.

Run the focused checks on a compute node first:

```bash
source /home/sarthak.malla/.zshrc
conda activate /home/sarthak.malla/.conda/envs/dllm
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:${PYTHONPATH:-}
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 \
  python -m pytest \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_non_lookahead.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_decoding_summary.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_ablation_launchers.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_reduced_full_launchers.py -q
```

Submit an optional eight-example GPU check, then the full run:

```bash
sbatch --export=ALL,PATH_ABLATION_LIMIT=8 /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_ie_maxconf_n4_diagnostics.slurm.sh
sbatch --export=ALL,PATH_ABLATION_LIMIT= /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_ie_maxconf_n4_diagnostics.slurm.sh
```

Neither command has been submitted here. Full results use this fresh directory:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/max_confidence_ie_diagnostics_v1/gsm8k_cot/max_confidence/diagnostics/seed_incoming_entropy1.0/entropy_budget2.0/candidates4/seed42
```

The limited check gets a separate `limit8` component. Response caching is
disabled for diagnostic runs: cached answers bypass the sampler and would lose
the trajectories. The completion marker still prevents accidental reruns of a
finished run. Use a fresh `ABLATION2_RUN_TAG` for an additional full experiment.

Saved files in that directory:

- `results.json_max_confidence_diagnostics_rank*.json`: chronological steps,
  with every candidate's size, positions, proposed token IDs, seed identity and
  rank, seed confidence/entropy, mean/min confidence, summed/mean entropy,
  conflict/support scores, stopping reason, singleton-refill flag, and winner.
  Each step records base calls, zero lookahead calls, and global/block indices.
- `results.json_max_confidence_decoding_rank*.json`: per-example call totals,
  action-size histograms, mean group confidence/entropy, singleton counts,
  refill selections, and selected-candidate counts. Records carry task/document
  IDs, rank, request index, and prompt SHA256 for alignment with evaluation.
- `results.json_max_confidence_runtime.json`: aggregate
  `decoding_summary.mean_model_calls_per_example`, base/lookahead averages,
  counts of measured/unique/duplicate requests, and action-size histogram.
  Distributed padding is deduplicated using task/document/request/prompt
  identity. Total work including repeated examples is reported separately.
- `results.json_max_confidence_decoding_manifest.json`: paths to both summary
  shards; diagnostic shards have their own manifest.
- `source_hashes.sha256`: sampler/evaluator source hashes at job launch.

Calls include generation of the complete 256-position canvas, before answer
stop-string trimming. At batch size one and CFG zero, each recorded base call
is an actual model invocation; candidate construction and logging add none.
Evaluation batches are not decoding steps. For other batch sizes the per-example
counts represent participation in shared forwards, not additive invocation counts.
No logits, vocabulary distributions, Q/K tensors, or attention matrices are saved.

Read the aggregate call count after completion:

```bash
jq '.decoding_summary' /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/max_confidence_ie_diagnostics_v1/gsm8k_cot/max_confidence/diagnostics/seed_incoming_entropy1.0/entropy_budget2.0/candidates4/seed42/results.json_max_confidence_runtime.json
```

These commands have not been executed here. The focused validation command
below also covers the maximum-confidence alias and no-lookahead execution.

## Parallel candidate lookahead: entropy budget 2.0, N=4 or N=8

The existing decoder supports this through `candidate_chunk_size`. Set it to
the candidate count to evaluate all candidate actions in one batched lookahead
forward per decoding step. The shared launcher exposes it as
`PATH_ABLATION_CANDIDATE_CHUNK_SIZE`, defaulting to `1` (sequential).
Setting `dependency_parallel_variant` is unrelated to this batching setting.

The following runs use the corrected **IE** seed policy: confidence times
entropy-weighted incoming attention, multiplied by `exp(-own_entropy)`.
Set `PATH_ABLATION_SEED_STRATEGY=incoming` and
`PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0` to select it. The I policy uses the same
strategy with weight `0.0`. IE scored 214/300 and I scored 213/300 in the
[fixed-size seed ablation](/home/sarthak.malla/dllm-selection-ensemble/research_ideas/approaches/corrected_seed_results_2026-09-11.md);
that one-question difference did not establish a reliable advantage.

These runs retain the LLaDA-8B-Instruct checkpoint, entropy budget `2.0`, maximum
action size and block size `64`, temperature `0.0`, generation seed `42`, full
five-shot GSM8K evaluation, and the original two GPU ranks. Companion scoring
keeps outgoing attention and confidence exponent `0.0` from the uncapped
reference. Corrected incoming seed scores always include confidence with
exponent `1.0` and exclude the anchor-support boost. Companion anchor support,
conflict penalties, refill ranking, entropy stopping, and verifier scoring
retain their existing rules.

```bash
source /home/sarthak.malla/.zshrc
conda activate /home/sarthak.malla/.conda/envs/dllm

# N=4: IE seed, four candidates in one verifier batch.
sbatch --job-name=abl2-ie-ent2-n4-parallel \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=entropy_drop,PATH_ABLATION_CANDIDATE_BUDGET=4,PATH_ABLATION_CANDIDATE_CHUNK_SIZE=4,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,ABLATION2_MAXIMUM_ACTION_SIZE=64,ABLATION2_RUN_TAG=parallel_ie_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget2_uncapped.slurm.sh

# N=8: IE seed, eight candidates in one verifier batch.
sbatch --job-name=abl3-ie-ent2-n8-parallel \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=entropy_drop,PATH_ABLATION_CANDIDATE_CHUNK_SIZE=8,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,ABLATION3_RUN_TAG=parallel_ie_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/3_candidate_budget/run_gsm8k_candidate_budget8.slurm.sh
```

Results and response caches are separated by seed policy, own-entropy weight,
candidate count, and chunk size:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/parallel_ie_v1/gsm8k_cot/seed_incoming_entropy1.0/entropy_budget2.0/candidates4/chunk4/seed42
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/3_candidate_budget/parallel_ie_v1/gsm8k_cot/seed_incoming_entropy1.0/entropy_budget2.0/candidates8/chunk8/seed42
```

Seed controls and chunk size appear in the launch log, W&B config, and run names.
Compare accuracy, generation time, lookahead model-call count, and peak GPU
memory against the sequential references. Batching uses more GPU memory, and
bfloat16 rounding can change the selected candidate. If needed, chunk size `2`
or `4` allows partial batching. Leaving seed controls and chunk size unset
retains legacy seed selection, sequential evaluation, and original output paths.
To isolate the effect of batching under IE, run a comparison with the same IE
settings and `PATH_ABLATION_CANDIDATE_CHUNK_SIZE=1`.

These commands submit the existing ablation jobs; they have not been run here.

### FP32 entropy in small chunks

The parallel retries `245636` and `245637` exhausted GPU memory after the
lookahead forward, while creating full-size FP32 entropy intermediates. The
[entropy implementation](/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/counterfactual.py)
now computes and validates at most 64 token distributions at a time. Every
distribution still covers the full vocabulary, and entropy and score arithmetic
remain FP32. Candidate model forwards remain batched with
`candidate_chunk_size=N`; entropy chunks do not add model forwards.

At vocabulary size 126,464, each 64-row FP32 intermediate occupies about
30.9 MiB. Several intermediates coexist, and the model output logits still
remain resident. This change bounds the entropy workspace during inference;
GPU validation is still needed to establish that the full jobs fit.

Run the focused numerical and candidate-selection checks on a compute node:

```bash
source /home/sarthak.malla/.zshrc
conda activate /home/sarthak.malla/.conda/envs/dllm
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:${PYTHONPATH:-}
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --time=00:15:00 \
  python -m pytest \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_counterfactual_oracle.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_batched_lookahead.py -q
```

Rerun the two full IE experiments with fresh result/cache directories:

```bash
# N=4, one lookahead forward with four candidate sequences.
sbatch --job-name=abl2-ie-ent2-n4-chunked --exclude=gpu-05,gpu-51,gpu-54 \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=entropy_drop,PATH_ABLATION_CANDIDATE_BUDGET=4,PATH_ABLATION_CANDIDATE_CHUNK_SIZE=4,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,PATH_ABLATION_LIMIT=,ABLATION2_MAXIMUM_ACTION_SIZE=64,ABLATION2_RUN_TAG=parallel_ie_entropy_chunks_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/run_gsm8k_entropy_budget2_uncapped.slurm.sh

# N=8, one lookahead forward with eight candidate sequences.
sbatch --job-name=abl3-ie-ent2-n8-chunked --exclude=gpu-05,gpu-51,gpu-54 \
  --export=ALL,PATH_ABLATION_SAMPLER_TYPE=entropy_drop,PATH_ABLATION_CANDIDATE_CHUNK_SIZE=8,PATH_ABLATION_SEED_STRATEGY=incoming,PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0,PATH_ABLATION_LIMIT=,ABLATION3_RUN_TAG=parallel_ie_entropy_chunks_v1 \
  /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/3_candidate_budget/run_gsm8k_candidate_budget8.slurm.sh
```

For an eight-example GPU check first, use the same commands with
`PATH_ABLATION_LIMIT=8` and `--time=01:00:00`. Limited runs use a separate
`limit8` directory; inspect their completion markers and peak GPU memory before
the full evaluation. Passing this check does not guarantee every longer prompt
in the full dataset fits. Neither the tests nor these jobs were executed here.

### Validate corrected seeds on a compute node

The focused tests cover fixed/adaptive seed parity, incoming direction and own
entropy, unchanged companion/refill rules, entropy stopping, legacy behavior,
sampler configuration propagation, and launcher output/cache separation.
Run them yourself with:

```bash
source /home/sarthak.malla/.zshrc
conda activate /home/sarthak.malla/.conda/envs/dllm
export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble:${PYTHONPATH:-}
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --cpus-per-task=2 --time=00:30:00 \
  python -m pytest \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_adaptive_cardinality.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_parallel_candidates.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_ablation_launchers.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_non_lookahead.py \
  /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_reduced_full_launchers.py -q
```
