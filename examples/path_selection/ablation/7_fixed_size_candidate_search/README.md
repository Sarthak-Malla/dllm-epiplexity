# Ablation 7: Position selection at fixed commitment size

This experiment asks whether additional candidate position sets help when every
candidate commits four tokens, and whether dependency-based construction improves
over confidence-based construction at the same candidate budget. It uses the
existing sampler without changing its implementation.

| Arm | Existing construction variant | Candidate budget | Confidence exponent | Tokens per action |
|---|---|---:|---:|---:|
| D4 | `soft_full` | 4 | 0.0 | 4 |
| D8 | `soft_full` | 8 | 0.0 | 4 |
| C4 | `top_confidence` | 4 | 0.0 | 4 |
| C8 | `top_confidence` | 8 | 0.0 | 4 |
| CD4 | `soft_full` | 4 | 1.0 | 4 |
| CD8 | `soft_full` | 8 | 1.0 | 4 |

CD4/CD8 are follow-up arms: each differs from D4/D8 only in
`dependency_confidence_exponent=1.0`. They test whether direct confidence
weighting improves the dependency proposals. The base utility becomes
`confidence[i] * sum_j attention[i,j] * entropy[j]`, affecting both seed and
companion ranking. Committed-anchor support is added afterward; its formula and
the conflict penalty are unchanged. This encourages confident positions but does
not impose an entropy bound or match the confidence control's group uncertainty.
No sampler implementation change is needed.

The corrected-seed follow-up below adds I/IE/CS arms using separate seed scores;
it does change the sampler implementation behind an opt-in setting.

All arms evaluate the same **300 GSM8K test documents (IDs 0–299)**, using the
dataset's existing order. This is the ablation dataset, with no separate pilot.
All arms use the same pinned LLaDA-8B-Instruct checkpoint, five-shot
chat prompts, 256 generated tokens, 64-token blocks, 64 decoding steps, token
temperature zero, generation seed 42, and entropy-drop verification divided by
action size. Candidates are verified sequentially with chunk size one. Full
diagnostics are retained. Response caching is disabled so every evaluated document
has a complete decoding trace.

Each arm uses **two GPUs**, with one model replica and one evaluation process per
GPU. The harness distributes the subset across them: rank 0 evaluates the 150 even
document IDs, and rank 1 evaluates the 150 odd IDs. Batch size remains one per GPU;
the candidate budget and decoding procedure for each document stay the same.

The confidence control uses confidence for seed and companion ranking. Its first
seed is deterministic, later seeds use the existing position Gumbel ordering, and
companions are selected deterministically given the seed. This is not a sampler
that independently perturbs every position when constructing each whole set.
Attention capture still runs in both variants, allowing comparable implementation
overhead; C4/C8 are not optimized confidence-only latency baselines.

**Entropy-budget stopping is disabled in this fixed-size experiment.** The recorded
`dependency_entropy_budget=2.0` field is inactive. These results should not be
treated as a direct explanation of the existing adaptive entropy-budget policy.
Equal action sizes also do not guarantee equal entropy or confidence. The analysis
reports entropy costs so differences in uncertainty remain visible.

There is no additional singleton arm. The existing fixed-size deduplication refill
also returns four-token sets. At the end of each block, only one four-token set
remains, so the realized candidate budget falls to one in both N4 and N8 runs.
Expected model evaluations per document are:

| Candidate budget | Base forwards | Verifier forwards | Total forwards |
|---|---:|---:|---:|
| 4 | 64 | 244 | 308 |
| 8 | 64 | 484 | 548 |

## Files

- [Slurm array](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_fixed_size_candidate_search.slurm.sh): defaults to the original four arms; indices 4/5 select CD4/CD8 on the same subset.
- [Launcher](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_ablation.py): starts two Accelerate workers per arm; writes a manifest and a completion marker.
- [Analysis](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py): validates both ranks from the requested runs and writes paired comparisons and candidate-pool diagnostics.
- [Focused checks](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/test_ablation.py): synthetic artifact and launcher checks without model inference.

## Submit the ablation

Submit the array yourself:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_fixed_size_candidate_search.slurm.sh
```

The array maps `0=D4`, `1=D8`, `2=C4`, `3=C8`, `4=CD4`, `5=CD8`.
The default array still submits only indices 0–3. Each task requests two GPUs,
24 CPUs, 64 GB host memory, and three hours. At most two arms run concurrently
(`--array=0-3%2`), using **four GPUs total**. It uses partition `cscc-gpu-p`, QoS `cscc-gpu-qos`, and excludes
`gpu-51`, matching the existing ablation scripts. Allocation settings can be
overridden with `sbatch` options. The internal `srun` inherits the partition/QoS.
Keep `--ntasks=1`: that task runs one launcher, which starts two evaluation workers
with `accelerate launch --multi_gpu --num_processes 2`. Each arm selects an
available communication port independently, so concurrent arms can share a node.

The script sources `/home/sarthak.malla/.zshrc` when present, otherwise
`/apps/local/conda_init.sh`, then activates `/home/sarthak.malla/.conda/envs/dllm`.
No manual environment activation is needed before `sbatch`.

Logs are written to the existing directory:

```text
/home/sarthak.malla/dllm-selection-ensemble/.logs/abl7-fixed-k4_<array-job-id>_<task-id>.out
/home/sarthak.malla/dllm-selection-ensemble/.logs/abl7-fixed-k4_<array-job-id>_<task-id>.err
```

Outputs are separated by arm below:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v1/gsm8k_cot/limit300/seed42
```

The launcher refuses to overwrite an existing arm directory, including incomplete
runs. Use a fresh `ABLATION7_RUN_TAG` consistently across all four arms when repeating an
experiment. It records source hashes, model arguments, and the exact evaluator
command in each `manifest.json`. A `completed.json` marker is written only after
both workers exit successfully and both ranks' diagnostics/runtime files and the
aggregate runtime/diagnostics manifest are present. The
analysis performs the deeper content/comparability validation.

Optional exported controls are `ABLATION7_RUN_TAG` (default `fixed_k_v1`),
`ABLATION7_GENERATION_SEED` (default `42`), and `ABLATION7_WANDB_MODE` (default
`disabled`). Set these before submission to apply them to every arm. The array
always passes `--limit 300`; direct launcher use also defaults to 300 documents.
Other direct-launcher limits must be even, so both ranks receive equal numbers of
documents without repeated padding requests.
One seed provides an exploratory comparison, not an estimate of variation across
generation seeds. Compare accuracy within this subset; full-dataset results have
a different evaluation scope.

## Submit CD4 and CD8 separately

These two commands submit only the new arms, with two GPUs and four hours each.
They exclude `gpu-05` and `gpu-51`, which encountered failures in earlier attempts.
Both use the fresh run tag `fixed_k_conf1_v1`:

```bash
sbatch --array=4 --time=04:00:00 --exclude=gpu-05,gpu-51 \
    --export=ALL,ABLATION7_RUN_TAG=fixed_k_conf1_v1,ABLATION7_GENERATION_SEED=42 \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_fixed_size_candidate_search.slurm.sh

sbatch --array=5 --time=04:00:00 --exclude=gpu-05,gpu-51 \
    --export=ALL,ABLATION7_RUN_TAG=fixed_k_conf1_v1,ABLATION7_GENERATION_SEED=42 \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_fixed_size_candidate_search.slurm.sh
```

Outputs go to the `CD4` and `CD8` directories below:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_conf1_v1/gsm8k_cot/limit300/seed42
```

Compare CD4 against D4, CD8 against D8, and each CD arm against the corresponding
confidence control. CD8 versus CD4 tests whether confidence weighting changes the
benefit of additional candidates. Inspect group entropy and conflict alongside
paired accuracy; an improvement over D alone does not establish added value over C.

## Analyze

After all four arms finish:

```bash
if [ -f /home/sarthak.malla/.zshrc ]; then
    source /home/sarthak.malla/.zshrc
else
    source /apps/local/conda_init.sh
fi
conda activate /home/sarthak.malla/.conda/envs/dllm

srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py --run-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v1/gsm8k_cot/limit300/seed42
```

Use your normal `PARTITION` and `QUOTATYPE` values for this compute-node command.
If you override the run tag or generation seed, update the analysis path accordingly.
The default analysis directory is:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v1/gsm8k_cot/limit300/seed42/analysis
```

It contains `report.md`, `summary.json`, `run_metrics.csv`, and
`paired_comparisons.csv`. The analysis requires matching documents, prompts, and
configuration across arms except for the arm-defined construction variant,
candidate budget, and confidence exponent.
It checks complete four-token commitments, candidate uniqueness and feasible
counts, and retained full diagnostics. Both ranks must be present, and all requested
arms must use the same number of GPUs. The diagnostic shard manifest identifies
the files ending in `_rank00000-of-00002.json` and `_rank00001-of-00002.json`.
The analyzer maps each rank's local example index back to the shared document IDs.
It reports:

- Strict and flexible accuracy, paired wins/losses, and exact McNemar tests.
- Actual base and verifier forward counts across both ranks, generation time, and action sizes.
- Pairwise Jaccard overlap and changed positions within each candidate pool.
- Candidate and selected entropy costs, conflicts, and refill sources.

The per-document forward counts remain 308 for N4 and 548 for N8. Timing records
include each rank's summed generation time, the maximum of those rank totals,
their sum as generation GPU-time, and launcher wall time. Rank generation timers
exclude model loading and waits between batches; launcher wall time includes
loading, evaluation, and saving results.

The primary comparisons are D8 versus D4, C8 versus C4, D4 versus C4, and D8
versus C8. Generation trajectories can diverge between arms. Document-level
accuracy comparisons are paired; intermediate states from separate runs are not
assumed identical. A fixed-size benefit supports position search in this setting;
an absent benefit does not prove that singleton choices explain the adaptive
policy's behavior. The p-values are exploratory and unadjusted for the four
original comparisons and two answer-extraction filters; follow-up comparisons are
also exploratory.

### Compare the follow-up against existing runs

After CD4/CD8 finish, use the same environment preparation above and run this on
a compute node. The explicit directory overrides reuse D4/D8 from `fixed_k_v2`
and C4 from `fixed_k_v1`, without copying or changing their artifacts. C8 is omitted
until its result is complete.

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py \
    --run-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_conf1_v1/gsm8k_cot/limit300/seed42 \
    --arms D4 D8 C4 CD4 CD8 \
    --arm-dir D4=/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v2/gsm8k_cot/limit300/seed42/D4 \
    --arm-dir D8=/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v2/gsm8k_cot/limit300/seed42/D8 \
    --arm-dir C4=/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_v1/gsm8k_cot/limit300/seed42/C4 \
    --allow-launcher-hash-change
```

The final flag permits the recorded hash of `run_ablation.py` to differ because
the launcher gained CD4/CD8 definitions. All other recorded source hashes must
still match. The analyzer records this exception and continues to check exact
arm settings, shared evaluation settings, and document/prompt identity. Without
the flag, differing launcher hashes are rejected. To include a completed C8 run,
add `C8` to `--arms` and provide its absolute directory with `--arm-dir C8=...`.

## Optional synthetic checks

These checks use synthetic artifacts and do not run a dataset pilot. After the
same environment preparation above, run them on a compute node:

```bash
srun -p "$PARTITION" --quotatype="$QUOTATYPE" --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 python -m unittest discover -s /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search -p test_ablation.py -v
```

No jobs, Python scripts, or tests were executed when updating this setup.

## Corrected-seed follow-up

This experiment varies only seed selection while keeping CD's companion
construction. Let `A[i,j]` mean position i reads position j, `c[i]` be its maximum
token probability, and `H[i]` be its own Shannon entropy in nats. The sum below
includes eligible masked positions j other than i.

| Array index | Arm | Candidates | Seed score |
|---:|---|---:|---|
| 0 | I4 | 4 | `c[i] * sum_j A[j,i] * H[j]` |
| 1 | I8 | 8 | `c[i] * sum_j A[j,i] * H[j]` |
| 2 | IE4 | 4 | `c[i] * exp(-H[i]) * sum_j A[j,i] * H[j]` |
| 3 | IE8 | 8 | `c[i] * exp(-H[i]) * sum_j A[j,i] * H[j]` |
| 4 | CS4 | 4 | `c[i]` |
| 5 | CS8 | 8 | `c[i]` |

The own-entropy coefficient is **tau=1.0**, as agreed. I means incoming attention;
IE adds own entropy; CS means confidence-only **seed**, not confidence-only
companions. All three seed policies exclude the committed-anchor boost from the
seed score. Their companions retain outgoing attention, confidence exponent 1,
committed-anchor support, and the symmetric uncertainty-weighted conflict penalty.
All still commit four tokens and use entropy-drop lookahead. The same first 300
GSM8K questions, seed 42, 256 output tokens, 64-token blocks, and two GPUs per arm
are retained. N4/N8 cost 308/548 model forwards per question.

Implementation controls:

- `dependency_seed_strategy=incoming|confidence`; default `legacy` preserves the
  original seed behavior, including its anchor support.
- `dependency_seed_entropy_weight=0.0|1.0`; nonzero is allowed only with incoming
  seeds. Custom seed policies are supported for fixed dependency groups with k>1
  and for entropy-budget or marginal-utility stopping. Other decoding paths
  reject them rather than silently ignoring the setting. The
  [uncapped entropy-budget rerun](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/README.md#parallel-candidate-lookahead-entropy-budget-20-n4-or-n8)
  uses the same seed-scoring implementation while retaining its own companion
  settings.
- The first seed is deterministic; later seeds use the existing Gumbel ordering
  at position temperature 1.0. Score scale changes its effective randomness, so
  these runs test the full seed policy, not only the first seed's ranking.
- Companion scores and combination-refill ranking remain on the legacy CD rule.
  Different seeds can still change duplicates and which refill sets are needed.
- Diagnostics record both seed settings and each candidate's absolute seed
  position. For a refill, that position is a bookkeeping anchor, not a newly
  sampled seed. The analyzer verifies seed-setting propagation at every step.

Submit the six requested arms:

```bash
sbatch --export=ALL,ABLATION7_RUN_TAG=fixed_k_seed_v1,ABLATION7_GENERATION_SEED=42 \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_corrected_seed.slurm.sh
```

The array allows two concurrent arms: **two GPUs per arm, four GPUs total**.
Each arm gets four hours. For separate submissions, add `--array=0` through
`--array=5` to the same command. Logs use `abl7-seed_<job>_<index>` in
[/home/sarthak.malla/dllm-selection-ensemble/.logs](/home/sarthak.malla/dllm-selection-ensemble/.logs).

After completion, prepare the environment as above and run analysis on a compute node:

```bash
srun -p cscc-gpu-p -q cscc-gpu-qos --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py \
    --run-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_seed_v1/gsm8k_cot/limit300/seed42 \
    --arms I4 I8 IE4 IE8 CS4 CS8
```

Primary comparisons: I vs CS measures the seed attention factor; IE vs I measures
the own-entropy factor; IE vs CS measures their combined contribution. Compare
each at the same candidate count, then examine N8 vs N4 within each policy.
Inspect paired accuracy, first-group entropy, conflict, refills, and diversity.
An incoming-attention advantage would support a useful seed proxy; it would not
establish causal influence or conditional independence.

The core sampler sources changed for this feature. Earlier CD results remain
historical context; the analyzer deliberately rejects mixing their old sampler
hashes with these runs, even with `--allow-launcher-hash-change`. For a direct
legacy-seed comparison under the same code, optional array indices 6/7 rerun
CD4/CD8 under this tag. They are **not submitted by default**. Add those arms to
`--arms` when analyzing their completed reruns.

### Validate the seed change on a compute node

After the environment preparation above, these checks use synthetic tensors and
a tiny CPU model, not a dataset pilot:

```bash
srun -p cscc-gpu-p -q cscc-gpu-qos --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 \
    python -m pytest \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_parallel_candidates.py \
    /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/test_ablation.py -q
```

The checks cover seed direction and own entropy, unchanged companion inputs and
refill ordering, legacy defaults, decoder wiring, and analysis comparability.
They have been added but not executed in this session.

## Exact token commitment comparison

Diagnostics now save `token_ids` aligned one-to-one with `positions` for every
candidate. `selected_candidate` carries the same fields for the executed group.
These values come directly from the base predictions used by lookahead and the
subsequent commit; the logging adds no model forward or selection change.

The completed `fixed_k_seed_v1` runs saved positions but not per-step token IDs.
Their final decoded responses omit special tokens and are cut at evaluation stop
strings, so retokenizing those responses cannot recover the original full trace
reliably. The earlier 290/300 I4–IE4 statistic concerns identical first proposed
**position sets**; actual first committed position sets matched on 253/300.
Neither count directly verifies token-value equality.

To record exact IDs for the I4/IE4 comparison on the same 300 questions:

```bash
sbatch --array=0,2%2 \
    --export=ALL,ABLATION7_RUN_TAG=fixed_k_seed_tokens_v1,ABLATION7_GENERATION_SEED=42 \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/run_gsm8k_corrected_seed.slurm.sh
```

After both complete, prepare the environment as above and analyze on a compute node:

```bash
srun -p cscc-gpu-p -q cscc-gpu-qos --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/7_fixed_size_candidate_search/analyze_results.py \
    --run-root /home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/7_fixed_size_candidate_search/fixed_k_seed_tokens_v1/gsm8k_cot/limit300/seed42 \
    --arms I4 IE4
```

`report.md`, `summary.json`, and `token_comparisons.csv` distinguish first proposed
position equality, first proposed position/token equality, first committed
position/token equality, full step-by-step commitment equality, and final response
ID equality independent of commit order. `summary.json` also includes per-document
flags and the first divergent commitment step. The final comparison includes all
256 generated positions, including special tokens and text beyond stop strings.
Historical logs return `available=false`; partially recorded or inconsistent token
traces are rejected. A rerun measures the new recorded execution, not recovered
historical token IDs. No rerun or Python tests were executed in this session.
