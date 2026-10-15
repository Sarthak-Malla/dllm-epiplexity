# TSE experiment tracking and evaluation

The source of truth is
[`/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json`](/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json).
It records the scientific run, while W&B is the dashboard and `.logs` stores raw
artifacts. Scheduler attempts belong under the scientific run rather than becoming
separate experiments. Experiments belong to exactly one of two research groups:
`tse` for same-tokenizer TSE-only studies, or `ctca` for cross-tokenizer CTCA and
the native controls needed to evaluate it.

To view the current ledger and resolve metrics from the result JSON files:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
python /home/sarthak.malla/dllm-epiplexity/scripts/tse/track_experiments.py --check
python /home/sarthak.malla/dllm-epiplexity/scripts/tse/track_experiments.py --group ctca
```

## Current snapshot

| Experiment | Group | Status | Protocol | Flexible EM | Strict EM | Hours |
|---|---|---|---|---:|---:|---:|
| `tse-llada-probability-fusion-20261013` | TSE | completed | `tse-gsm8k-v1` | 69.22 ± 1.27 | 46.55 ± 1.37 | 6.94 |
| `dream-native-20261013` | CTCA | completed | `ctca-gsm8k-v1` | 40.56 ± 1.35 | 18.73 ± 1.07 | 1.72 |
| `ctca-llada-dream-20261014` | CTCA | completed | `ctca-gsm8k-v1` | 38.29 ± 1.34 | 36.62 ± 1.33 | 10.32 |
| `llada-native-20261014` | CTCA | running | `ctca-gsm8k-v1` | — | — | — |

TSE and CTCA are separate groups and are not ranked against one another. The TSE
group currently uses `max_new_tokens=512, steps=128`; the CTCA group uses
`max_new_tokens=384, steps=64`. The LLaDA-native ledger entry retains two failed
attempts (jobs `310867` and `312048`) so failures are not lost.

## Standard evaluation protocol

Protocol IDs: `tse-gsm8k-v1` and `ctca-gsm8k-v1`.

### Frozen benchmark setup

- Task: `gsm8k_cot`, version 3.0, full 1,319-example test set (no `--limit`).
- Prompting: five-shot and `--apply_chat_template`.
- Harness seeds: random `0`; NumPy, Torch, and few-shot `1234`.
- Arithmetic: `bfloat16`; greedy final selection (`temperature=0.0`).
- TSE-group budget: `max_new_tokens=512` and `steps=128`.
- CTCA-group budget: `max_new_tokens=384` and `steps=64`, including both native
  controls.
- LLaDA settings: `block_size=32`, `remasking=low_confidence`,
  `probability_temperature=1.0`, and the suppression settings in the native config.
- Dream settings: `alg=entropy`, `alg_temp=0.0`, `top_p=1.0`, `top_k=50`, and
  right-shifted logits.
- CTCA settings are frozen by
  [`/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k_static.conf`](/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k_static.conf),
  including the master model, projection mode, temperatures, anchors, and fusion
  weight. Any change creates a new protocol ID.

Batch size may change to fit memory because it should not change deterministic
outputs. Record it anyway. If a batch-size change does alter results, treat it as
a protocol change and investigate before comparison.

### Metrics and reporting

- Primary quality metric: `exact_match,flexible-extract`.
- Secondary quality metric: `exact_match,strict-match`.
- Always report the harness bootstrap standard error beside both metrics in papers
  or reports.
- Efficiency metrics: wall-clock duration, GPU count/type, peak memory, and total
  model forward passes. These are required before making speed or cost claims.
- Canonical wall time is the lm-eval result field
  `total_evaluation_time_seconds`. It spans evaluation-tracker construction through
  aggregated-result saving, so it includes model loading, request construction,
  generation, and metric aggregation. It does not include scheduler queue time,
  Python process startup, or post-result logging.
- Also retain Slurm elapsed time for operational accounting. Do not substitute it
  for lm-eval wall time in method comparisons. Report GPU-hours separately as
  `wall_time_hours × GPU_count`, since CTCA and native controls use different GPU
  counts.
- Deterministic runs require one successful full evaluation. Any stochastic
  setting requires at least three predeclared seeds and must report mean and
  standard deviation.
- Compare methods only when both `group` and `protocol_id` match. TSE and CTCA
  answer different research questions and should be reported in separate tables.
  Do not select the better of strict and flexible extraction after seeing results;
  flexible EM is primary.

### Run lifecycle

1. Before submission, add a ledger record with `status=planned`, a unique ID,
   `group=tse` or `group=ctca`, a frozen config path, protocol ID, model
   identifiers, and the intended result glob. Commit the config and ledger together.
2. After submission, set `status=running`, record the scheduler job ID, Git commit,
   W&B run/group, host/GPU type, and start time. Retries go in `attempts`.
3. On success, preserve the raw lm-eval JSON, set `status=completed`, and run the
   tracker with `--check`. Never hand-copy metric values into the JSON ledger; the
   tracker reads them from the raw artifact.
4. On failure, set the attempt status and concise root cause. Keep failed attempts;
   they explain configuration changes and resource usage.
5. Before publishing a comparison, verify matching protocol IDs, complete artifacts,
   the expected 1,319 examples, standard errors, and efficiency metadata.

## Naming convention

Experiment IDs must identify the characteristics that change the scientific
behavior of the run:

```text
<group>-<task>-<model>-<members>-<weighting>-<fusion>-<rank>-<YYYYMMDD>
```

Use filesystem-safe decimal values such as `t0p2`, `t0p7`, and `w0p5`.
Omit an axis only when it is fixed by the protocol and cannot distinguish two
runs in the same study. In particular, record probability fusion and commit
ranking separately: Jensen–Shannon agreement changes the position-ranking score,
not the fused token distribution.

Examples for the TSE temperature study:

```text
tse-gsm8k-llada8bi-single-t1p0-nativeconf-20260926
tse-gsm8k-llada8bi-single-t0p2-nativeconf-20260926
tse-gsm8k-llada8bi-single-t0p7-nativeconf-20260926
tse-gsm8k-llada8bi-ens-t0p2-t0p7-w0p5-probmean-jsrank-20260927
tse-gsm8k-llada8bi-ens-t0p2-t0p7-w0p5-probmean-confrank-20261013
```

The ledger must also store the corresponding characteristics as structured
fields: `study_id`, `fusion`, `weighting`, `alpha`, `probability_temperatures`,
and `commit_score`. Names make runs recognizable; structured fields make them
queryable and prevent ambiguous comparisons.

Use `<protocol-id>/<experiment-id>` as the W&B group/name prefix. Append the
scheduler job ID only to an attempt name, never to the scientific experiment ID.
A scientific experiment can have multiple scheduler attempts, but every attempt
must point to the same immutable config or explicitly start a new experiment.

## Locked CTCA parity study design (2026-10-15)

These protocols define the primary controlled comparison between native
single-model generation and cross-tokenizer fusion. They are design-locked but
are not scheduler submissions; add individual `planned` records to
[`/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json`](/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json)
only after their immutable config files exist.

### Shared controls

- Models: `GSAI-ML/LLaDA-8B-Instruct` and
  `Dream-org/Dream-v0-Instruct-7B`; record their resolved Hugging Face revisions
  with every run.
- Arithmetic: `bfloat16`.
- Decoding: deterministic (`temperature=0.0`,
  `stochastic_transfer=false`) with greedy position commitment.
- Commitment score: selected-token probability (`confrank`). LLaDA uses
  `remasking=low_confidence`; Dream uses `alg=maskgit_plus`. Dream entropy
  ranking is not part of the primary parity protocol.
- Probability calculation: full vocabulary (`top_p=1.0`, no `top_k` truncation)
  and probability temperature `1.0`.
- Harness seeds: random `0`; NumPy, Torch, and few-shot `1234`.
- Chat formatting: `--apply_chat_template` for every run.
- Evaluation batch size: `8`. A memory-driven change is permitted only if it is
  recorded and verified not to change deterministic outputs.
- Single-model controls use one A100 40 GB. CTCA uses one process on two A100
  40 GB GPUs, with LLaDA on `cuda:0`, Dream on `cuda:1`, and fusion on the
  Dream-master device, `cuda:1`. The models are not data-parallel or
  tensor-parallel.

### `ctca-gsm8k300-v2`

- Dataset: the first 300 examples in the canonical `gsm8k_cot` test order,
  selected by `--limit 300`; do not tune on these examples.
- Prompting: five-shot.
- Generation: `max_new_tokens=512`, `steps=128`, `block_size=32` (16 blocks,
  8 refinement steps per block).
- Suppression by active output tokenizer:

  | Output tokenizer | `suppress_tokens` | `begin_suppress_tokens` |
  |---|---|---|
  | LLaDA | `[]` | `[126081;126348]` |
  | Dream | `[]` | `[151643;151645]` |

  The pairs are the tokenizer-specific EOS and turn-end tokens. For CTCA, use
  the row belonging to the master because the generated canvas is in the
  master's vocabulary. The locked Dream-master run therefore uses
  `master_suppress_tokens=[]` and
  `master_begin_suppress_tokens=[151643;151645]`.
- Required experiment IDs:
  - `ctca-gsm8k300-llada8bi-single-t1p0-nativeprob-confrank-20261015`
  - `ctca-gsm8k300-dream7bi-single-t1p0-nativeprob-confrank-20261015`
  - `ctca-gsm8k300-llada8bi-dream7bi-dreammaster-ens-t1p0-t1p0-w0p5-probmean-confrank-20261015`

### `ctca-humaneval-v1`

- Dataset: all 164 HumanEval problems using the common `humaneval_instruct`
  task for every method; model-specific LLaDA and Dream task variants are not
  comparable and are excluded from this protocol.
- Prompting: zero-shot; code execution must be explicitly enabled through the
  harness safety flag.
- Generation: `max_new_tokens=768`, `steps=192`, `block_size=32` (24 blocks,
  8 refinement steps per block).
- Suppression by active output tokenizer:

  | Output tokenizer | `suppress_tokens` | `begin_suppress_tokens` |
  |---|---|---|
  | LLaDA | `[126081]` | `[]` |
  | Dream | `[151643]` | `[]` |

  HumanEval always suppresses the tokenizer-specific EOS token so a premature
  EOS prediction cannot terminate the code body. For CTCA, use the row belonging
  to the master. The locked Dream-master run therefore uses
  `master_suppress_tokens=[151643]` and
  `master_begin_suppress_tokens=[]`.
- Primary metric: deterministic `pass@1` over all 164 problems.
- Required experiment IDs:
  - `ctca-humaneval-llada8bi-single-t1p0-nativeprob-confrank-20261015`
  - `ctca-humaneval-dream7bi-single-t1p0-nativeprob-confrank-20261015`
  - `ctca-humaneval-llada8bi-dream7bi-dreammaster-ens-t1p0-t1p0-w0p5-probmean-confrank-20261015`

### CTCA-only settings

- LLaDA is model A and the auxiliary; Dream is model B and the master
  (`master_model=b`). The generated canvas therefore uses Dream token IDs.
- Fusion: static arithmetic probability mean, `alpha=0.5`, where `alpha` is the
  model-A/LLaDA weight.
- Logit probability temperatures: `temperature_a=1.0` and
  `temperature_b=1.0`; `weight_temperature=1.0`.
- Projection: `sparse_topk`, `ctca_projection_top_k=64`, all deterministic
  shared anchors (`ctca_num_anchors=auto`, `ctca_min_anchors=128`),
  `ctca_anchor_temperature=0.01`, `ctca_projection_temperature=0.05`, and
  `ctca_chunk_size=512`.
- Other settings: `selection_mode=tse`, `weighting_mode=static`,
  `normalize_entropy=true`, `epsilon=1e-9`, `capture_logits=false`, and
  `ctca_canvas_workers=16`.
- Apply the task-specific Dream suppression list on the master grid. HumanEval
  uses always-on `master_suppress_tokens=[151643]`; both always-on and
  begin-only master suppression are implemented and covered by focused tests.

The suppression IDs above are semantic equivalents, not interchangeable IDs:
LLaDA EOS/turn-end/mask are `126081`/`126348`/`126336`, while Dream
EOS/turn-end/mask are `151643`/`151645`/`151666`. Never pass one tokenizer's IDs
to the other tokenizer's canvas. The native Dream and LLaDA controls, as well
as CTCA, expose the same `suppress_tokens` and `begin_suppress_tokens` behavior
used by this protocol.

### Four-job execution layout

- LLaDA controls: GSM8K-300 followed by HumanEval in
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/llada_parity_suite_slurm.sh`.
- Dream controls: GSM8K-300 followed by HumanEval in
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/dream_parity_suite_slurm.sh`.
- Dream-master CTCA GSM8K-300 in
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_parity_slurm.sh`.
- Dream-master CTCA HumanEval in
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_humaneval_parity_slurm.sh`.

The three GSM8K-300 runs share W&B group `ctca-gsm8k300-v2`; the three
HumanEval runs share `ctca-humaneval-v1`. Each scientific run has its own
response-cache directory and retains lm-eval's per-sample JSONL output.

Before the full HumanEval jobs, the operational smoke test in
`/home/sarthak.malla/dllm-epiplexity/scripts/tse/humaneval_smoke_slurm.sh`
runs one problem through Dream in a single-GPU allocation. It uses
`max_new_tokens=128`, `steps=32`, and `batch_size=1`; therefore its quality
score is not comparable to the locked protocol. Smoke outputs and W&B runs are
isolated under group `smoke-humaneval-v1` and are not scientific ledger
entries. Separate LLaDA and CTCA smoke configurations remain available for
targeted diagnostics but are not invoked by this launcher.

Every runnable shell launcher under `/home/sarthak.malla/dllm-epiplexity/scripts/tse`
prints its runner hash, Git state, Slurm allocation, embedded launch settings,
and complete external configuration file before model loading. GPU inference is
started inside a Slurm job step. The parity launchers additionally run
`/home/sarthak.malla/dllm-epiplexity/scripts/tse/cuda_preflight.py` before model
loading so an unavailable or incomplete GPU allocation fails immediately with
diagnostic device information.

Model-recommended native references are separate from these protocols. In
particular, Dream `alg=entropy`, LLaDA `512/512`, and Dream `768/768` must not be
reported as parity controls. Higher-step TSE-only experiments also require a
separate protocol ID and will be specified independently.
