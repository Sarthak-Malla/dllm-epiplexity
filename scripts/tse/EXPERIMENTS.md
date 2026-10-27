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

Use filesystem-safe decimal values such as `st0p0`, `pt0p2`, and `w0p5`.
For new IDs, `st` always means token sampling temperature (greedy at `st0p0`),
and `pt` always means the temperature applied to logits before softmax for
confidence, entropy, or fusion. In two-model runs, use `pta` and `ptb` for
models A and B; for example, `pta0p2-ptb0p7`. Dream's native `temperature`
controls both token sampling and its scoring softmax, so a native Dream run at
0.1 is named `st0p1-pt0p1`. Native LLaDA exposes separate `temperature` and
`probability_temperature` settings. CTCA's `temperature_a` and
`temperature_b` are `pta` and `ptb`; its separate global `temperature` is
the token sampling setting where used. TSE fusion selects its fused top token
deterministically. Never use a bare `t` in a new ID.

Omit an axis only when it is fixed by the protocol and cannot distinguish two
runs in the same study. Record both axes when the distinction matters, even
when one implementation couples them. In particular, record probability
fusion and commit ranking separately: Jensen–Shannon agreement changes the
position-ranking score, not the fused token distribution.

Examples under the new convention:

```text
ctca-gsm8k300-llada8bi-single-st0p0-pt1p0-nativeprob-entrank-fiveshot-20261019
ctca-gsm8k300-dream7bi-single-st0p1-pt0p1-nativeprob-entrank-zeroshot-20261019
tse-gsm8k-llada8bi-ens-st0p0-pta0p2-ptb0p7-w0p5-probmean-confrank-20261019
```

The earlier TSE temperature-study IDs below are legacy IDs. Their bare `t`
means probability temperature, as their configs confirm. Keep historical IDs
stable because result paths and W&B runs use them; consult the config rather
than inferring a historical run's sampling temperature from `t`.

The same ambiguity occurs in the existing CTCA records: job `314638` has
`t1p0` in its native Dream ID but actually used sampling temperature `0.0`
and an unscaled softmax; job `323687` has `t0p1` in its native Dream ID and
used sampling temperature `0.1`; `t0p1` in the LLaDA-master CTCA calibration
ID means Dream/model-B probability temperature `0.1`, with deterministic CTCA
token selection. These IDs are retained as artifact identifiers, not examples
of the new naming convention.

```text
tse-gsm8k-llada8bi-single-t1p0-nativeconf-20260926
tse-gsm8k-llada8bi-single-t0p2-nativeconf-20260926
tse-gsm8k-llada8bi-single-t0p7-nativeconf-20260926
tse-gsm8k-llada8bi-ens-t0p2-t0p7-w0p5-probmean-jsrank-20260927
tse-gsm8k-llada8bi-ens-t0p2-t0p7-w0p5-probmean-confrank-20261013
```

The ledger must also store the corresponding characteristics as structured
fields: `study_id`, `fusion`, `weighting`, `alpha`,
`token_sampling_temperature`, `probability_temperatures`, and `commit_score`.
For CTCA, order `probability_temperatures` as model A then model B. Names make
runs recognizable; structured fields make them queryable and prevent ambiguous
comparisons.

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

  The pairs are the tokenizer-specific EOS and turn-end tokens. CTCA stores
  both rows as model-owned settings (`model_a_*` for LLaDA and `model_b_*` for
  Dream), then automatically applies the row belonging to `master_model`
  because the generated canvas is in the master's vocabulary.
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
  EOS prediction cannot terminate the code body. CTCA stores both model-owned
  rows and inherits the selected master's row automatically.
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
- Define task-specific suppression independently for LLaDA/model A and
  Dream/model B. The master grid inherits the selected model's suppression
  policy; changing `master_model` therefore cannot leave the other tokenizer's
  token IDs active. Both always-on and begin-only suppression are covered by
  focused tests.

The suppression IDs above are semantic equivalents, not interchangeable IDs:
LLaDA EOS/turn-end/mask are `126081`/`126348`/`126336`, while Dream
EOS/turn-end/mask are `151643`/`151645`/`151666`. Never pass one tokenizer's IDs
to the other tokenizer's canvas. The native Dream and LLaDA controls, as well
as CTCA, expose the same `suppress_tokens` and `begin_suppress_tokens` behavior
used by this protocol.

### `ctca-gsm8k300-master-ablation-v1`

This protocol isolates output-master direction while retaining the locked
`ctca-gsm8k300-v2` dataset, prompts, models, generation schedule, static
probability mean, temperatures, projection, suppression policies, seeds, and
batch size. It changes only `master_model` from Dream/model B to LLaDA/model A
and moves `fusion_device` from `cuda:1` to the LLaDA-master device, `cuda:0`.
EOS handling remains unchanged so that master direction is the only scientific
factor in this ablation.

- Required experiment ID:
  - `ctca-gsm8k300-llada8bi-dream7bi-lladamaster-ens-t1p0-t1p0-w0p5-probmean-confrank-20261018`
- Configuration:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada_master_ablation.conf`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_llada_master_ablation_slurm.sh`.

### Model-specific GSM8K prompting and temperature

The study `ctca-native-prompt-temperature-20261018` tests Dream under the
requested zero-shot MaskGIT+ configuration and then combines model-native prompt
counts in LLaDA-master CTCA. The native Dream control uses `temperature=0.1`,
`alg=maskgit_plus`, `alg_temp=0.0`, and zero shots while retaining the parity
generation budget and full-vocabulary `top_p=1.0` calculation. Because positive
Dream sampling temperature introduces categorical token sampling, this first run
is a fixed-seed diagnostic; a paper result requires the predeclared repeated-seed
policy for stochastic experiments.

The two CTCA conditions use five demonstrations for LLaDA/model A and zero for
Dream/model B. The launcher requests enough demonstrations from lm-eval for
either model, then the CTCA eval layer independently selects each model's prompt
prefix while preserving the same target question. Only the selected master's
rendered prompt determines the sampler canvas, so these LLaDA-master runs have a
five-shot canvas without constraining Dream's independent prompt count. LLaDA
uses `temperature_a=1.0`; the paired
control uses Dream `temperature_b=1.0`, while the calibration condition uses
Dream `temperature_b=0.1`. All other settings are identical, including static
`alpha=0.5` fusion and deterministic CTCA decoding.

- Native Dream ID:
  `ctca-gsm8k300-dream7bi-single-t0p1-maskgit-confrank-zeroshot-20261018`.
- Native Dream protocol/config:
  `ctca-gsm8k300-dream-maskgit-zero-v1`,
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k300_maskgit_t0p1_zeroshot.conf`.
- Native Dream submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_controls_dream_native_alpha1_slurm.sh`.
  This three-hour control bundle runs the short alpha-one equivalence validation
  and the native Dream condition; the matched Dream control took 1.01 hours.
- CTCA neutral-temperature control ID:
  `ctca-gsm8k300-llada8bi-dream7bi-lladamaster-a5-b0-ens-t1p0-t1p0-w0p5-probmean-confrank-20261018`.
- CTCA neutral-temperature config:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada5_dream0_t1p0.conf`.
- CTCA Dream-temperature calibration ID:
  `ctca-gsm8k300-llada8bi-dream7bi-lladamaster-a5-b0-ens-t1p0-t0p1-w0p5-probmean-confrank-20261018`.
- CTCA protocol/calibration config:
  `ctca-gsm8k300-model-prompt-v1`,
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada5_dream0_t0p1.conf`.
- Standalone Dream-temperature-0.1 CTCA submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_llada5_dream0_temperature_slurm.sh`.
  Its eight-hour limit is based on the 5.96-hour matched CTCA run. The neutral
  temperature condition is paired with the earlier LLaDA-master run in the
  prompt-suite launcher below.

### Four-job launch plan

The six pending experiments fit the four-job user limit as follows. Bundled
launchers preserve the failure code but continue to their second experiment so
one failed condition does not prevent the other from running.

1. `ctca_controls_dream_native_alpha1_slurm.sh` (3 hours): alpha-one
   equivalence plus native Dream MaskGIT+ at temperature 0.1.
2. `ctca_humaneval_parity_slurm.sh` (12 hours): CTCA HumanEval parity. The
   estimate scales the 5.96-hour GSM8K CTCA runtime by its larger
   token-step workload and retains additional buffer.
3. `ctca_gsm8k300_llada_master_prompt_suite_slurm.sh` (16 hours): LLaDA-master
   five/five prompting followed by the five/zero neutral-temperature condition.
4. `ctca_gsm8k300_llada5_dream0_temperature_slurm.sh` (8 hours): the five/zero
   Dream-temperature-0.1 condition.

### `ctca-gsm8k32-dream-batch-v1`

The study `ctca-dream-batch-diagnostic-20261026` checks whether Dream's poor
GSM8K output in job `323687` depends on mixed-length batching. The diagnostic
evaluates the first 32 examples from the same canonical `gsm8k_cot` test order.
It retains the Instruct checkpoint, zero-shot chat prompt, `bfloat16`, 512
generated tokens, 128 steps, sampling temperature 0.1, `top_p=1.0`, no top-k
filter, `maskgit_plus`, `alg_temp=0.0`, right-shifted logits, suppression policy,
and harness seed tuple from job `323687`. Only `batch_size` changes from 8 to 1.

- ID: `ctca-gsm8k32-dream7bi-single-st0p1-pt0p1-nativeprob-confrank-b1-zeroshot-20261026`.
- Config:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k32_maskgit_st0p1_pt0p1_batch1_zeroshot.conf`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/dream_gsm8k32_batch1_slurm.sh`.

Compare the new samples by `doc_id` with the first 32 samples of job `323687`.
Record flexible and strict exact match, empty-response count, and count of
responses that reproduce a suffix of the question ending in `A:`. Sampling at
temperature 0.1 means per-example text need not match across batch sizes. This
32-example diagnostic must not replace the 300-example control or be used as a
paper accuracy estimate. If the failure persists at batch size 1, inspect raw
generated token IDs before `sample_trim` and compare Dream's native 256-token,
256-step recipe separately.

Job `248059` completed on one NVIDIA RTX 5000 Ada GPU: 15/32 flexible EM
(46.88% ± 8.96%), 3/32 strict EM (9.38% ± 5.24%), and 474 seconds of lm-eval
time. The W&B summary and sample artifact are stored under the configured
`.logs` path. No response is empty or repeats a question suffix ending in `A:`.

### `ctca-gsm8k32-dream-suppression-v1`

The study `ctca-dream-suppression-diagnostic-20261026` isolates the
`begin_suppress_tokens` setting from Dream job `323687` on its first 32 GSM8K
questions. The new run keeps batch size 8, the same zero-shot chat prompts,
checkpoint, 512/128 generation budget, `maskgit_plus`, sampling temperature
0.1, `top_p=1.0`, no top-k filter, right-shifted logits, `bfloat16`, and seed
tuple. It changes `begin_suppress_tokens` from `[151643;151645]` to `[]`.
The always-on `suppress_tokens` list is `[]` in both runs.

Both pending 32-example Dream runs use the shared launcher with
`--device cuda:0` passed to lm-eval, no duplicate `device` in `--model_args`,
and no `--cache_requests` option. Response caching via `--use_cache` remains
enabled. Jobs `323687` and `327581` used `device=cuda:0` in model arguments
and `--cache_requests true`; their logs show all 300 requests were generated
and results saved, so those options did not cause their low scores.

- ID: `ctca-gsm8k32-dream7bi-single-st0p1-pt0p1-nativeprob-confrank-b8-nosuppress-zeroshot-20261026`.
- Config:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k32_maskgit_st0p1_pt0p1_batch8_nosuppress_zeroshot.conf`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/dream_gsm8k32_batch8_nosuppress_slurm.sh`.

Compare by `doc_id` with the first 32 samples of job `323687`, recording both
exact-match metrics, empty responses, and question-suffix responses. The two
runs have the same batch composition, but token sampling at 0.1 remains
stochastic. This diagnostic is not a paper accuracy estimate and does not
isolate the other differences from the earlier full Dream entropy run.

Job `248066` completed on one NVIDIA RTX 5000 Ada GPU: 20/32 flexible EM
(62.50% ± 8.70%), 0/32 strict EM, and 355 seconds of lm-eval time. The W&B
summary and sample artifact are stored under the configured `.logs` path. No
response is empty or repeats a question suffix ending in `A:`. The two new
runs used the same 32 prompts and targets as job `323687`, whose first 32
scored 2/32 flexible and 0/32 strict, with 20 question-suffix responses and
five empty responses. Both new jobs ran on another cluster, so the comparison
with job `323687` also changes the execution environment. The raw lm-eval
result JSON files remain on that cluster; the ledger reads the downloaded
W&B summaries until those files are copied over.

### `ctca-gsm8k300-dream-suppression-v1`

The study `ctca-dream-nosuppression-baseline-20261026` extends successful
job `248066` from the first 32 to the first 300 canonical `gsm8k_cot` test
questions. It keeps Dream-v0-Instruct-7B, zero-shot chat prompting, batch size
8, 512 generated tokens, 128 steps, `maskgit_plus`, `temperature=0.1`,
`top_p=1.0`, no `top_k`, `alg_temp=0.0`, `right_shift_logits=true`, and both
suppression lists empty. The new experiment ID and response-cache path keep
its requests separate from job `323687`.

- ID: `ctca-gsm8k300-dream7bi-single-st0p1-pt0p1-nativeprob-confrank-b8-nosuppress-zeroshot-20261026`.
- Config:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k300_maskgit_st0p1_pt0p1_batch8_nosuppress_zeroshot.conf`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/baseline/dream_gsm8k300_batch8_nosuppress_slurm.sh`.

Job `248130` completed on one NVIDIA RTX 5000 Ada GPU. The downloaded W&B
summary reports 170/300 flexible EM (56.67% ± 2.87%), 0/300 strict EM, and
2,950 seconds of lm-eval time. Its first 32 prompts and responses exactly
match job `248066`; across all 300 responses, none is empty or a repeated
question suffix ending in `A:`. The remote run used the protocol group
`ctca-gsm8k300-dream-suppression-v1`; the local config and ledger now use
that group. The raw lm-eval result JSON remains on the other cluster.

Dream's native `temperature=0.1` scales logits before softmax and samples
candidate tokens from that distribution. For later LLaDA-master CTCA runs,
keep `temperature_a=1.0`, set Dream `temperature_b=0.1`, leave global
`temperature=0.0` for the current greedy fusion path, and use empty Dream
suppression lists. With LLaDA as master, only LLaDA's begin-suppression list
is applied to commit scores. With Dream as master, use an empty master
begin-suppression list. The CTCA TSE path chooses fused tokens greedily even
when Dream probabilities use temperature 0.1; this native stochastic baseline
therefore does not isolate fusion alone. A later sampling-matched comparison
needs a Dream control with greedy token proposals and 0.1 probability scaling,
or a CTCA path that samples fused tokens.

### `ctca-gsm8k300-dual-master-nosuppress-v1`

The study `ctca-dual-master-nosuppression-20261026` runs the same 300 GSM8K
questions through two 50/50 LLaDA/Dream probability ensembles, changing the
master canvas between LLaDA and Dream. LLaDA is model A, five-shot, with
`temperature_a=1.0`; Dream is model B, zero-shot, with
`temperature_b=0.1`. Both use `bfloat16`, batch size 8, 512 tokens, 128
steps, block size 32, confidence ranking, greedy fused token proposals,
static `alpha=0.5`, `weight_temperature=1.0`, and empty Dream suppression
lists. LLaDA retains its native `[126081;126348]` beginning-token list.

| Master | Canvas prompt | Fusion device | Config | Submission script |
|---|---|---|---|---|
| LLaDA | five-shot | `cuda:0` | `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_llada_master.conf` | `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_nosuppress_llada_master_slurm.sh` |
| Dream | zero-shot | `cuda:1` | `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_dream_master.conf` | `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_nosuppress_dream_master_slurm.sh` |

In the CTCA implementation, beginning-token suppression applies only to the
master's commit scores. Clearing Dream's list affects the Dream-master run;
it does not change the LLaDA-master computation because Dream is auxiliary.
The LLaDA-master condition is therefore a new-cluster rerun of the earlier
five/zero, `temperature_b=0.1`, confidence-ranked CTCA condition. The Dream
master directly tests the no-suppression setting on a Dream token canvas.
Both CTCA runs choose fused tokens greedily and use blockwise generation,
whereas the native Dream baseline samples at temperature 0.1 without CTCA
blocks. Interpret native-versus-ensemble differences with those distinctions.

### `ctca-gsm8k300-entropy-v1`

The study `ctca-entropy-comparison-20261019` pairs two native entropy-ranked
GSM8K-300 controls with one entropy-ranked LLaDA-master CTCA ensemble. All three
use the first 300 `gsm8k_cot` test questions, chat
formatting, `bfloat16`, `max_new_tokens=512`, `steps=128`, batch size 8, the
same seed tuple, and the model-specific suppression tokens from the locked
parity controls. Native controls retain independent single-GPU execution.

- Dream ID:
  `ctca-gsm8k300-dream7bi-single-st0p1-pt0p1-nativeprob-entrank-zeroshot-20261019`.
  Relative to job `323687`, only `alg` changes from `maskgit_plus` to `entropy`;
  Dream still samples tokens at `temperature=0.1`, with `top_p=1.0`, no `top_k`,
  and zero-shot prompting. Entropy is computed from the sharpened probability
  distribution. This is a fixed-seed diagnostic; the stochastic paper result
  requires at least three predeclared seeds.
- LLaDA ID:
  `ctca-gsm8k300-llada8bi-single-st0p0-pt1p0-nativeprob-entrank-fiveshot-20261019`.
  It uses the official LLaDA sampling temperature `0.0`, an unscaled softmax
  (`probability_temperature=1.0`), five-shot prompting, `block_size=32`, and
  native greedy tokens. The new `remasking=entropy` policy ranks positions by
  negative categorical entropy of the softmax, matching the position-score
  definition in Dream's entropy policy. The official LLaDA generator supports
  `low_confidence` and `random`, so this policy is a local extension rather
  than an official LLaDA default.
- Native configs:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/dream_gsm8k300_entropy_st0p1_pt0p1_zeroshot.conf`
  and
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/llada_gsm8k300_entropy_fiveshot.conf`.
- Single submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_native_entropy_controls_slurm.sh`.
  It runs LLaDA then Dream in one eight-hour, one-GPU allocation and attempts
  both even if the first evaluation fails.
- CTCA ID:
  `ctca-gsm8k300-llada8bi-dream7bi-lladamaster-a5-b0-ens-st0p0-pta1p0-ptb0p1-w0p5-probmean-entrank-20261019`.
  This keeps the LLaDA-master five/zero prompt split, `alpha=0.5`, and
  `temperature_a=1.0`, `temperature_b=0.1` from the existing CTCA
  Dream-temperature calibration. Token proposals remain greedy. Only the
  commit ranking changes: it selects lowest entropy of the fused probability
  distribution instead of highest fused top-token confidence.
- CTCA config and submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada5_dream0_entropy_ptb0p1.conf`
  and
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_entropy_slurm.sh`.
  The latter requests two GPUs for twelve hours.

The LLaDA five-shot and Dream zero-shot prompts match the respective model
inputs in the LLaDA-master CTCA five/zero conditions; the generation budgets
and raw model probability temperatures also align. The new CTCA condition
allows an entropy-ranking comparison against both native controls and a
within-CTCA comparison against the existing confidence-ranked run. It does not
perfectly isolate fusion against native Dream: Dream samples tokens at 0.1,
whereas CTCA proposes the fused argmax deterministically. The shared protocol
stays separate from the locked confidence-ranked parity and model-prompt
protocols.

### `ctca-llada-master-equivalence-v1`

### CTCA component trace: LLaDA master, one GSM8K question

This mechanism-only smoke run uses the existing structured CTCA trace callback
to record Dream-to-LLaDA canvas alignment, spatially warped and vocabulary-
projected probabilities, model and fused top candidates, confidence scores,
per-position ranks, scheduled transfers, and actual commits. It uses
`alpha=0.5`, `pta=1.0`, and `ptb=0.1`, so both models contribute to fusion.
It replays question 0 from the prior low-scoring LLaDA-master samples. The
standalone trace prompt is zero-shot, so this diagnoses pipeline mechanics and
does not provide a directly comparable GSM8K accuracy result.

- Experiment ID:
  `ctca-gsm8k1-llada-master-trace-ptb0p1-a0p5-20261026`.
- Configuration:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k1_llada_master_trace.conf`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k1_component_trace_slurm.sh`.
- Structured trace:
  `/home/sarthak.malla/dllm-epiplexity/.logs/ctca-gsm8k1-component-trace/ctca-gsm8k1-llada-master-trace-ptb0p1-a0p5-20261026/<job-id>/trace.jsonl`.

This operational validation checks the static-fusion endpoint requested for the
LLaDA-master pipeline. It runs native LLaDA and normal CTCA TSE selection on the
same LLaDA model instance and prompt, with `master_model=a`, `weighting_mode=static`,
and `alpha=1.0`. The auxiliary Dream forward and CTCA projection remain active;
this is deliberately an endpoint-correctness test, not an optimized single-model
fast path.

The check passes only when every post-commit canvas, the final token tensor, and
the native-EOS-trimmed decoded text are identical. It also records native and CTCA
wall times so the cost of retaining the inactive auxiliary branch is explicit.
Because this is a one-prompt implementation validation rather than a benchmark
quality result, it is not a scientific ledger entry.

The smoke schedule is `max_new_tokens=64`, `steps=16`, and `block_size=32`.
This exercises two generation blocks and repeated confidence-ranked commit
decisions while remaining eight times shorter than the full 128-step run.

- Validation ID:
  `ctca-llada8bi-alpha1-nativeequiv-20261018`.
- Configuration:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/configs/ctca_llada_alpha1_native_equivalence.conf`.
- Python runner:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments/verify_llada_alpha1_native_equivalence.py`.
- Submission script:
  `/home/sarthak.malla/dllm-epiplexity/scripts/tse/ctca/ctca_llada_alpha1_native_equivalence_slurm.sh`.
- Result artifact:
  `/home/sarthak.malla/dllm-epiplexity/.logs/ctca-llada-master-equivalence-v1/ctca-llada8bi-alpha1-nativeequiv-20261018/<job-id>/equivalence.json`.

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
