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

### Planned GSM8K300 CTCA agreement ranking (2026-11-17)

User-approved paired LLaDA-master and Dream-master runs use
`remasking="consensus"`: fused top-token probability multiplied by
JS agreement (`1 - weighted_JS / weight_entropy`, clamped to [0, 1]).
This changes the order of committed positions; greedy fused token proposals,
512-token / 128-step / 32-token-block schedule, static alpha 0.5, probability
temperatures 1.0/1.0, LLaDA five-shot / Dream zero-shot prompts, batch size 8,
suppression, and exact-shared sparse projection remain inherited from the
matching at0p03/pt0p10 confidence controls. Projection uses all shared anchors
and top-k 64. The two planned records in
[`experiments.json`](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/experiments.json)
use protocol `ctca-gsm8k300-dualmaster-jsrank-at0p03-projtemp0p10-v1`
and separate response caches. No scheduler jobs have been submitted.

- LLaDA config: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_lladamaster_jsrank_at0p03_pt0p10.conf`.
- Dream config: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_dreammaster_jsrank_at0p03_pt0p10.conf`.
- Submit LLaDA: `sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_lladamaster_jsrank_at0p03_pt0p10_slurm.sh`.
- Submit Dream: `sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_dreammaster_jsrank_at0p03_pt0p10_slurm.sh`.

### Experiment ID format

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
- Evaluation batch size: `4` for LLaDA, Dream, and CTCA. The initial LLaDA
  batch-size-8 attempt ran out of memory at 128/164 problems; use the same
  batch size for every HumanEval condition.
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

### `ctca-humaneval-exactshared-v1`

This follow-up evaluates the exact-shared projection and both fusion endpoints
on the same 164 HumanEval problems, common zero-shot `humaneval_instruct`
prompts, and 768/192/32 decoding schedule. It matches jobs `352346` and
`352347` in master direction, static 50/50 probability fusion, and
`temperature_a=temperature_b=1.0`. It also runs alpha 1.0 with LLaDA as master
and alpha 0.0 with Dream as master, analogous to endpoint jobs `350508` and
`352165`. Both models remain loaded in endpoint runs. All four use the exact
shared-token policy, model-owned HumanEval EOS suppression, deterministic
confidence ranking, and the required code-evaluation safety flag. The endpoint
comparisons are against the matching master direction.
All four runs use batch size 4, matching the HumanEval single-model controls.

- Configs:
  - `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_humaneval_lladamaster_exactshared.conf`
  - `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_humaneval_dreammaster_exactshared.conf`
  - `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_humaneval_lladamaster_alpha1_endpoint.conf`
  - `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_humaneval_dreammaster_alpha0_endpoint.conf`
- Submission wrapper:
  `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_humaneval_parity_slurm.sh`
- Submit each condition by passing its config path to the wrapper; the wrapper
  defaults to the original Dream-master parity config when no argument is given.

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
| LLaDA | five-shot | `cuda:0` | `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_llada_master.conf` | `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_nosuppress_llada_master_slurm.sh` |
| Dream | zero-shot | `cuda:1` | `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_nosuppress_dream_master.conf` | `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_nosuppress_dream_master_slurm.sh` |

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

### CTCA top-64 distribution trace: two GSM8K questions

Planned experiment `ctca-gsm8k2-llada8bi-dream7bi-st0p0-pta1p0-ptb0p1-w0p5-probmean-confrank-fulltrace-20261027` replays two questions from the LLaDA-master CTCA job `323690` sample log. Document 23 (candle, gold 8) had flexible exact match 1; document 0 (Janet's ducks, gold 18) had flexible exact match 0 with extracted answer 4. These are **previous-run labels**: this zero-shot, 96-token mechanism trace does not reproduce the five-shot GSM8K evaluation and must be evaluated separately for any new accuracy claim.

The JSONL records decoded token strings with top-64 probabilities and omitted mass for every native masked position of both models, the spatially warped auxiliary distribution, auxiliary-to-master projection, and fusion at every decoding step. Numeric vocabulary IDs are omitted from the stored candidates. It also retains alignment contributors, weights, scores, ranks, and commits. The two cases use the same model and fusion settings as the preceding one-question trace.
The final response is stored with a simple last-number extraction for convenient inspection; this diagnostic extraction is not the harness flexible exact-match metric.
The viewer's master-weight slider recomputes the fusion probabilities for the union of saved master, projected-auxiliary, and recorded-fusion top-k tokens. It reports that candidate set's probability coverage. The slider is a per-position preview; changing a weight does not rerun decoding or update later commits.

- Configurations: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_fulltrace_correct.conf` and `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_fulltrace_wrong.conf`.
- Submission: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_fulltrace_slurm.sh` (`long`, `gpu-12`, two GPUs).
- Output: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-fulltrace/<experiment-id>/<job-id>/{correct,wrong}.jsonl`.
- Local viewer: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/fulltrace_viewer.html`.
- Latest-run manifest: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-fulltrace/latest.json`, written after both JSONL files finish. The viewer loads those two traces on refresh or when **Load latest run** is clicked; before completion it loads the CPU demo.
- CPU viewer demo: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/demo_dummy_trace.jsonl`, generated by `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/generate_dummy_trace.py`. It shows `The cat sat on the [MASK][MASK]`: the dummy master prefers `mat`, the dummy auxiliary prefers `rug`, and both prefer `.` for the second slot.

### Two-question full-trace Dream temperature control

The planned diagnostic `ctca-gsm8k2-llada8bi-dream7bi-st0p0-pta1p0-ptb1p0-w0p5-probmean-confrank-fulltrace-20261029` repeats both the candle (previously correct) and Janet (previously wrong) questions from job `350507` with `temperature_b=1.0` instead of `0.1`. It keeps the zero-shot 96-token, 24-step, 32-token-block schedule, LLaDA master, static 50/50 fusion, confidence ranking, top-64 trace, and cached relative-anchor projection fixed. This isolates Dream probability sharpness from the known vocabulary map error: Dream's standalone space token currently projects to LLaDA's `Ġand` with nearly all its mass.

- Configs: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_fulltrace_correct_ptb1p0.conf` and `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_fulltrace_wrong_ptb1p0.conf`.
- Submission: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_fulltrace_ptb1p0_slurm.sh`.
- Traces: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-fulltrace-ptb1p0/<experiment-id>/<job-id>/{correct,wrong}.jsonl`.
- Viewer manifest: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-fulltrace-ptb1p0/latest.json`, written after both traces finish.
- Compare the first committed positions, spatial and projected distributions, selected-token counts, and final diagnostic responses against the paired `correct.jsonl` and `wrong.jsonl` from job `350507`. A more coherent response would support a temperature contribution; it would not validate the vocabulary map or establish benchmark accuracy.

Attempt `351424` reached `gpu-57` but failed before Python started: the `srun nvidia-smi` preflight could not satisfy the node's CPU binding request. Neither trace nor the new manifest exists; the viewer therefore displayed the previous `temperature_b=0.1` run. The shared smoke launcher now invokes `nvidia-smi` directly from its allocated batch shell, avoiding a separate Slurm step. Resubmit the same two-question wrapper after this change.

Rerun `351805` completed both JSONL traces and wrote the new viewer manifest. The candle and Janet responses end in diagnostic last numbers 8 and 18, matching their gold answers; the preceding `temperature_b=0.1` run ended in 8 and 16. Selected `Ġand` commits fell from 16 to 1 for candle and 91 to 5 for Janet. At the first bad Janet position (absolute position 89, step 0.0), the old projected Dream probability of `Ġand` was 0.9994 and its fused probability was 0.5036, so it was committed. At `temperature_b=1.0`, the projected `Ġand` probability is 0.2144 and the fused probability is 0.1111; LLaDA's space wins at 0.1210. The erroneous space-to-`Ġand` mapping remains in the cache, so this result supports temperature amplification of the error rather than a projection fix. Both are 96-token zero-shot mechanism traces, not a five-shot GSM8K accuracy measurement.

### Next problem: `ctca-projection-exact-token-guard-v1`

Status: first mapping correction implemented after the two-question temperature control completed. The initial experiment changes only ordinary shared-token rows; nonshared and special-token policies remain separate factors.

In the cached Dream-to-LLaDA projection, Dream's standalone space token (`Ġ`, ID 220) maps almost entirely to LLaDA `Ġand` (ID 301). In the Janet full trace from job `350507`, Dream's spatial space probability at the first bad position is about 0.9994; after projection it becomes `Ġand` with nearly the same probability. The 50/50 fused probability of `Ġand` is 0.5036 despite LLaDA assigning it only 0.0078; it is committed while agreement is only 0.0348. LLaDA-master job `350510` scored 26/300 flexible EM versus 209/300 for the LLaDA-only endpoint job `350508`; 81/300 ensemble responses contain a run of at least five consecutive `and` tokens. The reverse map sends LLaDA's standalone space mostly to Dream punctuation, and splits LLaDA's digit `2` about equally between Dream's digit `2` and Dream's mask-token ID 151666. Both directional caches map only about 68% of tokens shared by exact raw spelling to that same token as their top choice. These are concrete mapping and commit-selection defects; they do not yet quantify how much each contributes to benchmark accuracy.

The first correction uses exact raw vocabulary spelling to pair ordinary shared token IDs, then assigns 100% of each such auxiliary row to its matching master ID. It runs when loading the existing relative-anchor table, so the expensive cache remains valid and is reused. This fixes Dream `Ġ` → LLaDA `Ġ` and Dream `Ġeats` → LLaDA `Ġeats` without decoded-response matching. At Janet position 79, step 0.1 in job `351805`, Dream's spatial `Ġeats` probability 0.2662 was split into `Ġeats` 0.1331 and `ĠEats` 0.1331, letting `3` (0.2073) rank first in the projected distribution. With the exact row, that 0.2662 contribution goes entirely to `Ġeats` (plus any incoming mass from other rows), so it outranks `3` at that position if the rest of the distribution is unchanged.

The controlled paired candle/Janet trace `ctca-gsm8k2-llada8bi-dream7bi-st0p0-pta1p0-ptb1p0-w0p5-probmean-confrank-exactshared-fulltrace-20261029` uses `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_fulltrace_exactshared_slurm.sh`. Its only intended difference from job `351805` is this shared-row override. Inspect first commits, mapped probabilities, repeated tokens, and final diagnostic answers. The existing cache's nonshared rows still use relative-anchor similarities; some may be wrong, and a token that needs multiple master tokens cannot be faithfully represented as one independent output token. Special tokens also need an explicit semantic or abstention policy; do not force their mass onto a lexical token. Audit both cases separately before a 300-question benchmark. Test agreement-based commit ranking as a later independent factor, with the schedule fixed.

Job `351988` completed both traces and its viewer manifest. The existing relative-anchor cache was loaded without rebuilding (`sparse_topk_cache_rebuilt=false` for both questions), and the trace reports `relative_anchor_plus_exact_shared`. Both responses end in the diagnostic gold numbers (candle 8, Janet 18). Relative to job `351805`, selected `Ġand` commits fell from 1 to 0 for candle and 5 to 0 for Janet; mean model agreement rose from 0.3664 to 0.6386 and from 0.3607 to 0.7372. At the same initial Janet canvas (position 89, step 0.0), Dream's spatial standalone-space probability is 0.1759 in both runs. The old projection gave LLaDA space only 0.00006 and `Ġand` 0.2144; the exact-shared projection gives LLaDA space 0.1759 and `Ġand` 0.0380. At initial positions 87 and 88, Dream `Ġeats` probability is 0.0744 and 0.0629; the old map split each half between `Ġeats` and `ĠEats`, whereas the corrected map puts the full contribution on `Ġeats`. The user's later example at position 79, step 0.1 cannot be compared directly because earlier token commits changed the context. A CPU audit of the reverse LLaDA-to-Dream cache confirms that LLaDA space, digit `2`, and `Ġand` now map one-to-one to Dream's matching IDs; no Dream-master trace has yet tested the effect during decoding. The Janet response still starts with the false phrase `Janet's ducks lay 3 eggs per day` despite computing 18 later; final-number agreement is not full reasoning correctness. Continue auditing nonshared and special-token rows before treating this as a benchmark improvement.

### Dream-master paired mapping trace: `ctca-gsm8k2-dreammaster-exactshared-20261029`

Planned controlled diagnostic: run the candle and Janet questions twice in one two-GPU Slurm job, first with the original cached relative-anchor rows and then with exact shared-token rows overlaid in memory. Both conditions use Dream as master, LLaDA as auxiliary, zero-shot prompts, 96 generated tokens, 24 steps, 32-token blocks, static 50/50 probability fusion, both probability temperatures 1.0, greedy fused token proposals, confidence-ranked commits, and no Dream master-token suppression. The only within-pair algorithm difference is `ctca_shared_token_policy=relative` versus `exact_shared`. The current LLaDA-master trace does not supply a matched Dream-master control because changing the master also changes the tokenizer canvas and projection direction.

- Relative configs: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_dreammaster_correct_relative.conf` and `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_dreammaster_wrong_relative.conf`.
- Exact-shared configs: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_dreammaster_correct_exactshared.conf` and `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k2_dreammaster_wrong_exactshared.conf`.
- Submission: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k2_dreammaster_exactshared_slurm.sh`.
- Viewer manifests: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-dreammaster-exactshared/{relative,exactshared}/latest.json` after each pair finishes.

Evaluate each pair's first decoding step, then track the earliest differing commit and subsequent numeric or repetition errors. At the same initial canvas, verify that LLaDA space and digit `2` send all their mass to Dream's matching IDs instead of punctuation or Dream's mask token. Later-step positions have different contexts after divergent commits and should not be treated as direct probability controls. The final-number check is diagnostic only; read the complete responses before judging reasoning quality. Keep this short trace separate from a five-shot/zero-shot GSM8K-300 benchmark.

### Paired GSM8K-300 benchmark with the current projection

The user elected to benchmark the current projection as implemented, including
exact raw-spelling rows for ordinary shared tokens and the unchanged cached
relative-anchor fallback for nonshared and special-token rows. The paired
scientific runs are `ctca-gsm8k300-lladamaster-exactshared-ptb1p0-20261029` and
`ctca-gsm8k300-dreammaster-exactshared-ptb1p0-20261029`, both under protocol
`ctca-gsm8k300-dualmaster-exactshared-ptb1p0-v1`. Each evaluates the first 300
`gsm8k_cot` questions with five-shot LLaDA and zero-shot Dream prompts,
`max_new_tokens=512`, `steps=128`, `block_size=32`, static 50/50 fusion,
`temperature_a=temperature_b=1.0`, and confidence-ranked greedy commitments.
The only within-pair change is the master model and its resulting canvas and
projection direction. Both jobs exclude `gpu-38` along with the existing
`gpu-51,gpu-24` exclusions.

The completed Dream-master two-question paired trace job `352151` is a warning
about interpreting aggregate scores. Its relative condition ended in candle 8
and Janet 18; its exact-shared condition ended in candle 800 and Janet 18.
Exact-shared mean agreement rose from 0.209 to 0.658 for candle and from 0.301
to 0.707 for Janet, but high agreement did not guarantee quality. At candle
step 0.7, position 89, LLaDA assigned about 0.999 to `Ġcentimeters`, whose
nonshared cached row sends 0.999982 to Dream `Ġkilometers`; the 50/50 fused
probability of kilometers 0.4995 narrowly beat Dream `Ġcm` at 0.4973. This
mapping predates the exact-shared override and was exposed by the changed
decoding context. The benchmark measures aggregate performance with this known
defect; it is not a clean estimate of exact-shared projection quality alone.
The previous 300-question ensemble jobs used `temperature_b=0.1` and the
original relative map, so compare them descriptively rather than as one-factor
controls. The planned Dream-master alpha-zero endpoint shares the temperature
and decoding schedule with this study and remains the relevant same-schedule
control for auxiliary fusion.

- LLaDA-master config: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_lladamaster_exactshared_ptb1p0.conf`.
- Dream-master config: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_dreammaster_exactshared_ptb1p0.conf`.
- Submission wrappers: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_lladamaster_exactshared_ptb1p0_slurm.sh` and `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_dreammaster_exactshared_ptb1p0_slurm.sh`.
- Result directories: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k300-dualmaster-exactshared-ptb1p0-v1/<experiment-id>/<job-id>/`.

Jobs `352346` (LLaDA master) and `352347` (Dream master) completed 300/300
questions and W&B logging. The jobs collected lm-eval samples and metrics, not
per-position full JSONL traces. The matched short full traces from jobs
`351988` and `352151` remain the position-level diagnostics; they use 96 tokens
and zero-shot prompts, unlike this 512-token benchmark.

| Condition | Flexible EM | Strict EM | Evaluator time |
|---|---:|---:|---:|
| LLaDA-only CTCA endpoint `350508` | 209/300 (69.67% ± 2.66%) | 133/300 (44.33% ± 2.87%) | 5.67 h |
| LLaDA-master exact-shared `352346` | 187/300 (62.33% ± 2.80%) | 141/300 (47.00% ± 2.89%) | 4.80 h |
| Dream-only CTCA endpoint `352165` | 184/300 (61.33% ± 2.82%) | 2/300 (0.67% ± 0.47%) | 5.05 h |
| Dream-master exact-shared `352347` | 194/300 (64.67% ± 2.76%) | 145/300 (48.33% ± 2.89%) | 4.49 h |

On paired flexible EM, Dream-master fusion gained 59 and lost 49 questions
against Dream-only under the same schedule (net +10/300, exact paired
McNemar/binomial p≈0.387). LLaDA-master fusion gained 32 and lost 54 against
LLaDA-only (net -22/300, p≈0.023). The two ensemble masters share 167 correct
questions; LLaDA alone wins 20 and Dream alone wins 27 (net +7/300 for Dream,
p≈0.382). Thus the current ensemble has not clearly improved on its same-master
endpoint for Dream and has reduced performance for LLaDA. The strict filter
requires a literal `The answer is <number>` pattern; its near-zero Dream-only
score reflects formatting and should not be used as a reasoning-quality
comparison. Flexible EM is the primary metric.

All 300 Dream-master ensemble responses had a flexible number extraction;
LLaDA-master had two failures, versus eight for the Dream-only endpoint. Neither
new ensemble had a five-`and` loop, but six LLaDA-master and five Dream-master
responses still had a run of at least 15 digits, and those examples were
incorrect. The old `temperature_b=0.1`, relative-map ensembles scored 26/300
LLaDA master and 139/300 Dream master; the large difference from the new
scores combines temperature and mapping changes and cannot identify their
individual effects. On candle doc 23, the 300-question Dream-master ensemble
again changes centimeters to kilometers and also misreads 1:00 as 12:00,
answering 10 instead of 8; the Dream-only endpoint answers 8. The two-question
trace's answer 800 does not reproduce exactly because its prompt and generation
budget differ, but the known projection defect is visible in both settings.

#### Failure decomposition before the next algorithm change

The same-schedule endpoints make auxiliary fusion's aggregate effect observable:
Dream-master fusion gains 59 and loses 49 endpoint-correct questions, while
LLaDA-master fusion gains 32 and loses 54. The majority of losses are not
invalid outputs or long digit loops: among LLaDA-master's 54 losses, one has no
flexible extraction and one has a run of at least 15 digits; among Dream-master's
49 losses, none lack extraction and four have a long digit run. Most are
plausible-looking answers with a changed premise, arithmetic result, or final
number. For example, candle doc 23 misreads 1:00 as 12:00 in both ensembles;
Dream master also substitutes kilometers for centimeters. Dream-master doc 18
uses an incorrect count of days in a week; LLaDA-master doc 11 computes $694
in the body but ends with $94. These examples show distinct failure modes and
cannot be assigned to a specific token-level cause without a matched full
benchmark-prompt trace.

The harness flexible-extract score also penalizes some numerically equivalent
forms such as `23.0` for gold `23`. A separate diagnostic that parses the
filtered response as a decimal, removes currency symbols and commas, and
compares numerical value yields 196/300 for LLaDA-master fusion versus 219/300
for its endpoint, and 203/300 for Dream-master fusion versus 186/300 for its
endpoint. This diagnostic is not the official metric; the central conclusion
persists under it. Each ensemble has nine official losses that this numeric
comparison recovers.

In the short Dream-master traces, exact-shared projection reduced selected
content positions where the auxiliary spatial top text changed under projection
from 26/45 to 4/94 for candle and from 39/74 to 5/94 for Janet. Different
decoding histories and response lengths prevent treating these as matched
position-level rates. All four remaining candle changes in the exact-shared
trace are `centimeters` to `kilometers`. At its first unit commitment (block 0,
step 7, position 89), Dream master gives `Ġcm` probability 0.9937; the
spatially warped LLaDA distribution gives `Ġcentimeters` about 0.999, which
the cached fallback row projects to Dream `Ġkilometers` with weight 0.999982.
The fused probabilities are 0.4995 for kilometers and 0.4973 for cm, so greedy
fusion commits kilometers while full-distribution agreement is 0.0055. This is
the final step of that block, when every remaining position must be committed.
The same unit substitution later reaches agreement 0.947 after the wrong
context has propagated, so high downstream agreement cannot clear the initial
mapping. The code calculates agreement and consensus scores but uses fused top
probability for commit ranking under `remasking=low_confidence`; neither the
projection reliability nor agreement affects static fusion weights or token
selection. Separate from the unit row, the uncaught LLaDA end-of-text row maps
mostly to Dream punctuation (`).` variants), and LLaDA end-of-turn spreads
nearly uniformly across arbitrary Dream tokens; special-token semantics remain
unhandled by the exact-shared override.

The saved 300-question samples do not include per-position distributions.
Consequently, the unit substitution is a trace-proven projection/fusion error,
whereas the cause of most benchmark losses and shared digit loops is still
unresolved. The highest-value follow-up is a benchmark-prompt replay with the
same 512/128/32 schedule and both model-specific few-shot histories, capturing
full position traces for representative losses (for example candle doc 23,
wrong-premise doc 18, digit-loop doc 9, and final-number corruption doc 11)
alongside their same-master endpoint controls. A 300-question relative-map,
temperature-1.0 control would separately isolate the effect of exact-shared
rows on aggregate accuracy; the earlier 0.1 run cannot do so.

### Concurrent Dream-master endpoint control

The completed independent experiment `ctca-gsm8k300-dreammaster-a0-endpoint-st0p0-pta1p0-ptb1p0-confrank-20261029`, job `352165`, evaluates the first 300 `gsm8k_cot` questions with Dream as master and `alpha=0.0`, giving LLaDA's projected distribution zero fusion weight. It retains CTCA's zero-shot Dream canvas, five-shot LLaDA auxiliary prompt, 512 generated tokens, 128 steps, 32-token blocks, confidence-ranked greedy commits, and Dream probability temperature 1.0. The exact-shared policy is recorded explicitly but cannot alter the fused probabilities when the auxiliary weight is zero. This is a same-schedule CTCA endpoint to quantify Dream's performance without ensemble influence; native Dream MaskGIT+ differs in token sampling and eligible-position scope, and the previous Dream-master 300 run used temperature 0.1, so neither is a one-factor accuracy comparison with this control.

- Config: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_dreammaster_alpha0_endpoint_ptb1p0.conf`.
- Submission: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_dreammaster_alpha0_endpoint_slurm.sh`.
- Result: `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k300-dreammaster-a0-endpoint-v1/<experiment-id>/<job-id>/results_*.json` and paired `samples_*.jsonl`.

A CPU audit of the two existing relative-anchor cache tables found 82,790 shared raw vocabulary tokens. Before the exact-shared override, only 56,867/82,790 LLaDA-to-Dream rows and 56,622/82,790 Dream-to-LLaDA rows placed the matching token first; mean mass on that exact match was 0.5552 and 0.5322, respectively. The tables also contain at least 43,674 LLaDA rows and 69,274 Dream rows without a shared raw spelling, including unassigned and special-token rows. This is a vocabulary-wide structural audit, not a frequency-weighted estimate of generated-token error. The paired Dream-master trace is needed to see which rows actually affect decoding.

The Dream-master ensemble job `350512` scored 139/300 flexible EM, below both the LLaDA-only endpoint (209/300) and the separately run native Dream MaskGIT+ control (170/300). The paired LLaDA endpoint and Dream-master samples share 106 correct questions; Dream-master loses 103 of LLaDA's correct questions and gains 33 others. Dream-master has 32/300 responses with a run of at least 15 digits and 14/300 with no flexible extraction, versus 2/300 and 0/300 for the LLaDA endpoint. These errors are observations, not proof that LLaDA projection caused the whole gap. Native Dream and CTCA both use top-confidence commitment and commit four tokens per step under these settings. The implementations differ in eligible positions: native Dream considers all 512 masked positions at each of 128 steps; CTCA restricts each eight-step segment to the current 32-token block, as native LLaDA does. Native Dream samples token proposals at `temperature=0.1`, while CTCA takes greedy fused top tokens. Keep the current CTCA schedule fixed in the next ensemble experiments. A Dream-master `alpha=0` CTCA endpoint under that same schedule is needed to measure the effect of adding LLaDA's projected probabilities, and paired exact-projection and confidence-gating controls can then test improvements without changing commit timing or eligible positions.

The full GSM8K-300 alpha=1 endpoint benchmark extends the one-prompt native
equivalence check to the locked five-shot GSM8K protocol. It uses LLaDA as the
master, static `alpha=1.0`, LLaDA probability temperature `1.0`, and Dream
probability temperature `0.1` to match the current ensemble configuration.
Dream should receive zero fusion weight at this endpoint; the run tests whether
CTCA's full evaluation path preserves the native LLaDA outputs and score. It
retains the Dream forward and projection for a direct CTCA endpoint check.

- Experiment ID:
  `ctca-gsm8k300-llada8bi-dream7bi-lladamaster-a1-endpoint-pta1p0-ptb0p1-confrank-20261026`.
- Configuration:
  `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/configs/ctca_gsm8k300_llada_alpha1_endpoint_ptb0p1.conf`.
- Submission script:
  `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_llada_alpha1_endpoint_slurm.sh`.
- Compare its per-sample output and metrics with native LLaDA job `314694`.

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

### Benchmark failure replay preparation

The replay driver `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/benchmark_replay.py`
prepares eight full traces: Dream-master ensemble and alpha-zero endpoint for
GSM8K docs 23, 18, and 9, and LLaDA-master ensemble and alpha-one endpoint for
doc 11. It reads both native rendered prompts from jobs 352346 and 352347,
checks document identity and endpoint prompt equality against jobs 350508 and
352165, and uses the existing smoke trace logger with 512 tokens, 128 steps,
32-token blocks, both probability temperatures 1.0, and exact-shared projection.
These are batch-one replays of batch-eight benchmarks. Each pair saves its
original and replay response plus a reproduction flag; a mismatch prevents
claiming that its trace explains the original benchmark trajectory.

Submission wrapper: `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh STUDY_ID`.
The study ID is `ctca-gsm8k4-benchmark-replay-20261029`; job 352946 produced
all four pairs. The wrapper excludes gpu-38, gpu-51, and gpu-24. The viewer
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/fulltrace_viewer.html`
accepts `?manifest=/.logs/STUDY_ID/doc23.json` (or doc18, doc9, doc11) and displays
both traces and reproduction flags. Each pair becomes available when completed;
`latest.json` points to the last completed pair. No scheduler job was submitted
by the agent.

The CPU-only span audit `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/span_audit.py --study-id
ctca-gsm8k4-benchmark-replay-20261029` reads saved top-64 traces and writes
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-benchmark-replay-20261029/span_audit.json`. Probability
products are parallel approximations, not model rescoring. In doc 23 at step
0.0, the trace contains a text-exact Dream candidate ` cent` + `imeters` in
slots 69–70, with native master probability product about 0.0010. Slot 70
committed ` every` at step 0.0, so this span was blocked when slot 69 committed
` kilometers` at step 0.2. The audit separates the wrong one-token projection
from this earlier commitment constraint.
The audit's illustrative equal-weight span score for ` cent` + `imeters` at
step 0.0 is 0.277, below the weakest of the four actual committed single-slot
scores (0.356). Simply adding the span to the existing greedy candidate list
would therefore leave this failure unchanged; a decoder intervention needs a
separate span-ranking or slot-reservation policy and live model replay.
Across all four ensemble traces, high-confidence non-special source tokens
whose projected top text differs from the source occur at two selected
positions in doc 23 (both `centimeters` -> `kilometers`), zero in docs 18 and
9, and nine in doc 11 (punctuation/newline fragments). Thus, span scoring
targets a real projection defect but does not address the exact-token errors
that start the egg and overtime failures. Treat projection and commitment
quality as separate experimental axes.
An offline tokenizer-wide representability check found exact round-trip
spellings of length at most three tokens for 125,658/126,349 ordinary LLaDA
tokens in Dream (99.45%) and 142,881/151,643 ordinary Dream tokens in LLaDA
(94.22%). These are vocabulary counts, not probability-weighted coverage;
standalone token decoding may also differ from decoding in context.
Among selected ordinary auxiliary top predictions in the four replays, only
2/57 (doc 23), 0/98 (doc 18), 0/510 (doc 9), and 10/255 (doc 11) require two
or three master tokens to reproduce their text. This confirms broad vocabulary
representability but a much smaller impact on these actual decisions.

### Cross-tokenizer method review

Inference-time Agreement-Based Ensembling compares model outputs by their
surface strings and tracks when one autoregressive tokenizer trails another
(Wicks et al., 2025, https://arxiv.org/abs/2502.21265). Exact byte-level
probability conversion supports heterogeneous autoregressive model ensembles
(Phan et al., 2025, https://arxiv.org/abs/2410.09303), but its next-byte
factorization does not directly apply to out-of-order masked diffusion.
SweetSpan proposes candidate text spans and cross-scores them with each model
(Xu et al., 2025, https://arxiv.org/abs/2409.18583). UnMaskFork retokenizes
committed text when switching between masked diffusion models, including Dream
and LLaDA (Misaki and Akiba, 2026, https://arxiv.org/abs/2602.04344), but does
not solve probability-level fusion. The most promising CTCA adaptation is
therefore native-tokenizer candidate generation followed by mutual scoring of
the same completed text on each model's masked canvas. This is a research
direction, not a validated decoder change; first test candidate coverage,
score calibration, slot feasibility, and compute cost offline before a new
GSM8K run.

### Native text-span rescoring diagnostic

The diagnostic `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/native_span_rescore.py` reads the completed
four-question replay and reconstructs each saved master canvas from its prompt
and preceding commits. At six checkpoints across docs 23, 18, 9, and 11, it
collects the top eight native candidates from each model, deduplicates their
decoded text, and retokenizes each candidate in both native vocabularies. It
reports exact round-trip representability and whether all required master
slots remain free. For feasible candidates, each model scores its native token
sequence at the masked positions on its own canvas; the report uses mean log
probability within each model and an equal-weight mean across models. This is a
parallel masked-token pseudo-score, not an exact joint span probability. It
does not change commitment timing or perform decoding.

The approved study ID is `ctca-gsm8k4-native-span-rescore-20261030`. The
launcher is `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_native_span_rescore_slurm.sh STUDY_ID`; it
excludes gpu-38, gpu-51, and gpu-24. Its GPU output is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/STUDY_ID/rescore.json`. The CPU preparation output is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/STUDY_ID/prepared.json`. CPU validation found 69/76 candidate strings
feasible across six checkpoints. In doc 23, ` centimeters` is available at
step 0.0 with Dream IDs `[2889, 54236]` and LLaDA ID `[88161]`, but its second
Dream slot is occupied by step 0.2. These checks establish candidate coverage
and slot feasibility only; they do not measure an accuracy gain.
The saved trace contains committed token text rather than token IDs, so the
reconstruction re-encodes each committed token as one native master token.
The CPU validation passed for these six checkpoints, but the reconstruction
cannot guarantee original IDs when distinct IDs decode to the same text.
Compare the scored rankings with the original selected tokens and native
top candidates before considering a decoder prototype.

Job 354090 completed on gpu-17 and wrote `rescore.json`. The maximum absolute
gap between a replayed native master top-token probability and its saved trace
value is 1.7e-8 across the six checkpoints, supporting faithful canvas
reconstruction for these examples. The per-checkpoint model-forward time was
roughly 0.5–1.0 seconds for both models combined, excluding loading and canvas
work. The measured rankings are diagnostic, not accuracy results:

- Doc 23 step 0.0, position 69: ` centimeters` ranks first by mean token log
  probability across models (-2.049), ahead of `:` (-2.335). Its Dream native
  representation has two tokens, however. Using each model's **sum** of token
  log probabilities instead ranks `:` above ` centimeters` (-2.335 versus
  -3.773), so the ranking is sensitive to length normalization. At this step
  the actual scheduler selected positions 64, 67, 70, and 71; position 69 was
  eighth by original fused confidence, and position 70 committed ` every`.
  By step 0.2, when position 69 committed the projected ` kilometers`, the
  correct two-slot candidate was infeasible.
- Doc 11 step 1.3, position 541: Dream's top candidate was `.` followed by
  newline as one Dream token with probability 0.9863. LLaDA needs two tokens
  for that text, but position 542
  had committed a two-space token at step 1.2. Thus the candidate was
  infeasible; the best feasible candidate was newline, supported strongly by
  LLaDA and weakly by Dream. This is a second, independent instance of
  commitment blocking a cross-tokenizer span.
- Doc 18 step 0.0: the erroneous `3` ranks first under native rescoring.
  Doc 9 step 1.0: the repeated ` overtime` ranks first, only slightly above
  ` pay` (-2.210 versus -2.272). Doc 11 step 0.0 keeps the original `To`.

Native rescoring therefore validates a way to compare feasible strings across
tokenizers, but it does not by itself repair the observed decoding errors.
The next investigation should test a variable-width candidate selection rule
that reserves all slots of a chosen surface span while retaining the same
per-step token commitment budget, and evaluate calibration across both losses
and ensemble gains. A semantic completion such as ` traj` to ` trajectory`
is outside exact-surface matching and requires explicit candidate generation.

### Schedule-preserving span reservation audit

The approved CPU study ID `ctca-gsm8k4-span-budget-20261030` runs
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/span_reservation_audit.py` on the four saved ensemble traces.
It proposes protection for a top auxiliary string with probability at least
0.4 when that string exactly retokenizes to two or three currently free master
slots. A selector excludes protected slots from that step's rank list and
fills the same commitment budget from the highest remaining positions. It
rejects a proposed protection if too few positions remain. The output is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-span-budget-20261030/reservation_audit.json`. This is an
offline counterfactual: after the first changed commitment, later recorded
states do not represent the protected trajectory.

On the recorded states, doc 23 has four candidate protections, two of which
would displace a commitment. At step 0.0, protection of positions 69–70 for
` centimeters` would replace the commitment of ` every` at position 70 with
`2` at position 68 while still committing four positions. Doc 11 has 30
proposals across 25 steps; 29 fit the schedule and 10 original commitments
would be displaced after enforcing budget feasibility. At step 1.2 it would
protect positions 541–542 for Dream's `.` plus newline. Docs 18 and 9 have no
qualifying spans. These counts indicate that naive protection changes many
formatting decisions and cannot address the egg or overtime failures.
Increasing the threshold to 0.95 misses both early protections needed in
docs 23 and 11, so a simple confidence cutoff does not isolate the desired
cases. Protection alone also does not decide which text to commit later.
Do not add it to the live decoder until a calibrated span score and release
rule are tested.

The follow-up conditional score check adds model forwards after inserting the
earlier native tokens of a candidate span. It tests whether a parallel masked
product substantially understates multi-token candidates. The added
pre-blocking checkpoint is doc 11 step 1.2; doc 23 is scoreable at step 0.0,
but its second slot is already blocked by step 0.1. The score remains a
masked-diffusion pseudo-conditional rather than an exact joint probability.
The prepared ID is `ctca-gsm8k4-conditional-span-20261030`; the name was sent
to the user for selection. The CPU preparation passed for eight checkpoints
and wrote `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-conditional-span-20261030/prepared.json`.
The GPU diagnostic command was
`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_native_span_rescore_slurm.sh ctca-gsm8k4-conditional-span-20261030 --conditional`.
It wrote `rescore.json` under that study directory, including the
parallel and conditional scores, conditional rank, extra forward time, and
native master replay probability checks. The previously completed six-point
parallel-score study remains separate.

Job 354818 completed on gpu-62 and wrote the eight-checkpoint conditional
report. Saved versus replayed master top probabilities differ by at most
2.1e-8. In doc 23 step 0.0, ` centimeters` improves from parallel mean log
score -2.049 to conditional -1.033. Dream's conditional two-token product
probability is 0.05875 and LLaDA's one-token probability is 0.52220;
their equal-weight arithmetic event probability is 0.29048. At doc 11 step
1.2, Dream's `.` plus newline improves from -2.720 to -1.568 in mean log
score; LLaDA's conditional product is 0.01087 and Dream's
one-token probability is 0.41700, giving arithmetic event probability
0.21394. In both cases, the second native token becomes highly likely once
the first is inserted. Extra forward time was 0.23 seconds for doc 23 and
0.29 seconds for doc 11, on top of the original two model forwards.

These approximate event probabilities remain below the original single-position commit
cutoffs (about 0.356 in doc 23 step 0.0 and 0.652 in doc 11 step 1.2).
The mean-token score can rank a multi-token string higher, but this score
is sensitive to span length and is not directly comparable to the decoder's
single-token fused confidence. The erroneous one-token candidates `3` and
` overtime` remain first in docs 18 and 9. Do not treat the conditional
ranking alone as evidence that live span commitment would improve accuracy.

The next calibration set should include ensemble gains as well as losses.
The replay driver now accepts repeated `--case DOC:MASTER` arguments. A CPU
preparation of doc 3 with LLaDA master and doc 6 with Dream master passed
document identity and native prompt equality checks against the matching
endpoints. The proposed study ID is `ctca-gsm8k2-gain-replay-20261104`,
pending user name selection. Its command is
`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k2-gain-replay-20261104 --case 3:a --case 6:b`.
The wrapper has a one-hour allocation, following the current job limit.
After checking whether batch-one replay reproduces each 300-question output,
compare projection mismatches, candidate spans, commitment conflicts, and
conditional scores on both gains and losses before choosing a live decoder
rule. Retain the 512/128/32 schedule and four-token step budget in that rule.

The user submitted the gain replay as job 367862. It stopped before generating
the first trace: LLaDA model construction on `cuda:0` raised CUDA out of
memory in `nn.Embedding`. The CUDA preflight had confirmed two A100 40 GiB
devices but had not recorded free memory, so the log cannot establish whether
another process occupied the allocated GPU or model loading had a transient
peak. The job wrote no replay manifest, and doc 6 was not attempted. The
replay wrapper now logs the Slurm node and requires at least 30 GiB free on
each visible GPU before model loading. The same study ID can be retried;
each submission writes into its own Slurm job directory.

Retry job 367881 landed on gpu-41 and failed in CUDA preflight before model
loading: `torch.cuda.mem_get_info(0)` raised CUDA out of memory, despite two
A100 devices being visible. No gain trace or manifest was produced. The
replay launcher now excludes gpu-41 as well as gpu-38, gpu-51, and gpu-24,
prints node GPU memory through `nvidia-smi`, and reports PyTorch memory-query
failures explicitly. The next retry should use the same study ID and case
arguments; its separate Slurm job directory preserves both failed attempts.

Job 367906 completed on gpu-59. Its preflight recorded about 39.08 GiB free
on each assigned A100 and all four traces were written. The doc 3 LLaDA-master
ensemble and doc 6 Dream-master ensemble replays each match their original
300-question response exactly and give the correct numerical answer (540 and
260). Neither endpoint replay matches its original benchmark response: the
original endpoint answers were 180 and 2260, whereas both batch-one endpoint
replays give the correct numbers 540 and 260. Thus the saved ensemble traces
describe real benchmark gain trajectories, but the paired endpoint traces
cannot establish what changed the endpoint trajectory.

The CPU span audit of these gain ensemble traces found one selected high
confidence non-special projection mismatch in each. Doc 3 step 0.7 has Dream
`.\n` (probability 0.979) projected to LLaDA newline, while LLaDA commits
period; the exact two-token `.` plus newline remains possible at that step.
Doc 6 step 0.6 has LLaDA ` Toul` (probability 0.994) projected to Dream
`stras`, and Dream commits `stras` into the malformed city name
`strasouse`; its exact Dream spelling is ` T` plus `oul` across two free
slots. Both examples show projection defects even in numerically correct
ensemble responses.

The offline reservation audit for the gain traces proposes three protections
in doc 3; none would displace an original commitment, and the last is
rejected to preserve the four-token budget. It proposes one protection in doc
6, for ` Toul`, which would displace two original commitments and substitute
two others. That counterfactual might improve the city spelling, but it
changes a successful trajectory and needs a live replay before any quality
claim. The audit report is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-gain-replay-20261104/reservation_audit.json`.

CPU preparation for a conditional score check on the gain traces passed at
doc 3 step 0.7 position 500 and doc 6 step 0.6 position 88. Dream's `.\n`
round-trips as LLaDA IDs `[13, 198]` and Dream ID `[624]`; LLaDA's ` Toul`
round-trips as Dream IDs `[350, 10965]` and LLaDA ID `[122895]`. All required
slots are free at those checkpoints. The prepared report is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-gain-replay-20261104/prepared.json`. The GPU check can
reuse this study ID and run the native scorer with `--source-study-id`
set to the same gain study, two explicit `--checkpoint` arguments, and
`--conditional`. This compares score behavior at successful trajectories
without generating a new answer or changing the decoder.
The command is
`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_native_span_rescore_slurm.sh ctca-gsm8k2-gain-replay-20261104 --source-study-id ctca-gsm8k2-gain-replay-20261104 --checkpoint 3:0.7:500 --checkpoint 6:0.6:88 --conditional`.
The scorer launcher has a one-hour limit, checks for 30 GiB free per GPU, and
excludes gpu-24, gpu-38, gpu-41, and gpu-51.

### Balanced ensemble trajectory sample

The approved study ID `ctca-gsm8k4-balanced-ensembletrace-20261104` adds four
ensemble-only full traces from the same GSM8K-300 benchmark protocol. Gains
are doc 13 with LLaDA master and doc 15 with Dream master; losses are doc 5
with LLaDA master and doc 19 with Dream master. The original benchmark
endpoint outcomes identify gain or loss, while the replay reproduction flag
determines whether each new batch-one trace can explain its benchmark
ensemble trajectory. The driver accepts `--condition ensemble` so the study
does not spend GPU time on endpoint replays that failed to reproduce in the
previous gain study. CPU preparation checked document identity and native
prompt equality for all four cases. Two independent two-case Slurm jobs may
run concurrently under this study ID; each writes its own job directory and
different document manifests. The launcher has a one-hour limit and excludes
gpu-24, gpu-38, gpu-41, and gpu-51. The full-trace viewer accepts single-trace
replay manifests for these cases.

Gain pair command:
`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-balanced-ensembletrace-20261104 --case 13:a --case 15:b --condition ensemble`.
Loss pair command:
`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-balanced-ensembletrace-20261104 --case 5:a --case 19:b --condition ensemble`.
After completion, compare the two loss and two gain ensemble traces with the
existing four loss and two gain traces. Check exact replay reproduction first,
then count high-confidence projection mismatches, blocked exact spans, and
counterfactual reservation conflicts. Use conditional native scoring only for
positions where a multi-token candidate is feasible and decision-relevant.

### Distribution-level relative-space projection: `ctca-gsm8k4-relative-distribution-20261104`

Approved projection-only comparison. The new optional
`ctca_projection_mode=relative_distribution` leaves the existing `sparse_topk`
and `exact` modes and their defaults intact. It represents each ordinary token
by a softmax-normalized vector of cosine similarities to shared-token anchors.
For each already spatially warped auxiliary distribution, it averages these
vectors by auxiliary probability, retrieves up to 512 candidate master tokens
(global profile matches plus neighbors of up to eight source tokens carrying at
least 5% of residual probability), and runs 20 multiplicative mixture updates
to match that average. Residual mass at or below `1e-4` is omitted and the
output renormalized. The master
distribution is not an input to this projection; the existing CTCA probability
fusion and four-slot schedule remain unchanged. `exact_shared` still copies
ordinary shared-token mass directly. The new profile cache is separate from
the existing sparse mapping cache. The 2048-anchor pilot set that count
explicitly for both methods. After the pilot, `num_anchors=auto` was changed to
select the full shared-anchor set. For more than 2048 anchors, the distribution
projector now builds half-precision profile tables in vocabulary chunks, stores
them as memory-mapped `.npy` files in the CTCA cache, and streams chunks during
projection; candidate fitting processes positions individually. This removes
the full-table GPU memory requirement but creates roughly 30 GiB of cache for
the LLaDA/Dream pair and has not yet been timed on a GPU. It preserves the
whole-distribution calculation and all-anchor search. A small case matched the
dense projector numerically, including cache reload, and the focused CTCA
tests passed (41/41). This is a projection diagnostic, not a span scorer.

Use reproducible GSM8K gains (doc 3, LLaDA master; doc 6, Dream master) and
losses (doc 11, LLaDA master; doc 23, Dream master). The replay driver accepts
`--projection-mode` and `--num-anchors` and saves mode-specific manifests so
both conditions can coexist. The following commands are for the user to run;
the agent must not submit jobs:

`sbatch --time=03:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 3:a --case 6:b --condition ensemble --projection-mode sparse_topk --num-anchors 2048`

`sbatch --time=03:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 3:a --case 6:b --condition ensemble --projection-mode relative_distribution --num-anchors 2048`

`sbatch --time=03:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 11:a --case 23:b --condition ensemble --projection-mode sparse_topk --num-anchors 2048`

`sbatch --time=03:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 11:a --case 23:b --condition ensemble --projection-mode relative_distribution --num-anchors 2048`

Inspect `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-relative-distribution-20261104/doc<DOC>-<MODE>-2048.json`
in the full-trace viewer. Compare known wrong projection events, selected
tokens, full response correctness, per-position inverse-fit KL and top-source
coverage, and wall time. Run `TSE_TIMERS=1` in a diagnostic job if detailed
projection timing is needed. Only proceed to matched GSM8K-300 jobs after the
trace cost and stability are acceptable; the native endpoints and CTCA
schedule should remain fixed across that comparison.

### Old sparse projection temperature sweep: `ctca-gsm8k4-anchor-projection-temp-sweep-20261104`

The user approved this study ID for hyperparameter tuning of the existing
`sparse_topk` relative-anchor projector, separate from the new distribution
projector. Four selected ensemble checkpoints include two losses (doc 23
Dream-master ` centimeters` -> ` kilometers` at step 0.2, position 69; doc 11
LLaDA-master `.` plus newline at step 1.3, position 541) and two gains (doc 3
LLaDA-master `.` plus newline at step 0.7, position 500; doc 6 Dream-master
` Toul` -> `stras` at step 0.6, position 88).

The CPU script `/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/projection_temperature_sweep.py`
reweights the existing cached top-64 rows at projection temperatures 0.01,
0.025, 0.05, 0.1, and 0.2. It combines the top spatial auxiliary token with
the saved top-64 master probabilities. Its report is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-anchor-projection-temp-sweep-20261104/cached_projection_temperature.json`.
The result is a local counterfactual only: omitted auxiliary and master mass
is bounded in the report, but other source tokens, position selection,
subsequent decoding, and a new full-vocabulary top-k search are not replayed.

At the centimeter event, the estimated fused winner stays ` kilometers`
through temperature 0.1 and becomes ` cm` at 0.2. At the Toul event it stays
`stras` through 0.1 and becomes ` T` at 0.2. In both events the source token's
highest-weight destination remains the wrong one; the warmer projection only
reduces its dominance enough for a different fused decision. Both punctuation
events retain their original estimated winner over this sweep. These results
do not yet show a better completed answer or a corrected cross-tokenizer map.

The CPU script
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/anchor_temperature_sweep.py`
loaded only cached embedding matrices, reconstructed the same 57,909 shared
anchors, and sweeps anchor temperatures 0.003, 0.01, 0.03, and 0.1. For each
checkpoint it compares the original top-64 master destinations plus explicit
same-text candidates, at projection temperatures 0.05, 0.1, and 0.2. The
baseline cell matched the cached top-64 weights within `2.3e-8` in all four
cases. This is a restricted-candidate diagnostic; different anchor
temperatures could introduce other top-64 destinations in a full rebuild.
The script wrote
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-anchor-projection-temp-sweep-20261104/restricted_anchor_temperature_grid.json`.
At anchor temperature 0.03, ` cent` moves from rank 65 to rank 13 among the
fixed centimeter candidates, but ` kilometers` still leads. Dream's ` T`
remains rank 65 for the Toul source and LLaDA's period remains rank 65 for
Dream's period-plus-newline source. At 0.1 the top mapped destinations for
the centimeter and Toul source tokens become unrelated tokens, indicating
that increasing anchor temperature indiscriminately can worsen alignment.
The local fused winner becomes ` cm` for the centimeter case at anchor 0.03
and projection 0.1 or 0.2, and ` T` for Toul at the same pairs. Both are
fragments or abbreviations, not evidence of correct completed text. No GPU
job was necessary. These local counterfactuals do not establish GSM8K accuracy
or the effect on a full decoding trajectory.

### Tuned sparse-anchor GSM8K-300 pair

The user approved two 300-question runs with the existing `sparse_topk`
projector at `ctca_anchor_temperature=0.03` and
`ctca_projection_temperature=0.10`:
`ctca-gsm8k300-lladamaster-at0p03-pt0p10-20261104` and
`ctca-gsm8k300-dreammaster-at0p03-pt0p10-20261104`. Their shared protocol ID
is `ctca-gsm8k300-dualmaster-sparse-at0p03-pt0p10-v1`. Both inherit their
respective earlier exact-shared, Dream-probability-temperature-1.0 GSM8K-300
configs and retain the same prompts, first 300 examples, batch size 8,
512/128/32 schedule, four commits per step, static 50/50 fusion, 64 projected
destinations, all 57,909 anchors, and exact-shared-token override. Changing
both temperatures together tests the combined configuration; it does not
identify which temperature caused an accuracy change. Baselines are the
previous matched runs, jobs `352346` (LLaDA master, 187/300 flexible EM) and
`352347` (Dream master, 194/300 flexible EM).

The new cache metadata differs from the prior cache in both temperatures, so
each master direction will rebuild its sparse projection table before
evaluation. A previous full-anchor table build took 1,423.6 seconds in job
`350507`; the two new jobs have 12-hour limits. Separate run IDs and cache
paths avoid reusing prior lm-eval responses. The Slurm wrappers exclude the
known faulty gpu-38 as well as gpu-24, gpu-41, and gpu-51. The agent must not
submit Slurm jobs. User commands:

`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_lladamaster_at0p03_pt0p10_slurm.sh`

`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_gsm8k300_dreammaster_at0p03_pt0p10_slurm.sh`

After completion, compare flexible and strict GSM8K exact match, per-question
gain/loss overlap, malformed-output counts, and cache-build timing against
the two baseline jobs. These benchmark jobs do not produce full traces; use
targeted replay only if a question-level difference needs explanation.

### Results of jobs 368499, 368508, and 368509

Job `368499` completed the conditional native-span rescore for the two
reproduced ensemble gain checkpoints in job `367906`. Its report is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-gain-replay-20261104/rescore.json`.
The replayed master top probabilities match the saved trace within `3e-8`.
At doc 3 step 0.7 position 500, the shared-text candidate `.` plus newline
(Dream one token, LLaDA two) ranks first among 13 candidates under the
conditional mean-log score (`-0.658`), ahead of the single period (`-2.448`).
At doc 6 step 0.6 position 88, ` Toul` (LLaDA one token, Dream two) ranks
first among 11 (`-2.094`), while ` T` ranks fourth (`-5.154`). These rankings
show that native-tokenizer span scoring can recover same-text candidates that
the current projection missed. They are not calibrated commit scores and do
not establish that changing those successful benchmark trajectories improves
the final answer. The Dream conditional joint log probability of ` Toul` is
lower than its parallel pseudo-score, so score definition matters.

Jobs `368508` and `368509` both completed two ensemble-only full traces under
`ctca-gsm8k4-balanced-ensembletrace-20261104`. None of the four batch-one
replays reproduces its original batch-eight benchmark response; the first
text differences occur before trailing special-token padding. The originally
correct doc 13 (LLaDA master, gold 18) replay ends in repeated LaTeX slashes
without a final answer; originally correct doc 15 (Dream master, gold 125)
replay gives 2971. The originally incorrect doc 5 (LLaDA master, gold 64)
replay loops over glasses and truncates before an answer; originally incorrect
doc 19 (Dream master, gold 6) replay gives 1 rather than the benchmark's 3.
These traces are valid independent decoding examples, but they cannot explain
the specific GSM8K-300 gains or losses they were chosen to study.

A CPU-only span audit of the four new traces is stored at
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-balanced-ensembletrace-20261104/span_audit.json`.
Selected high-confidence non-special source tokens whose top projected text
differs occur eight times in doc 5, twice in doc 13, and zero times in docs
15 and 19. All ten are punctuation/newline multi-token spellings, with exact
master-token spans available in the saved top-64 candidates. The mismatches
do not by themselves explain arithmetic errors or final-answer quality; in
particular, neither doc 15 nor doc 19 has such a selected mismatch. The audit
uses parallel saved probabilities and cannot validate a live span-commit rule.

### One-example distribution-projection pilot

Before running the four-case `ctca-gsm8k4-relative-distribution-20261104`
study, start with GSM8K doc 23, Dream master. Its previous full-anchor
ensemble replay from job `352946` reproduced the benchmark response and
committed ` kilometers` when LLaDA's top auxiliary token was
` centimeters`. Run the new `relative_distribution` mode and the existing
`sparse_topk` control on the same saved native prompts, with 2048 shared
anchors, exact-shared-token policy, 512/128/32 decoding schedule,
`temperature_a=temperature_b=1.0`, and static 50/50 fusion. The reduced anchor
count is necessary for the new mode's dense profile matrices; it is identical
in both pilot conditions, so the comparison isolates projection method. It
does differ from the earlier 57,909-anchor benchmark replay. Both pilot
commands passed CPU preparation. The original doc-23 sampler took about
317 seconds; use the launcher's one-hour default for the sparse control and
two hours for the unmeasured distribution projector. The user must submit
the jobs:

`sbatch --time=02:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 23:b --condition ensemble --projection-mode relative_distribution --num-anchors 2048`

`sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 23:b --condition ensemble --projection-mode sparse_topk --num-anchors 2048`

Open explicit mode-specific viewer manifests at
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-relative-distribution-20261104/doc23-relative_distribution.json`
and
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-relative-distribution-20261104/doc23-sparse_topk.json`.
Check completion and runtime first, then compare both trajectories at the
first divergent commit and at the centimeter checkpoint, final-answer
extraction, malformed output, inverse-fit KL, and top-source coverage. One
example can validate feasibility and mechanism, not GSM8K accuracy.

Pilot result (jobs `368807` relative distribution and `368808` sparse top-k,
both on `gpu-47`): both completed without stderr errors. Sampling took
356.79 s and 345.71 s respectively. Both produced the same 512 committed
tokens and the same response: "The candle melts by 2 cm every hour. From
12:00 PM to 5:00 PM is 5 hours in total. So, the candle will melt by 2 cm*5 =
10 cm. The answer is 10." Gold is 8. The selected token sets are identical
by absolute position; only a few commit steps differ. Neither reproduces the
old 57,909-anchor benchmark response. At step 0.2, absolute position 69,
both 2048-anchor modes commit ` cm` where the old full-anchor replay committed
` kilometers`. The auxiliary top token is ` centimeters` (probability
0.999237); the Dream master assigns ` cm` probability 0.964483. Relative
distribution assigns projected ` cm` probability 0.238206 (fused 0.601344,
fit KL 0.586377, 256 candidates). Sparse top-k assigns it 0.135541 (fused
0.550012). Both improve the unit at this checkpoint, so the pilot does not
attribute that improvement specifically to the new method. The remaining
wrong answer comes from reading the question's 1:00 PM start as 12:00 PM.
The relative projector also gives some mass to unrelated strings (` perfil`,
`oured`) at this checkpoint; inspect its profile/candidate quality before
scaling. Trace viewer manifests are
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-relative-distribution-20261104/doc23-relative_distribution.json`
and
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k4-relative-distribution-20261104/doc23-sparse_topk.json`.

Full-anchor follow-up: replay doc 23 with `relative_distribution` and
`--num-anchors auto` using the chunked memory-mapped backend. The replay driver
adds an `-auto` suffix to this trace and manifest, preserving the 2048-anchor
pilot under the same approved study ID. This has passed CPU preparation; GPU
runtime and numerical output remain unmeasured. The user submits:

`sbatch --time=06:00:00 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_benchmark_replay_slurm.sh ctca-gsm8k4-relative-distribution-20261104 --case 23:b --condition ensemble --projection-mode relative_distribution --num-anchors auto`

Review preparation (2026-11-17): numeric anchor counts now appear in replay
trace, reproduction, and manifest filenames, so distinct counts coexist.
Earlier pilot artifacts may retain filenames without the `-2048` suffix.
The temperature sweep requires a unique full-anchor baseline cache at anchor
temperature 0.01, projection temperature 0.05, and top-k 64. The two approved
tuned confidence controls now have planned JSON ledger records; submission
and results have not been verified during this review. The tracker declares
requirements for their protocol and the HumanEval exact-shared protocol.
New numeric-anchor manifests use `doc<DOC>-<MODE>-<COUNT>.json`;
`auto` keeps its existing `-auto` suffix. Historical artifact links above
retain their recorded filenames.
