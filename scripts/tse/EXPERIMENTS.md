# TSE experiment tracking and evaluation

The source of truth is
[experiments.json](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/experiments.json).
It records the scientific run, while W&B is the dashboard and `.logs` stores raw
artifacts. Scheduler attempts belong under the scientific run rather than becoming
separate experiments. Experiments belong to exactly one of two research groups:
`tse` for same-tokenizer TSE-only studies, or `ctca` for cross-tokenizer CTCA and
the native controls needed to evaluate it.

To view the current ledger and resolve metrics from the result JSON files:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/track_experiments.py --ledger /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/experiments.json --check
python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/track_experiments.py --ledger /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/experiments.json --group ctca
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

### GSM8K300 CTCA agreement ranking design (2026-11-17)

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
and separate response caches. Jobs 415139 and 415140 completed. The paired
results appear in the CTCA results audit below.

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

### Research direction assessment (2026-11-17)

This assessment records proposed decision gates, not new registered experiments.
Study names, frozen configurations, and protocol IDs require user selection
before creating follow-up experiments. No scheduler commands were executed.

Read-only artifact inspection found completed tuned confidence controls that
the preceding review had not verified. Job 368727 (LLaDA master) reports
183/300 flexible EM, 61.00% with bootstrap SE 2.82 percentage points; job
368728 (Dream master) reports 197/300, 65.67% with SE 2.75 points. Their raw
results are, respectively:

- `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k300-dualmaster-sparse-at0p03-pt0p10-v1/ctca-gsm8k300-lladamaster-at0p03-pt0p10-20261104/368727/results_2026-11-04T22-17-37.438018.json`
- `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k300-dualmaster-sparse-at0p03-pt0p10-v1/ctca-gsm8k300-dreammaster-at0p03-pt0p10-20261104/368728/results_2026-11-04T21-55-03.697420.json`

These are descriptive comparisons with prior exact-shared controls (187/300
and 194/300); no new paired significance calculation or registry status
update was made. They do not establish a reliable temperature-tuning benefit.

A document-ID join of flexible-extract samples from endpoint jobs 350508
and 352165 found 300 matching document hashes: both correct 138, LLaDA-only
correct 71, Dream-only correct 46, and neither correct 45. The either-correct
count is 255/300 (85%). This is a hindsight selection bound for these saved
endpoint outputs, not an attainable method score or a bound on all possible
ensemble generations. Prompts and native schedules remain model-specific.

Recommended sequence: first establish reproducible matched controls and
probability-weighted lexical coverage; next isolate conservative exact-text
single-master-token fusion with master fallback for unmappable mass; then
consider selective short-span fusion only if native scoring and atomic span
commitment improve decision quality. Compare alternative completions of the
same region, preserve ordered token sequences, and explicitly account for
native mask counts, length calibration, and consumed slots. Existing native
rescoring diagnostics operate on existing canvases and are not evidence of
tokenizer-independent gap probabilities. Simple reservation alone and
first-token-only projection do not provide these guarantees.

Include compute-matched single-model extra sampling, output selection, and
diffusion trajectory relay as alternatives before claiming benefit from fine
probability fusion. Existing consensus-ranking studies should remain separate
from vocabulary-map interventions. Evaluate balanced gains and losses on a
development set, freeze the rule before held-out evaluation, and measure
forward calls, processed token counts, latency, and GPU-hours. Broaden to
dynamic gap canvases, learned scoring, or training-time alignment only if
the preceding diagnostics justify their complexity.

### CPU gate study: `ctca-toy4-mastergate-20261118`

User-selected study name; implemented, execution pending. This is a synthetic
CPU diagnostic, not a GPU benchmark or production sampler change. The script
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_master_gate_sandbox.py`
reuses project entropy, equal-weight JS, fusion, and fixed-budget selection.
It enumerates 286 four-token distributions (increments of 0.1), all 81,796
ordered pairs, and identity, swapped, and collapsed vocabulary maps.
It compares normalized JS and probability overlap agreement, using an
agreement-only gate, a both-confident gate, and an at-least-one-confident gate.
Agreement thresholds are 0:0.05:1 and normalized-entropy thresholds 0:0.1:1;
equality accepts. Accepted rows use a 50/50 blend, rejected rows use the master.
Argmax uses the lowest token index; tied labelled predictions are excluded
from correctness comparisons. Synthetic labels are illustrative, not sampled
model accuracy. Native and projected auxiliary entropy are reported separately.

Outputs: `report.json`, `thresholds.csv` (2,898 aggregate rows), and three
mapping summaries under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-toy4-mastergate-20261118`.
No threshold recommendation is inferred. Tests cover score extremes, two
uncertain models, asymmetric uncertainty, threshold equality, normalization,
empty inputs, invalid thresholds, ties, projection-induced confidence,
deterministic examples, fixed commit counts, and artifact serialization.

The test launch was attempted but stopped before Python because
`/home/juan.guevara/.zshrc` is absent in the agent workspace. Neither tests nor
the full study have executed here. `git diff --check` passed. Run on the
configured machine:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
python -m pytest /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_master_gate.py -q
python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_master_gate_sandbox.py
```

No scheduler submission or checkpoint loading is involved. Benchmark registry
entries and production configs remain unchanged; this CPU diagnostic is tracked
here rather than assigned a GSM8K benchmark protocol.

#### Version 2: balanced correlated cases and corrected canvas demonstration

The user ran the initial sandbox; its original artifacts remain unchanged.
Review found that the canvas example accepted every position, the independent
simplex-pair grid was permutation-invariant, and labelled collapse examples
all shared answer token 0. Version 2 corrects the canvas example to reject its
first position and change both its proposal and the selected positions while
retaining two commits. Its thresholds are explicitly recorded (agreement
0.75, uncertainty 1); these are demonstrative, not tuned recommendations.

The six labelled templates now each undergo all 24 token permutations before
the fixed projection corruption is applied: 144 paired cases, 36 answers per
token, including six answers per token within each template. Each case keeps
the same answer and native predictions across mapping conditions. Equal
template weighting is a controlled stress test, not realistic error frequency.
The independent full-grid statistics remain as unlabelled sanity checks only.
Canonical examples remain illustrative; threshold CSV correctness counts use
the balanced cases. Collapse to token 0 is now correct in only 36/144 cases.

Threshold rows additionally report comparable cases, excluded ties, master,
ungated and gated correct counts, harmful/helpful overrides, prevented harms,
and rejected useful corrections. All policies share the same eligibility mask
(master and ungated blend must both have unique maxima), avoiding denominators
that change with the gate. Tests assert the balanced labels, expected identity
case accounting, canvas rejection and changed selections, and report version.

The revised default output is
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-toy4-mastergate-20261118/v2`,
preserving the user's original outputs. Run the same script and pytest commands
above to generate and validate v2. The agent's test attempt again stopped
before Python because `/home/juan.guevara/.zshrc` is absent; runtime results
remain unverified here. Static whitespace checks passed. Production CTCA is
unchanged; no new experiment identity or GPU submission was introduced.

### Real-trace gate audit: `ctca-gsm8k6-offline-mastergate-20261118`

User selected this name. CPU implementation is prepared; execution is pending.
Driver:
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/offline_master_gate.py`.
It streams ensemble traces from the four-case benchmark replay (docs 23, 18,
9, 11; historical losses) and two-case gain replay (docs 3, 6). Per-case
manifest paths and ensemble reproduction flags are preserved. Endpoint replay
reproduction is not assumed, and historical answer gains/losses do not label
individual token decisions. These are previously inspected exploratory cases,
not a held-out evaluation set.

Frozen diagnostic grid: equal-weight normalized JS and probability overlap;
agreement thresholds 0.4, 0.5, 0.7; agreement-only, both-confident and
at-least-one-confident rules; normalized-entropy thresholds 0.5 and 0.8 for
the latter two rules. Accepted positions retain the recorded ensemble proposal
and rejected positions use the recorded master top token. Production
fusion weights and selected-position flags are retained as provenance.

Saved top-64 distributions cannot provide exact full-vocabulary entropy or
arbitrary agreement. The audit bounds unknown-tail entropy between a single
tail token and a uniform tail. JS uses the known support union plus a coarse
tail bin as its lower divergence bound; its upper divergence bound adds the
maximum conditional JS inside that tail. Overlap bounds use known overlap
plus at most the smaller remaining mass. Gates are marked ensemble/master
only if their entire intervals imply that decision, otherwise ambiguous.
Duplicate decoded support strings or invalid saved mass are marked unusable,
not silently merged. No tail is discarded or renormalized.

Position JSONL reports retain native-dominant-contributor, spatial and
projected auxiliary uncertainty, master uncertainty, source and projected
top text, saved selection, and confidence change under fallback. Contributor
entropy is not the complete spatial mixture; native and master vocabulary
normalizers differ. Aggregate gate counters report ambiguity, unusable rows,
proposal changes, selected-position changes, and high-confidence surface
mismatch coverage. Surface mismatch is not a correctness annotation.
The audit does not rerank commitments, apply a new suppression policy, or
claim a changed answer. Those require a fully specified decoder and live replay.

Outputs go under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k6-offline-mastergate-20261118`.
Pure-standard-library tests cover full-versus-truncated bounds, disjoint and
identical support, ambiguous gates, malformed probabilities, decoded-string
collisions, and streaming step/selection provenance. The local test command
stopped before Python because `/home/juan.guevara/.zshrc` is absent; static
whitespace checks passed. User commands on the configured machine:

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
python -m unittest discover -s /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests -p test_offline_master_gate.py
python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/offline_master_gate.py
```

No scheduler jobs, model forwards, production sampler changes, or benchmark
registry entries are involved in this offline diagnostic.

#### Offline audit coverage correction

The user generated the six reports. At doc 23, step 0.2, master position 69,
the recorded master proposes ` cm` at 0.964483 while fusion proposes
` kilometers` at 0.499612. Equal-weight JS agreement is bounded by
[0.00433769, 0.00435612], so every proposed agreement threshold rejects
fusion locally. This does not establish a corrected final response.

Review also found an overly conservative audit bug: duplicate decoded labels
caused 2,087/2,304 doc-23 positions to be discarded. Inspection of
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/smoke.py` confirms that
`fusion_support` constructs its rows from unique token IDs, with exact paired
probabilities. Distinct rows therefore remain distinct coordinates even when
their decoded text coincides. Audit version 2 preserves them separately and
reports duplicate label counts; the regression test compares identical-label
and distinct-label representations. Invalid mass still causes explicit
unusable rows. The earlier duplicate-label exclusions and resulting aggregate
gate counts are superseded and require rerunning the audit. Native/master
top-token labels remain text-only; the audit does not claim ID-level equality
from identical text. No production change was made.

### Paired live CTCA pilot: `ctca-gsm8k2-live-one-step-ablation-20261118`

User-selected study name. The pilot is ready for submission. It uses GSM8K
doc 1 with LLaDA as master and doc 2 with Dream as master, selected by ID
and master role. Both native prompts come from the matching benchmark inputs
in jobs 352346 and 352347. Historical model answers, correctness, and gold
are excluded from deciding where to intervene. The gold is used for scoring
the finished branches via the GSM8K flexible numeric extraction rule.

The greedy master baseline observes the first live step where a valid 50/50
CTCA proposal differs at an already scheduled master position. The pilot
replays the prompt twice. The no-op run retains master IDs and must reproduce
every canvas. The intervention run must match the complete prefix trajectory
and selected positions, then substitutes fused IDs once. Both paths follow
fresh forwards and the same master policy thereafter. The fixed schedule is
512 generated slots, 128 steps, 32-slot blocks, seed 1234, batch size one,
BF16, and model probability temperatures 1.0. It uses the existing exact-shared
sparse projector (all anchors, top 64, temperatures 0.01 and 0.05) and the
existing master suppression settings. The intervention preserves all selected
positions and excludes fused mask IDs and begin-suppressed substitutes.
If no admissible disagreement occurs, the report records an unpaired case.

This is a two-case feasibility pilot for estimating the effect of one forced
fusion step under master continuation. An every-step gate and benchmark-wide
gain require subsequent held-out evaluation. CPU preparation and all 20 CTCA
sampler tests passed, including a regression where a changed commit changes
the next live prediction. The GPU path remains unrun. Single-device preflight
requires at least 36 GiB free. Driver:
`/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/live_one_step_ablation.py`.

```bash
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
srun -p "$PARTITION" -q="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python -u /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/live_one_step_ablation.py
```

Batch submission:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_live_one_step_ablation_slurm.sh
```

The run prints its report path under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-live-one-step-ablation-20261118`.
Inspect the no-op and shared-prefix flags before reading the two finished
answers, then compare `correctness_delta` (+1, 0, or -1). No job has been
submitted by the agent. The pilot has no lm-eval result registry entry.

User-submitted job 421810 on gpu-62 hit the three-hour limit on 2026-11-20.
It produced empty stdout and no report. Slurm also reported that the step did
not exit on termination signals. The stalled stage and root cause are unknown.
The launcher now records shell startup and uses the cached checkpoints by
default (`HF_HUB_OFFLINE=1`), matching the native span launcher. The Python
driver logs imports, memory check, model loading, projection setup, and each
replay branch to distinguish startup and decoding failures on retry.

Retry job 422949 on gpu-47 completed both cases. Both no-op trajectories,
pre-intervention trajectories, and expected post-intervention canvases
matched exactly. LLaDA master (doc 1) replaced ` Since` with ` Half` at
block 0, step 4, position 477. Both branches answered 3 correctly. Dream
master (doc 2) replaced ` is` with ` was` at block 0, step 7, position 85.
The master answered 65,000 and the fusion branch answered 1,820,000, both
incorrect against the benchmark target of 70,000. Both correctness deltas
were zero. The changed reasoning in both continuations verifies a live
effect from the intervention under the master continuation policy. These
two examples do not establish a benefit from fusion or a predictive signal
for routing. Raw report:
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k2-live-one-step-ablation-20261118/422949/report.json`.

### CTCA results audit (2026-11-20)

The first 300 GSM8K questions were joined by document ID, document hash, and
target hash. All 300 prompt hashes matched within each master direction.
Saved scientific arguments in the tuned confidence and JS runs show the
same checkpoints, prompts, batch size, seeds, 512/128/32 schedule, 50/50
fusion, and projection settings. The declared intervention was position
ranking. The jobs ran from different dirty checkouts (base commits
`0f79748` and `3e85e18`). LLaDA-master runs also used different hosts.
The comparisons are retrospective. A controlled reproduction needs one
pinned checkout, hardware configuration, set of model revisions, and cache
configuration.

| Master | Confidence job, flexible / strict EM | JS job, flexible / strict EM | JS gains / losses on flexible EM | Paired flexible Δ (95% bootstrap interval) |
|---|---|---|---:|---:|
| LLaDA | 368727, 61.00% ± 2.82 / 46.67% ± 2.89 | 415139, 57.00% ± 2.86 / 45.33% ± 2.88 | 44 / 56 | −4.00 pp [−10.67, +2.33] |
| Dream | 368728, 65.67% ± 2.75 / 49.33% ± 2.89 | 415140, 58.00% ± 2.85 / 45.00% ± 2.88 | 37 / 60 | −7.67 pp [−14.00, −1.33] |

Scores carry lm-eval standard errors. Intervals use 100,000 paired question
resamples. Exact two-sided McNemar p-values are 0.271 and 0.025, respectively,
before any multiple-comparison adjustment. Flexible extraction failures
were 2 → 2 and 0 → 2. The net Dream loss therefore exceeds the change in
extraction failures. Lm-eval wall times were 5.21 h → 5.33 h for LLaDA and
4.84 h → 4.31 h for Dream. These timings include setup and generation and
span separate runs. Tuned projection versus the older exact-shared confidence
controls (352346 and 352347) changed only eight and five flexible correctness
labels respectively, yielding −4 and +3 net correct answers.

The best audited ensemble result, Dream-master confidence at 197/300, is
below the historic LLaDA CTCA endpoint at 209/300. Within master direction,
the tuned confidence results are 183 vs 209 for LLaDA and 197 vs 184 for
Dream. These endpoints come from different protocol records. Treat them as
descriptive context. The two endpoints' hindsight selector reaches 255/300
and uses five-shot LLaDA and zero-shot Dream outputs.

The four HumanEval exact-shared jobs scored all 164 problems. The same
document, target, and prompt hash checks passed within each master direction.

| Master | CTCA endpoint pass@1 | 50/50 CTCA pass@1 | Gains / losses |
|---|---:|---:|---:|
| LLaDA | 32/164, 19.51% ± 3.10 (369396) | 21/164, 12.80% ± 2.62 (369394) | 6 / 17 |
| Dream | 32/164, 19.51% ± 3.10 (369397) | 29/164, 17.68% ± 2.99 (369395) | 19 / 22 |

Registry completion is backed by saved results and sample files for all eight
jobs in the two tables. The two earlier JS attempts, 414544 and 414545, have
startup logs and no saved results. Their terminal scheduler status is unknown.
The tracker validates all eight audited records when run on that subset.
Full-ledger validation also encounters older trace-only entries whose JSONL
is parsed as a single JSON result.

The saved version-2 six-trace gate audit contains 13,824 active-position
observations. A JS agreement-only threshold of 0.4 would fall back at 9,343
observations, but changes only 24 of the 3,072 saved selected text proposals
when their positions are held fixed. Four of the six ensemble replays
matched the original saved benchmark responses. The
audit measures proposals on the observed canvases. The live two-case probe
checks continuation under one actual substitution, with zero answer
correctness changes in both cases. Together they leave open whether a gate
predicts useful overrides or merely changes the number of them.

The proposed fixed-budget gap intervention and its controls are described in
the [ensemble research plan](/home/juan.guevara/naruto/dllm-epiplexity/experiments/ENSEMBLE_RESEARCH_PLAN.md).
The proposed design is pending study naming and registration.

### Accuracy-first follow-up

Prioritize a matched TIE benchmark on GSM8K. Include independent native
generations with TIE's final selection rule and periodic intermediate
trajectory relay. Independent generation followed by selection is already
TIE's post-generation condition. A verifier based on masked reconstruction
with an LLaDA fallback can serve as an alternative selector in that condition
if its threshold is calibrated on development data and frozen before
held-out evaluation. [TIE, Sections 4.3–5.1](https://arxiv.org/html/2606.16281v1#S4.SS3),
[Bidirectional consistency](https://proceedings.mlr.press/v306/ruan26a.html)

On the saved endpoints, switching to Dream can recover 46 LLaDA failures and
can lose 71 LLaDA successes. These are hindsight counts from the development
slice. A selector's paired gain over LLaDA equals recovered failures minus
introduced failures. For example, 20 recoveries and five introduced failures
would improve the score by 15/300, or 5 percentage points. Track those counts
at every selector threshold and report the cost of verification forwards.

Compare both TIE conditions with the two true native models and a
stronger-model multi-sample control. Include scoring and verification
forwards in the matched inference budget. Use untouched GSM8K questions for
the primary paired flexible exact-match comparison. The proposed local-span
method then tests whether integrating a verified completion of one gap into
the current canvas improves accuracy beyond whole-trajectory relay. This is
a follow-up plan pending study naming.

#### Run order and decision criteria

A public-code search on 2026-11-20 found references to TIE in paper indexes
and reading lists. GitHub repository searches for the paper title, method
name, and arXiv ID found no implementation. The arXiv, OpenReview PDF, and
author-submitted Hugging Face record yielded no implementation link. Treat
the planned benchmark as an independent implementation of the published
algorithm and document the cross-tokenizer state transfer policy.
[arXiv paper](https://arxiv.org/abs/2606.16281),
[OpenReview PDF](https://openreview.net/pdf?id=DA2pElNUO2),
[Hugging Face paper record](https://huggingface.co/papers/2606.16281)

1. **Validate the TIE implementation on 32 development questions.** Use the
   current LLaDA-8B-Instruct and Dream-7B-Instruct checkpoints. Start with a
   256-generated-slot canvas, 16-slot blocks, and a relay interval of 16
   denoising steps, following TIE's GSM8K settings. Keep an answer region
   masked until reasoning finishes and use normalized answer-token change
   count. Freeze the total forward budget and answer-region size in the
   named protocol. Reuse the canvas retokenization utilities for the relay
   and test both directions. Check that transfer preserves committed text,
   unresolved gap order and the answer boundary. Require no-relay parity
   with independent generation under the same schedule and scoring scaffold.
   Report the tokenizer transfer policy as an adaptation. [TIE setup](https://arxiv.org/html/2606.16281v1#S5.SS1),
   [TIE answer positions](https://arxiv.org/html/2606.16281v1#A1.SS4),
   [canvas utility](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/ctca/canvas.py:212)
2. **Compare methods on GSM8K.** Use the first 300 questions for development
   and freeze settings before evaluating on the remaining 1,019 questions.
   Compare independent LLaDA and Dream, TIE final selection over those same
   trajectories, periodic TIE relay, and additional samples from the
   stronger single model selected by the same scoring rule under the
   ensemble's compute budget. Give each checkpoint the same prompt and
   generation settings across matched arms. Also measure the strongest
   native decoding setup without TIE's answer scaffold so that the
   scaffold's effect is visible. Primary outcome is paired flexible exact
   match against the strongest native baseline, with a paired 95% interval
   and gains and losses. Report the margin over the compute-matched control
   with forward counts, processed tokens, wall time and GPU-hours.
3. **Test local span verification on development states after establishing
   TIE as a reference.** From the same pre-commit canvas, compare keeping
   the master proposal, inserting one native-tokenizer-verified span, and
   taking the whole donor trajectory. Give each branch the same remaining
   decode budget. Include master-only span scoring with the same verification
   budget. Test whether the cross-model verifier predicts a positive net
   effect from local substitutions before running it across a full
   benchmark. Freeze the resulting method and evaluate the accuracy claim
   on a reserved comparison split or a second reasoning benchmark if the
   first held-out GSM8K results informed method changes.

#### Standalone TIE implementation

The [TIE sampler](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tie/sampler.py)
runs independent greedy low-confidence decoding between relays and uses
normalized answer-token change count. Dream uses its native right-shifted
logits. The default budget is 256 masked generation slots, including 16
answer slots, and up to 128 denoising forwards per model. The fixed answer
marker uses additional tokens. Answer slots remain masked until reasoning
finishes. The runner evaluates one question at a time and forwards members
sequentially.

The transfer policy retokenizes committed runs within logical blocks,
preserves the count and order of each unresolved mask run, and converts
terminal markers to the receiver's EOS. Native canvas length can change.
The relay is skipped if text fails round-trip verification or the result
exceeds a model's context limit. A successful relay resets the interval
scores and gives both branches the donor's accumulated trajectory score.

The GSM8K runner applies each model's chat template internally. Submit the
32-question validation with the Slurm wrapper:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/tie_gsm8k32_pilot_slurm.sh
```

Results and relay traces go under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/tie-gsm8k32/<job-id>/`.
Use `mode=post_generation`, `mode=llada`, and `mode=dream` in the eval entry point for the
matched controls on the same answer scaffold. The trace records every relay
decision and complete before/after canvases for the first successful relay
from each donor observed in the run. It also records member completions,
forward counts, and processed tokens. Check observed leader coverage before
reporting a live result in both transfer directions.

#### First local-span comparison after TIE

Start with a proposal-only intervention under the receiver's existing commit
schedule. At a predeclared checkpoint, choose one contiguous gap of one or
two slots that the receiver already plans to fill. Obtain both proposals
from the same visible partial solution, rendered with each model's native
tokenizer. Require exact in-context round trips and equal candidate lengths
within each tokenizer. The two tokenizers can use different lengths. Keep
the receiver's proposal when the candidate cannot fit its scheduled slots.

For each model, score the auxiliary and receiver completions with the same
within-gap reveal order. Use the difference in conditional log scores as a
candidate preference. Accept the edit only when the auxiliary margin exceeds
a development-set threshold and the receiver does not strongly prefer its
own candidate. Commit the entire accepted span and leave every other
scheduled commit intact. Hold deferral and lookahead until proposal
verification shows a benefit.

On development questions, fork keep and replace continuations from the same
canvas and count incorrect-to-correct and correct-to-incorrect flips. Score
the same candidate pair with a receiver-only verifier under a matching
budget to test the value of cross-model evidence. Use successful TIE traces
to compare local replacement with full relay at the same checkpoints. Freeze
the intervention policy for held-out testing against the strongest native
model, compute-matched single-model sampling, and TIE. Evaluate accuracy
and verification cost together. This design is pending study naming.

### Commit-action study implementation

Approved study ID `mdlm-gsm8k-llada-dream-commit-actions`. Primary hypothesis:
cross-model local span verification can recover more answers than it loses
and improve upon exact fusion or an LLaDA-only verifier. Test scheduling
only if the fixed-position span gate passes on validation. The old
32-question TIE attempt 423182 had startup logs and an empty result
directory at implementation time.

The launcher pins the two cached model revisions and records code revision,
worktree status, the original sample IDs, and a copy of the policy for each
job. It uses zero-shot GSM8K, BF16, batch one, and the 256/128/16 TIE
scaffold. The first 32 questions are the smoke split, IDs 32–199 are the
fitting split, IDs 200–299 are the development validation split, and IDs
300–1318 form the held-out split. Held-out runs use four shards to fit the
existing 12-hour allocation. Keep per-job outputs under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions`.

Run the pilot controls `llada`, `dream`, `post_generation`, and `relay`
one at a time with:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh pilot relay
```

Run `pilot-check LLADA_TRACE DREAM_TRACE POST_TRACE RELAY_TRACE` with the
study analysis CLI to check no-relay parity and relay direction coverage.
After the baseline checks, collect the two paired diagnostic traces:

```bash
sbatch --time=02:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit diagnostic
sbatch --time=02:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation diagnostic
```

Run the [study analysis CLI](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_study.py)
with `fit-span FIT_TRACE VALIDATION_TRACE` and save its JSON output as the
span policy. It selects thresholds by fitting net correctness flips, with
ties resolved by fewer interventions. Scheduling runs require
`span_pass=true`. The validation gate requires a strictly positive net gain
over keep and the independently calibrated LLaDA-only verifier. The first
version caps Dream's native gap at four tokens and tries at most one
candidate per question. Record eligibility, boundary failures, matching
texts, and accepted changes separately.

Submit `schedule_diagnostic` on both development splits with the span
policy as argument three. Run `fit-schedule FIT_TRACE VALIDATION_TRACE
SPAN_POLICY` to produce the scheduling policy. The scheduling feature is
Dream's geometric mean probability of LLaDA's proposed gap minus the
highest LLaDA confidence among displaced commits. This measures a
difference in confidence across models and is calibrated against
position-swap outcomes. Use the same scheduling threshold for `positions`
and `joint`. The validation controls are `exact`, `llada_verifier`,
`ctca`, `relay`, `post_generation`, `multi`, and the two native modes.
The exact control uses the top 32 Dream tokens with contextual checks,
preserving unmatched mass. The CTCA control uses 50/50 probability fusion
and confidence ranking on the common scaffold. Native reference runs
omit the scaffold. The multi-sample control includes one greedy sample and
uses token temperature 0.3 for extra samples, with per-question seeds.
Select its model and sample count on development data.

Pass the validation traces to `freeze POLICY ARM=TRACE ...` to select the
held-out candidate by accuracy, compute, and intervention count. Register
the selected held-out arms and shard IDs before submission. Use
`SHARD=0` through `SHARD=3` and pass the frozen policy as argument three.
The `report` command takes an `ARM=TRACE0,TRACE1,TRACE2,TRACE3` argument
per arm. Use each native arm's lm-eval samples JSONL in place of a
trajectory trace. The report joins original document IDs, computes paired
95% bootstrap intervals, and flags whether the single-model multi-sample
control is within 5% of the candidate's model-size-weighted processed
tokens. Report Slurm wall time and GPU-hours separately. Timing and
accuracy expectations remain hypotheses until the GPU jobs finish.

#### Pilot results and decision

| Arm | Slurm job | Flexible exact match | Evaluation time |
|---|---:|---:|---:|
| LLaDA alone | 423396 | 20/32 | 3 min 5 sec |
| Dream alone | 423747 | 22/32 | 2 min 55 sec |
| TIE post-generation selection | 423748 | 20/32 | 5 min 34 sec |
| TIE periodic relay | 423696 | 20/32 | 5 min 35 sec |

Both post-generation branches match their standalone trajectories on all
32 questions. Relay succeeded 103 times from LLaDA and 121 times from Dream,
with zero skips. Compared with final selection alone, relay recovered two
answers and lost two. Compared with Dream alone, relay recovered two and lost
four. At least one standalone model was correct on 25/32 questions. These
results establish implementation parity and bidirectional transfer
feasibility. They show no pilot accuracy gain from final selection or relay.
The study advances to the fixed-position diagnostic. Scheduling remains
gated on the span validation result.

The aggregate lm-eval JSON listed 1,319 effective samples under
`--samples`. Saved per-question traces and the explicit split manifest
verify that each arm evaluated the same 32 IDs. The tracker uses those
artifacts to validate the split. Run arguments now include the scaffold
settings explicitly, which future lm-eval results will save as metadata.

#### Fixed-position span results and scheduling decision

| Split | Job | LLaDA keep correct | Boundary failures | Same proposal | Changed proposal | Correctness gains / losses |
|---|---:|---:|---:|---:|---:|---:|
| Fit, IDs 32–199 | 423785 | 120/168 | 18 | 138 | 12 | 0 / 0 |
| Validation, IDs 200–299 | 423786 | 76/100 | 12 | 80 | 8 | 0 / 0 |

Each question reached one eligible gap before contextual verification.
The 20 distinct accepted-for-replay proposals generated different full
responses from the same pre-commit state, including eight two-slot edits.
None changed final answer correctness. The fitted gate accepted zero edits on
validation, as did the LLaDA-only verifier. The predeclared span gate failed,
so scheduling and held-out runs remain gated. The experiment tested one
greedy auxiliary candidate at one scheduled position group per question.
[Calibration report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/span-policy-423785-423786.json)

#### Proposed follow-up on disagreement from a shared canvas

Measure both models at several predeclared reasoning checkpoints before
either commits. Use the same question and visible response text. Preserve
gap boundaries and record native mask counts after retokenization. A native
prompt template and tokenization can still change the model input. Sample
starting canvases from both standalone decoders so results are not tied to
states reached by LLaDA alone.

Record whether the models favor different text for an aligned gap or
different gaps for the same commitment budget. Report equality of greedy
proposals separately from probability margins and alignment failures.
Replay a bounded set of text-only and position-only disagreements from
identical starting states, using the same continuation model and budget
within each pair. Measure how often the alternative recovers or loses a
correct final answer. Include the continuation model's own alternative
actions at matched compute.

Measure the hindsight improvement available among the tested branches
before fitting a selection rule. Compare any fitted rule with confidence-only
selection and random selection at the same intervention rate. The existing
scheduling and held-out runs remain gated.

The **Shared-canvas disagreement** diagnostic is implemented under the
existing commit-action study. Its revised protocol probes pre-commit states
at denoising steps 12, 44, and 76 of both independent trajectories. At each checkpoint
it can test up to two bounded text gaps and replay at most one disagreement
per action type. For position disagreements, each model ranks aligned
masks using its native probabilities. The replay changes positions while
retaining the owner's token proposals, and the owner-only control
displaces the same number of slots. For text disagreements, a separate
owner-only alternative replaces the first token with a viable candidate
from its top eight while retaining the rest of the proposed gap.

Run the 32-question smoke split first. Check that the trace contains
checkpoints for both draft owners and inspect alignment and replay
coverage before submitting the development splits.

```bash
sbatch --time=02:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh pilot shared_audit_midblock
sbatch --time=03:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit shared_audit_midblock
sbatch --time=02:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation shared_audit_midblock
```

Use [shared_audit_report.py](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/shared_audit_report.py)
with the fit and validation trajectory JSONL paths, in that order. These
runs use the existing development splits. The report compares the
hindsight improvement available at matched checkpoints and tracks
proposal-scoring and replay forwards. It evaluates a selector only if
the fitting data contains a recoverable answer of that type. The offline
selector chooses at most one action per question and draft owner at a
fixed 25% of eligible questions, comparing cross-model ranking against
owner confidence and seeded random selection at the same count. It may
rank options from multiple saved checkpoints, so this is a screening
diagnostic for signals, with online policy validation as a separate step.
The lm-eval headline measures the unmodified LLaDA branch. Consult the
shared-audit report for intervention effects.

#### Block-end pilot and checkpoint correction

Job 423957 completed all 32 questions on one GPU in 9 min 6 sec. All
192 canvases aligned. At steps 16, 48, and 80, every active block had
two remaining masks and a two-slot budget, so all 192 checkpoints
forced the same positions. Position agreement at these checkpoints
provides no evidence about ranking agreement. The revised steps 12,
44, and 76 fall inside the same blocks with 10 active masks and a
two-slot budget. The sampler rejects forced-choice audit checkpoints
and the report records their denominator.

On LLaDA drafts, nine peer text replays and nine owner-only replays
had zero correctness gains or losses. On Dream drafts, 15 peer replays
recovered two answers and lost zero. The 15 Dream-only replays
recovered the same two answers with the exact same token writes and
lost one additional answer. The pilot verifies replay feasibility.
The revised pilot uses separate registry records and output directories.
[Pilot trace](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/pilot/shared_audit/423957/trajectory.jsonl)

#### Mid-block pilot results

Job 423992 completed 32 questions in 19 min 21 sec. All 192 canvases
aligned and each had ten active masks for two commitments. The baseline
branches reproduced the standalone correctness totals, 20/32 for LLaDA
and 22/32 for Dream.

| Draft owner | Checkpoints with position disagreement | Peer position swaps, gains / losses | Owner-only position swaps, gains / losses |
|---|---:|---:|---:|
| LLaDA | 64/96 | 0 / 1 | 0 / 0 |
| Dream | 62/96 | 1 / 2 | 0 / 1 |

The models share 235/384 selected slots across both directions. Peer
position swaps diverged from the original decoding order frequently but
changed correctness only four times. The only recovery was on Dream's
draft for doc 28 at step 12. The matched Dream-only position alternative
did not recover that answer. Both kinds of position replay used 10,424
continuation forwards across the two orientations.

There were 62 identical aligned text proposals, four rejected text
alignments, and one text disagreement. The only distinct peer text
replay was on Dream's draft and left correctness unchanged. Its
Dream-only control performed the same token write. This limited
text coverage applies to these mid-block checkpoints and the strict
gap eligibility rule.

The revised smoke check passes on alignment, scheduling choice, and
paired continuation. The development audit may proceed to measure
recoverable answers and selection quality on more questions. The times
above the submission lines allow headroom over the 19 min 21 sec pilot.
[Audit summary](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/pilot/shared_audit_midblock/423992/pilot-audit.json)

#### Mid-block development results

Fit job 424098 completed docs 32–199. Validation job 424099 completed
docs 200–299. The standalone correctness counts were 120/168 for each
model on fit, and 76/100 for LLaDA and 68/100 for Dream on validation.
All 1,008 fit checkpoints and 598 of 600 validation checkpoints passed
canvas alignment and had a choice of positions. Dream's doc 251 failed
alignment at steps 44 and 76 because a committed fragment decoded to
empty or incomplete Unicode text.

| Split | Draft owner | Matched position replays per method | Peer gains / losses | Owner-only gains / losses |
|---|---|---:|---:|---:|
| Fit | LLaDA | 333 | 2 / 6 | 1 / 6 |
| Fit | Dream | 328 | 3 / 5 | 1 / 4 |
| Validation | LLaDA | 192 | 4 / 1 | 0 / 1 |
| Validation | Dream | 175 | 5 / 4 | 6 / 5 |

Each row counts independent replays at steps 12, 44, and 76. A
question can appear more than once. In validation, Dream's position
choices recovered LLaDA docs 213, 221, 233, and 285. The matched LLaDA-only
swaps recovered zero answers. Of the five Dream answers recovered by
LLaDA's choices, four were also recovered by Dream-only swaps at those
same checkpoints. Validation position replays used 30,444 continuation
forwards per method across both directions.

The fitted cross-model selector and the owner-confidence selector each
changed zero answers when selecting 24 interventions per draft owner
on validation. The 200 equal-rate random selections averaged +0.49 net
answers for LLaDA drafts and +0.22 for Dream drafts, with central 90%
ranges of [−1, +2] and [−2, +2]. The selector ranks saved checkpoints
offline. The results provide no measured accuracy gain for that rule.

Fit produced 16 distinct owner-versus-peer text proposals. One
recovered a LLaDA answer on doc 61, with identical token writes in the
matched owner-only replay. Three harmed Dream answers and the matched
controls also harmed those answers. Validation produced four distinct
proposals and zero correctness changes in either arm.

The position audit aligns unresolved masks by ordinal. Its disagreement
and recovery counts measure the effect of applying the resulting
position swaps to the draft owner's canvas. They cannot establish that
the models ranked the same semantic positions, since split tokens can
shift later words by a slot. The text audit uses shared visible
boundaries and native tokenization, but its four validation alternatives
give little coverage. Keep held-out evaluation gated. The four LLaDA
recoveries provide cases for checking whether the interventions survive
alignment by shared text boundaries.
[Development audit report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/shared-audit-report-424098-424099.json)

#### Text-boundary check of the position recoveries

The post-hoc check used validation job 424099's saved decoding decisions
and the original tokenizers. A promotion qualifies if the peer chose
the entire masked run, the run has visible text on both sides within
the active block, and both models' predicted strings preserve those
boundaries in context. The owner's phrase must have the same native
width in both models for the saved ordinal replay to qualify. Cases
with different native widths require a new forward on a resized gap.
The check applies to the promoted gaps. A displaced baseline slot can
remain unanchored.

| Draft owner | All promoted gaps aligned / peer replays | Peer gains / losses on aligned subset | Same-model gains / losses on those checkpoints |
|---|---:|---:|---:|
| LLaDA | 25/192 | 1 / 0 | 0 / 0 |
| Dream | 10/175 | 0 / 1 | 0 / 1 |

Doc 233 at step 12 was the only LLaDA recovery whose promotion passed.
Both models located the two-token gap between `follow the` and
`in the problem`. LLaDA predicted ` steps described` and Dream
predicted ` steps given`. The peer-guided replay committed LLaDA's
own prediction to slots 2 and 3 and recovered the answer. The original
commitments were slots 2 and 9. The same-model control chose slots 9
and 7 and did not recover the answer. Docs 213, 221, and 285 all
promoted slots in a ten-mask run with no visible right boundary within
the block. Their semantic alignment is unresolved.

This trace analysis gives one example of an aligned scheduling rescue
and measures the low coverage of strict two-sided anchors under the
current block schedule. The audit uses stored predictions and
continuations and does not test a rerendered variable-width policy.
Keep held-out evaluation gated. Any further scheduling experiment
should first demonstrate higher alignment coverage on development
questions and compare a prespecified peer-guided rule against the
matched same-model rule on the same eligible actions.
[Audit script](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/anchored_position_audit.py) · [Audit results](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/anchored-position-audit-424099.json)

#### Trajectory-level review

Pairing the independent outputs on validation gives 56 questions correct
for both models, 20 correct only for LLaDA, 12 correct only for Dream,
and 12 missed by both. A hindsight selector over the two outputs reaches
88/100, compared with LLaDA at 76/100. Fit gives 144/168 for the
corresponding hindsight selector and 120/168 for each single model.

Among the 12 validation questions missed by both native branches on the
matched scaffold, the saved peer-guided position replays recover four
(213, 221, 227, 285). The saved same-model position replays recover three
(227, 259, 290). These are per-question hindsight counts over multiple
checkpoints and both draft owners. None of the four peer recoveries in
this subset meets the two-sided alignment criterion. The aligned LLaDA
recovery on doc 233 was already correct on Dream's standalone branch.
[Validation trace](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/validation/shared_audit_midblock/424099/trajectory.jsonl)

Two visible boundaries identify the same interior unknown region for
local cross-tokenizer scoring and keep the surrounding text fixed.
This requirement belongs to the span diagnostic. It excludes much of
early decoding and many masked suffixes. A suffix with a shared prefix
can be compared by scoring explicit completions with a stopping rule.
Full candidate strings can be compared under native tokenization
without projecting unresolved slots. The current two-sided audit
therefore remains a diagnostic of local alignment.

Proposed next diagnostic under this study is to compare branch generation
and branch selection across early, middle, and late decoding. Start
with ordinary and alternative unmasking schedules on each model's own
state and record their eventual answers. Estimate the hindsight oracle
of each candidate pool before fitting an answer-agnostic selector.
Compare owner-only and cross-model scoring on identical completed
rollouts, using both verifiers on the same candidate text and calibrating
scores within each verifier. If this adds validation value, test repeated
online decisions with the chosen partial branch carried forward, the
native continuation available as a fallback, and scores recomputed on
the resulting state. A matched final-only selector tests whether
intervening during decoding adds value. A matched single-model search
tests whether a second model adds value over extra computation. Include
TIE relay as a full-state sharing baseline and count all verifier and
rollout compute. These are proposed comparisons on development data.
[TIE](https://arxiv.org/html/2606.16281v1) · [UnMaskFork](https://arxiv.org/html/2602.04344v2) · [Lookahead Unmasking](https://arxiv.org/abs/2511.05563)

#### Proposed action-level probability fusion

Autoregressive generation provides a shared output frontier. With a
common emitted prefix, both models can score the next output symbol in
an aligned vocabulary or byte space. In the current diffusion ensemble,
ordinal mask matching does not establish that the projected distributions
describe the same missing text. The scheduler also turns a change in
marginal confidence into a possible change in which positions become
visible next. A shared-tokenizer pair with fixed scheduling would help
measure token fusion separately from these effects. These controls remain
proposed. [Exact Byte-Level Probabilities](https://arxiv.org/abs/2410.09303)

For a first action-fusion diagnostic, use the usual LLaDA commitment as
action A and replace its last-ranked selected slot with its next-ranked
unselected slot as action B. Keep the number of commitments and LLaDA's
token proposal at each slot fixed across rules. Generate a reference
continuation to completion for each action from the same checkpoint.
Both verifiers read both response texts in their own tokenizer. A first
score is the mean log probability of reconstructing the candidate answer
from the question and its visible reasoning, with answer tokens masked.
This score measures reconstructability and can favor a coherent wrong
answer. Record scoring coverage by checkpoint and use action A when the
paired score is unavailable.

On fit, calibrate each verifier's B-minus-A score margin to three
outcomes determined by reference continuation correctness. Action B
helps if only B is correct, harms if only A is correct, and ties if both
have the same correctness. Average the two predicted outcome
distributions with equal weights in the first comparison. The net gain
score is the fused help probability minus the fused harm probability.
Choose B only above a threshold set on fit and otherwise keep A. Report
paired recovery, harm, and net counts on validation. Compare against
LLaDA-only scoring on the same candidates at the same intervention rate,
and match total inference compute in a separate single-model control.
The initial diagnostic uses full reference continuations. A later online
test must recompute the action scores on the path chosen at each
checkpoint and account for all lookahead and verification compute.

#### Training-free action-fusion pilot

Start with unanimous preference on the 32-question pilot. Both models
score the same two completed responses by masking the answer and
measuring mean native answer-token log probability. Reuse scores for
identical text. At steps 12, 60, and 108, action B replaces the last
slot LLaDA would commit with its next-ranked slot. Both actions use
LLaDA's token proposals. Dream plays no role in generating either
candidate or setting the number of commitments.

The report counts recoverable questions, coverage, and gains and harms.
The unanimous rule accepts positive margins in both models. The matched
LLaDA rule ranks by its own positive margins on the same scorable
candidate pool and selects the same number of questions. Both rules
select at most one checkpoint per question, using the largest LLaDA
margin among eligible swaps and the earliest checkpoint on a tie.
This comparison uses full rollouts and saved checkpoints. The lm-eval
headline measures the unmodified LLaDA response. Rollout and verifier
work are counted separately. The pilot checks feasibility and action
coverage before any calibration or repeated online intervention.

```bash
sbatch --time=01:30:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh pilot action_fusion
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/action_fusion_report.py)
with `pilot` and the resulting absolute trajectory JSONL path.

Job 424449 completed all 32 questions in 7 min 12 sec of lm-eval
evaluation time. Its LLaDA baseline reproduced all 32 response texts
from job 423396 and scored 20/32. Both verifiers scored all 96 action
pairs. Ten alternative responses differed from baseline, with one
recovery and one harm. The best tested branch in hindsight reached
21/32. The replays used 6,528 additional LLaDA forwards, and each
verifier used 42 scoring forwards after reusing identical texts.

| Checkpoint | Pairs scored | Recoveries | Harms |
|---|---:|---:|---:|
| 12 | 32 | 0 | 1 |
| 60 | 32 | 0 | 0 |
| 108 | 32 | 1 | 0 |

Unanimous fusion and LLaDA-only selection at the same rate both chose
doc 30 at step 108, with no correctness change. On doc 19 at step 108,
the alternative changed the answer from 0 to 6 and recovered correctness.
Dream preferred it by +1.112 mean answer-token log probability. LLaDA
preferred the baseline by 0.420. The unanimity rule excluded the
only recovery. Both verifiers preferred the baseline on the harmful
swap, doc 5 at step 12.

This pilot verifies score coverage and establishes that the tested
one-slot candidate pool has limited recoverable headroom. It does
not establish an accuracy gain from unanimity or from Dream alone.
A development follow-up should measure recovery coverage for a more
diverse fixed action pool before fitting a selector. Keep separate
LLaDA-only, Dream-only, and fused rules on identical candidate pairs at
matched rates. This would test whether the disagreement on doc 19 is
useful beyond one question and whether a fusion rule can exploit it.
[Pilot report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/pilot/action_fusion/424449/action-fusion-report.json)

#### Wider action-fusion development runs

The `action_fusion_wide` arm compares four actions at each of steps
12, 60, and 108. All writes come from LLaDA and use the two-slot
budget. Write `r1` through `r4` for the four highest-ranked available
slots at the current checkpoint.

| Action | Slots | Purpose |
|---|---|---|
| Baseline | r1, r2 | Usual LLaDA action |
| Defer last | r1, r3 | Reproduce the original one-slot alternative |
| Defer first | r2, r3 | Test deferring the top-ranked slot |
| Defer two | r3, r4 | Test changing both commitments |

Each alternative runs to completion from the same LLaDA checkpoint,
followed by the existing answer-reconstruction scores from both
models. The analysis counts recoveries per action and questions
recovered only by the wider pool. It also compares LLaDA-only,
Dream-only, and equal-weight fusion on pairs scored by both models.
Each model uses its B-minus-A score. Fusion averages their signed
percentile ranks in the distribution of absolute nonzero margins on
unique changed responses from fit, with a midrank for ties. The sign
lets strong evidence from one model outweigh weaker disagreement from
the other. No correctness labels enter these score scales. The
validation report reuses the fit scales.

Each rule picks at most one positive-scoring action per question.
The common number of swaps is the smallest count available to all three
rules, capped at 25% of questions with a scorable changed response,
rounded down with a minimum cap of one if such questions exist. The
matched comparisons screen for differences in selector quality using
saved checkpoint rollouts. The lm-eval headline is the unchanged
LLaDA branch. The increased pool adds 612 continuation forwards per
question, before verifier calls. Assess how much unique recovery
headroom the wider actions add before using the results to plan an
online test.

```bash
sbatch --time=03:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit action_fusion_wide
sbatch --time=02:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation action_fusion_wide
```

Use [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/action_fusion_report.py)
with `fit FIT_TRACE --wide` or
`validation VALIDATION_TRACE --wide --fit-trace FIT_TRACE`.

##### Wider action-fusion results

Fit job 424501 completed all 168 questions in 80 min 37 sec.
Validation job 424502 completed all 100 questions in 47 min 26 sec.
The LLaDA baseline responses matched the corresponding shared-audit
responses on all 268 questions. Both verifiers scored every action
pair.

| Split | Paired branches | Changed responses | Questions recoverable with original swap | Questions recoverable with all swaps | Hindsight best with baseline kept available |
|---|---:|---:|---:|---:|---:|
| Fit | 1,512 | 174 | 3 | 5 | 125/168 |
| Validation | 900 | 102 | 2 | 4 | 80/100 |

Validation uses the score scales from fit. At the common swap budget,
the results were as follows.

| Split | Swaps per rule | LLaDA-only gains / harms | Dream-only gains / harms | Fused gains / harms |
|---|---:|---:|---:|---:|
| Fit | 17 | 0 / 2 | 1 / 3 | 0 / 2 |
| Validation | 10 | 1 / 3 | 1 / 1 | 2 / 1 |

Validation scores were 74/100, 76/100, and 77/100 for the respective
rules, with unmodified LLaDA at 76/100. The paired bootstrap
intervals for fusion were [0, +7] percentage points versus LLaDA-only
scoring and [0, +3] versus Dream-only scoring. The margins are small
and compatible with no gain. In fit, all three rules scored 118/168
against the unmodified baseline of 120/168. Unanimity on validation
recovered two and harmed two at its separate 15-swap rate.

The wider pool added validation recoveries on docs 246 and 285.
Fusion selected both and harmed doc 225. LLaDA-only selected the
recovery on doc 285 and Dream-only selected the recovery on doc 246.
Both verifiers favored all three of these changes. Dream's standalone
branch already solved doc 246. The wide pool found correct branches on
three of the 12 questions missed by both independent models (221, 285,
and 290). The fused rule selected one of those three (285).

Validation used 61,200 continuation forwards and 174 scoring forwards
per verifier. The results concern offline selection using complete
rollouts at baseline checkpoints. They do not establish gains for
repeated online decisions or at matched inference compute. The next
comparison should use a back-to-back final-only and online policy at
the same compute if further development yields a larger or repeatable
selection gain. Keep the held-out set unused for this diagnostic.
[Fit report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/fit/action_fusion_wide/424501/action-fusion-report.json) · [Validation report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/validation/action_fusion_wide/424502/action-fusion-report.json)

##### Assessment and proposed follow-up

The `defer_two` action produced a correct branch on all five
recoverable fit questions and all four recoverable validation
questions. The simpler pool would therefore have the same
hindsight headroom on these traces. This is a post-hoc result.
The validation gain from the fused rule is two recoveries and one
harm over LLaDA, with a two-sided exact paired test of p = 1.00.
The corresponding comparisons with LLaDA-only and Dream-only
scoring have p = 0.25 and p = 1.00. Treat these as descriptive
diagnostics. They use the question as the unit, with all questions
retained. The apparent gain does not repeat on fit, and the
headline validation contrast against LLaDA is based on three
changed outcomes. The 11 statistical fallacy checks were reviewed.
The material limitations are the selected action pool, low rate
of correctness changes, exploratory comparisons, and use of
completed baseline rollouts to evaluate a proposed online rule.

In a descriptive random check with the same number of swaps,
sample questions uniformly among those with a scorable changed
response, then sample a unique altered response for each question.
The exact expected net change was −0.75 answers on fit and −0.19
on validation. In 20,000 seeded validation draws, 25.8% achieved
at least the fused rule's +1. This is a reference distribution for
that random rule. Each of the wider jobs used 5.78 times the
standalone LLaDA generation forwards before verifier calls.

1. Score the two independently generated LLaDA and Dream responses
   already stored for each development question. Use the same
   native answer-reconstruction scoring and fit-frozen fusion to
   choose between the completed responses. Record both standalone
   scores, this final-only selector, and hindsight union accuracy.
   This establishes how much complementary information is accessible
   through final-answer selection with little additional compute.
2. Compare the usual LLaDA action with `defer_two` at the same
   three checkpoints. On temporary branches, finish the active
   reasoning block and generate a tentative answer in the reserved
   answer block while later reasoning remains masked. Present
   the visible prefix and tentative answer as ordinary response
   text to both verifiers. Freeze score scales using fit margins
   from this shorter scoring path. Use complete reference rollouts
   to label recoveries and harms. Compare short-lookahead selection
   with LLaDA-only, Dream-only, and random selection at the same
   swap count. Measure scorable coverage, the share of recoveries
   retained relative to full-rollout fusion, harms, and compute.
   Treat the current validation split as exploratory because it
   informed the choice of action. If promising, repeat the frozen
   comparison on a separately fixed development sample before
   using the reserved test IDs 300–1318.
3. Advance to an online comparison if short-lookahead fusion
   yields repeatable positive net gain over unmodified LLaDA and
   both single-verifier rules at lower cost. Apply the chosen
   action to the live state and recompute the next decision. Use
   an inference-time threshold fixed on development in place of
   ranking all questions together. Compare with compute-matched
   LLaDA-only search, final-only selection with the same budget,
   and TIE relay. Report paired accuracy differences, processed
   tokens by model, forward counts, and GPU time. Freeze the policy
   and a primary paired comparison before held-out evaluation.

##### Final-answer fusion baseline

The `final_answer_fusion` arm reuses the independently decoded LLaDA
and Dream responses from shared-audit jobs 424098 (fit) and 424099
(validation). Check the exact question and stopping criteria against
the saved lm-eval samples. Both models score both saved responses
with the existing answer-reconstruction verifier. Identical texts
share a score within a verifier. This arm runs zero generation
forwards and at most four verifier forwards per question. The
lm-eval headline is the saved LLaDA baseline.

For each verifier, form the Dream-response score minus the
LLaDA-response score. The fused rule averages the two signed
percentile ranks, using unlabeled final-answer margins on fit as
the reference. Pick the saved Dream response if this fused margin
is positive. The report also evaluates each verifier's own choice
on the same scorable pairs. All three rules fall back to LLaDA on
a tie or when any paired score is missing. Report scoring coverage,
both standalone accuracies, the hindsight union, and paired
recoveries and harms relative to LLaDA. Inference can run
concurrently across splits. The report needs both traces.

```bash
sbatch --time=01:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit final_answer_fusion
sbatch --time=01:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation final_answer_fusion
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/final_answer_fusion_report.py)
with the absolute fit and validation trajectory JSONL paths, in
that order.

Job 426502 completed fit in 171 seconds and job 426503 completed
validation in 73 seconds of lm-eval evaluation time. They scored
saved responses with zero new generation forwards. The replay cost
was 614 verifier forwards on fit and 364 on validation.

| Rule | Fit correct / 168 | Validation correct / 100 | Validation gains / harms vs LLaDA |
|---|---:|---:|---:|
| Standalone LLaDA | 120 | 76 | 0 / 0 |
| Standalone Dream | 120 | 68 | 12 / 20 |
| LLaDA verifier | 120 | 76 | 5 / 5 |
| Dream verifier | 122 | 76 | 6 / 6 |
| Fused final-answer selector | 122 | 77 | 6 / 5 |
| Hindsight union | 144 | 88 | 12 / 0 |

Fusion chose the Dream response on 46 validation questions. Its
paired 95% bootstrap interval against standalone LLaDA was
[−6, +8] percentage points. Final-answer fusion and wider
action fusion both scored 77/100 on validation, with six questions
correct only under each policy. The two rules used different
candidate pools and intervention rates. Their equal totals give no
observed accuracy advantage for action fusion in this comparison.
The wide action diagnostic used 61,200 extra continuation forwards
in addition to scoring. The final-answer diagnostic cost excludes
generating the two independent responses.

Both verifiers could score both responses on 139/168 fit and 82/100
validation questions. All missing scores came from Dream responses
with empty reserved `Answer:` fields. The metric could still extract
correct answers from the body of 25 of these fit responses and 17
of these validation responses. LLaDA's fallback missed four and two
Dream-only correct answers on fit and validation respectively. The
unscorable Dream-only correct validation docs were 250 and 258.
A follow-up should evaluate a shared answer-extraction rule for
both candidate sources before using this verifier to judge shorter
lookahead, and measure coverage and accuracy separately.
[Final-answer fusion report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/final-answer-fusion-report-426502-426503.json)

##### Final-answer extraction rerun

The `final_answer_fusion_extract` arm reuses the same saved
responses and labels. For both candidate sources, take the last
numeric match using the GSM8K `flexible-extract` pattern on the
response after applying the original stops. Mask that span in its
original location. The verifier keeps the surrounding reasoning
visible. Each tokenizer may include adjacent whitespace or
punctuation in an overlapping token. Skip spans that would cover
additional reasoning or that fail an exact text round trip.
Earlier mentions of the same value remain visible, so reconstruction
measures the recoverability of the number from context. It can
assign high scores to a wrong but internally consistent answer.

The CPU tokenizer check matched the saved benchmark extraction for
all 268 LLaDA responses and verified paired tokenization on 168/168
fit and 99/100 validation questions, including docs 250 and 258.
Dream's saved response for doc 251 contains an incomplete Unicode
fragment, so the verifier still falls back to LLaDA for that question.
This checks representation and coverage. The GPU run measures
scoring and selection. Both splits continue to use the same
signed-rank fusion and LLaDA fallback, with validation scales frozen
from the extraction-specific fit run. Compare this rerun with the
original final-answer selector on all questions, and separately
inspect questions that were previously unscorable.

```bash
sbatch --time=01:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit final_answer_fusion_extract
sbatch --time=01:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation final_answer_fusion_extract
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/final_answer_fusion_report.py)
with the absolute fit and validation trajectory paths and
`--extract`.

Job 426545 completed all 168 fit questions and job 426546 completed
all 100 validation questions. The saved candidate texts and
correctness labels match the original final-answer runs. Paired
coverage increased from 139/168 to 168/168 on fit and from 82/100
to 99/100 on validation. Job 426546 used 398 verifier forwards
and zero generation forwards. Both verifiers still abstained on
Dream's malformed Unicode response for doc 251.

| Rule using extracted numeric spans | Fit correct / 168 | Validation correct / 100 | Validation gains / harms vs LLaDA |
|---|---:|---:|---:|
| Standalone LLaDA | 120 | 76 | 0 / 0 |
| LLaDA verifier | 121 | 76 | 3 / 3 |
| Dream verifier | 123 | 78 | 4 / 2 |
| Fused selector | 120 | 77 | 3 / 2 |

The fused selector stayed at 77/100, with six questions correct only
under each extraction rule. The paired 95% bootstrap interval for the
change relative to the original fusion was [−7, +7] percentage
points. In the 17 newly covered validation pairs, all three
selectors recovered doc 258 without a new harm. Both verifiers
favored the wrong LLaDA response on the other Dream-only recovery
opportunity, doc 250. The broader scoring rule also changed the
preferences on previously scorable questions.

Use the extracted-answer verifier to avoid the reserved-field
coverage problem. Dream-only verification is the strongest
observed final-answer control on both development splits. Its
validation gain over standalone LLaDA has a paired 95% bootstrap
interval of [−3, +7] points. The next short-lookahead diagnostic
should retain Dream-only, LLaDA-only, and fusion on identical
actions. Compare any online candidate with the final-answer
selector and account for the cost of generating both independent
candidates. These results support checking verifier quality after
coverage is fixed, with no observed advantage for equal-weight
fusion over Dream-only scoring under extracted-answer scoring.
[Extraction rerun report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/final-answer-fusion-extract-report-426545-426546.json)

##### Short-lookahead action diagnostic

The `action_fusion_short` arm tests LLaDA's usual top two slots
against filling its third and fourth choices at steps 12, 60, and
108. Each temporary branch finishes the four remaining steps in
its active reasoning block and then fills the reserved answer
block in eight steps. Later reasoning stays masked. The verifier
sees the visible prefix and tentative answer as response text.
A short branch without a numeric span in its reserved answer field
is unscorable. Both verifiers apply the same extracted-answer
reconstruction used in the final-answer baseline.

Use archived full-rollout jobs 424501 and 424502 to obtain terminal
correctness. Check baseline text, final token IDs, checkpoint block,
and both actions' slots and token writes against the saved reference.
Rescore the relevant full responses with the extracted-answer
verifier so the full and short comparison reflects the available
context. No full alternative is regenerated. The lm-eval headline
is the original LLaDA path. The action comparison is offline and
uses only saved terminal outcomes as labels.

The report compares LLaDA-only, Dream-only, and fused short scoring
with fused full scoring on the pairs both horizons can score. Fit
separate unlabeled score scales by horizon and freeze them on
validation. Every selector gets the same swap budget, capped at
25% of informative questions, with at most one swap per question.
Also report score coverage, recoveries, harms, correct actions
retained from full scoring, and an equal-rate random comparison.
The short diagnostic adds 72 LLaDA lookahead forwards per question
before verification. Count reference verification separately. The
held-out split stays reserved for a frozen online comparison.

```bash
sbatch --time=01:30:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit action_fusion_short
sbatch --time=01:00:00 --exclude=gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh validation action_fusion_short
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/short_action_fusion_report.py)
with the absolute fit and validation trajectory paths, in that order.

Validation job 426575 finished all 100 questions. Its original
LLaDA path scored 76/100. It saved all 300 action pairs, and 299
could be scored at both horizons. The short diagnostic used 7,200
lookahead generation forwards and 682 verification forwards.

Fit job 426574 saved 129/168 complete question records for docs
32–160. The latest log shows a CUDA out-of-memory error while
loading LLaDA on gpu-37, where another process held 26.50 GiB of
GPU memory. There is no complete fit result. The registered report
requires all fit docs.

Calibrating on the partial fit trace gives the following provisional
validation comparison at nine swaps per rule.

| Rule | Gains / harms | Provisional correct / 100 |
|---|---:|---:|
| LLaDA-only short scoring | 2 / 1 | 77 |
| Dream-only short scoring | 3 / 0 | 79 |
| Fused short scoring | 3 / 1 | 78 |
| Fused full scoring | 2 / 0 | 78 |

Fused short scoring retained both full-scoring recoveries (docs 221
and 290), added doc 285, and introduced one harm. The short fusion
gain versus baseline has a provisional paired 95% interval of
[−2, +6] points. Dream-only short scoring led the provisional
comparison. The partial fit itself shows a net loss of two from
short fusion and a net gain of three from full fusion at 12 swaps.
Repeat fit with the fixed protocol before drawing conclusions.

```bash
sbatch --time=01:30:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fit action_fusion_short
```

[Provisional report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/short-action-fusion-provisional-426574-426575.json)

###### Complete fit rerun and validation

Fit job 428334 finished all 168 questions on gpu-23. It exactly
reproduced all 129 saved trace rows from job 426574. The final report
uses job 428334 to set the fit scales and the original validation
trace from job 426575. The provisional validation outcomes were
unchanged. Both horizons scored 501/504 common fit pairs and
299/300 common validation pairs.

| Selector | Fit gains / harms at 14 swaps | Fit correct / 168 | Validation gains / harms at 9 swaps | Validation correct / 100 |
|---|---:|---:|---:|---:|
| Unmodified LLaDA | 0 / 0 | 120 | 0 / 0 | 76 |
| LLaDA-only short scoring | 2 / 3 | 119 | 2 / 1 | 77 |
| Dream-only short scoring | 3 / 4 | 119 | 3 / 0 | 79 |
| Fused short scoring | 2 / 3 | 119 | 3 / 1 | 78 |
| Fused full scoring | 3 / 0 | 123 | 2 / 0 | 78 |

On validation, Dream-only short scoring and short fusion recovered
the same three questions (221, 285, and 290). Fusion also chose a
harmful swap on doc 201. Short fusion matched full fusion in total,
with one additional recovery and one additional harm. Short fusion's
paired 95% interval against unmodified LLaDA was [−2, +6] points.
On fit, short fusion and LLaDA-only short scoring had identical
correctness outcomes. All short rules scored one below LLaDA.

The short rollout retained both full-fusion validation recoveries,
but equal-weight fusion showed no accuracy advantage over
Dream-only verification on either split. The matched swap counts
come from ranking saved baseline checkpoints and do not estimate
repeated online decisions or compute-matched end-to-end accuracy.
Keep the holdout reserved. A fresh development comparison should
retain Dream-only as a control and require a reproducible gain from
combining verifiers before advancing the fusion rule.
[Complete short-lookahead report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/short-action-fusion-report-428334-426575.json)

##### Fresh short-lookahead comparison

The `action_fusion_short_fresh` arm uses 192 questions sampled
without replacement from the 7,473 GSM8K training-split questions
with seed 20261122, sorted by source index. This is a fresh
development sample for this study. The lm-eval task alias uses the
same zero-shot GSM8K prompt, stopping rules, and numeric metric on
those rows. The reserved GSM8K test IDs remain outside this run.

Use the same checkpoints (12, 60, 108), usual LLaDA top-two
commitment, and defer-two alternative as the preceding diagnostic.
LLaDA follows its usual path for lm-eval and generates a complete
reference continuation for the alternative independently at each
checkpoint. Each short candidate uses only the completed active
block, earlier reasoning, and the tentative answer. Both models
score the same tentative responses. The full branches supply
correctness labels and a full-scoring reference, with their cost
recorded separately from short lookahead.

Freeze both score scales from complete fit job 428334. Keep the
one-action-per-question rule, shared scorable pool, and matched
swap budget capped at 25% of informative questions. Report paired
wins and losses, confidence intervals, scorable coverage, and
compute. The primary comparison is short fusion against Dream-only
short scoring. Advance to a live development test if short fusion
has positive net gain over the baseline and both single-verifier
rules. The offline report uses saved checkpoints and does not
establish repeated online effects.

```bash
sbatch --time=03:00:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh fresh action_fusion_short_fresh
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/short_action_fusion_report.py)
with the [frozen fit trace](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/fit/action_fusion_short/428334/trajectory.jsonl),
the fresh trajectory path, and `--fresh`.

Job 428416 completed the 192-question sample in 52 minutes of
lm-eval evaluation time. Both horizons could score 572 of 576
pairs. LLaDA scored 144/192. The complete alternatives produced
six recovery branches across five distinct questions and eleven
harmful branches. A hindsight selector that keeps LLaDA's usual
answer available reaches 149/192 on this action pool.

| Rule | Gains / harms at 19 swaps | Correct / 192 |
|---|---:|---:|
| Unmodified LLaDA | 0 / 0 | 144 |
| LLaDA-only short scoring | 0 / 3 | 141 |
| Dream-only short scoring | 1 / 2 | 143 |
| Short fusion | 1 / 2 | 143 |
| Full fusion | 1 / 2 | 143 |

Short fusion and Dream-only scoring chose 13 identical actions and
had identical correctness outcomes on every question. Both recovered
doc 2639 and harmed docs 1777 and 3513. Full fusion recovered the
same question and caused two harms, one on a different question.
The paired 95% interval for short fusion versus unmodified LLaDA
was [−2.08, +1.04] percentage points. Equal-rate random swaps averaged
−1.01 net answers. Short lookahead used 13,824 continuation forwards
and 1,324 verifier forwards. Creating the full references required
39,168 additional continuation forwards.

The preset criterion for advancing to live decoding was not met.
The wider sequence of fit, validation, and fresh comparisons gives
no repeatable gain from equal-weight fusion on these commitment
actions. The fresh action pool also has limited hindsight headroom.
Keep the held-out set reserved and require new evidence of useful
complementary actions or verifier signals before another online test.
[Fresh comparison report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/short-action-fusion-fresh-report-428334-428416.json)

##### Peer-proposed reasoning block pilot

The `peer_block_proposals` arm tests whether heterogeneous
proposals add correct continuations beyond single-model resampling.
Use the existing 32 pilot questions and branch at steps 17 and 65,
with a completed reasoning prefix and an empty next block. The
unmodified LLaDA path supplies the lm-eval baseline.

Dream receives the exact visible LLaDA prefix using its native
tokenizer, then generates 24 native tokens in the next block over
eight forwards. Use the earliest complete Dream token prefix that
encodes to exactly 16 LLaDA tokens and round-trips to the same
surface text in context. Apply those tokens to LLaDA's next block,
keeping the earlier state and remaining masks. This uses a known
prefix and a fixed owner budget. It requires no matching of
unresolved token indices across models.

For each eligible Dream proposal, compare a block generated by
LLaDA over eight forwards using the existing temperature-0.3
resampling primitive. Both branches then use the same greedy LLaDA
continuation schedule. Reject proposals after a terminal prefix or
when exact retokenization fails. Record coverage, differing native
widths, answer losses, unique correct recoveries for each proposal
source, and proposal and continuation cost. The primary outcome is
the number of questions recovered only by a Dream proposal. This is
an exploratory candidate-availability test. It uses gold outcomes
for analysis and does not fit or evaluate a live selector.

```bash
sbatch --time=01:30:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh pilot peer_block_proposals
```

Run [the report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/peer_block_proposals_report.py)
with the pilot trajectory path.

The tokenizer preflight on the archived standalone LLaDA path
verified that Dream can represent all 64 candidate prefixes exactly.
This establishes prefix coverage before Dream generates any new text.
Submitted the single pilot as Slurm job 428581. Its initial queue
state was pending on priority.

Job 428581 completed all 32 questions in 50.0 minutes of reported
lm-eval time. Of 64 attempted proposals, 59 paired with a LLaDA
resample. Dream used a different native width on 37 of the 59.
LLaDA resampling changed no correctness labels. Dream proposals
recovered docs 4, 19, and 31, and caused three regressions across
the paired checkpoint continuations. All three recoveries came
from step 17. Standalone Dream already solved docs 19 and 31.
Doc 4 was the only recovery missed by both native trajectories.
The oracle reached 23/32 against LLaDA's 20/32. That oracle
uses gold answers to keep LLaDA successes and select recoveries.
[Peer-block report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/pilot/peer_block_proposals/428581/peer-block-proposals-report.json)

##### Online exact-text token and position fusion pilot

The `exact_text_fusion` arm revisits Q6 and Q8 after the actions
in the fresh short-lookahead study failed to improve accuracy.
It uses the existing 32 GSM8K pilot questions. Each denoising
step drafts every active LLaDA mask with its native top prediction.
Retokenize each ordinary masked run's draft in Dream and align
positions whose decoded prefix boundaries agree. Show Dream only
the already committed text and the resulting mask layout. This
assumes the draft is a useful guide to relative token widths.
Mask counts reveal information about draft tokenization and can
change as the trajectory evolves. Both-bounded gaps are unnecessary.

For one-slot Dream spans, project the top 32 native probabilities
to exact single LLaDA tokens with contextual round-trip checks.
For two-slot Dream spans, use left-to-right path probabilities
on the same masked canvas, with each second prediction conditioned
on the first. Search up to two first-token prefixes per denoising
step, prioritizing prefixes compatible with LLaDA's top eight
and the best Dream prefix. Accumulate probability only for
paths that decode to the same text. Width-changing alternatives
and spans spanning multiple LLaDA slots are outside the pilot's
projection. With retained Dream mass `m` and projected
subdistribution `r`, use `q = (1 − 0.5m)p_LLaDA + 0.5r`.
The estimate covers scored native paths. Unscored mass goes back
to LLaDA rather than being redistributed over the surviving tokens.

Use the same LLaDA canvas, scaffold and commit counts in four
online arms. Compare native, fused tokens with native ordering,
native tokens with max-fused-probability ordering, and fused
tokens with fused ordering. Rebuild the map after every reasoning
step. Keep final answer-block generation native. The primary
comparison is joint fusion versus both standalone trajectories.
Observe paired flexible-extract gains and losses, projection
coverage, retained mass, changed commits, and model-wise forward
and processed-token counts. Treat a positive pilot as a reason
to replicate against both native baselines and a compute-matched
LLaDA control on fresh examples.

The pinned tokenizers verified ` centimeters every` as two LLaDA
tokens and three Dream tokens. The shared boundaries map LLaDA's
` centimeters` to Dream's first two tokens and ` every` to its
third. A CPU check using completed reasoning from the 32 archived
LLaDA trajectories as reference drafts mapped 10,523 of 11,520
synthetic masked positions at width one and 123 at width two.
These counts measure alignment with known drafts. Actual
online coverage and end-to-end accuracy require model inference.
See the [algorithm](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tie/exact_text_fusion.py)
and [paired report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/exact_text_fusion_report.py).

```bash
sbatch --time=02:00:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh pilot exact_text_fusion
```

Submitted the four-arm pilot in one Slurm job, 428609.

Job 428609 completed all 32 questions in 67.4 minutes of
reported lm-eval time across the four arms. The two pilot jobs
and both archived standalones matched document, target, and
prompt hashes for every question. Both jobs' LLaDA control
responses matched the archived LLaDA responses exactly.

| Arm | Flexible exact match | Paired gains / losses vs LLaDA | Paired gains / losses vs Dream |
|---|---:|---:|---:|
| LLaDA | 20/32 | — | 3 / 5 |
| Dream | 22/32 | 5 / 3 | — |
| Token fusion | 18/32 | 2 / 4 | 3 / 7 |
| Position fusion | 21/32 | 4 / 3 | 5 / 6 |
| Joint fusion | 20/32 | 3 / 3 | 3 / 5 |

Joint fusion projected 30,818 of 34,560 position observations
(89.2%), including 188 two-Dream-token observations. Mean retained
Dream mass was 93.1% over projected observations and 83.0% over all
observations. Joint fusion changed 283 written token IDs from the
LLaDA proposal on the current canvas and displaced 1,569 positions.
It used 4,096 LLaDA and 4,082 Dream forwards. The independent
two-model oracle is 25/32. Joint fusion solved one question both
standalones missed (doc 4), and the position-only arm solved two
(docs 4 and 12). These are descriptive results from reused questions.
[Exact-text fusion report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/pilot/exact_text_fusion/428609/exact-text-fusion-report.json)

##### Fused support for the committed token

The position-only arm ranked slots by `max_v q_i(v)` while
writing the owner's native top proposal `x_i`. A different
token can generate the high score and promote the slot even
as the peer assigns little support to `x_i`. The
`commit_support` arm tests the direct score `q_i(x_i)`.
Use Dream as the owner because it had the higher standalone
pilot accuracy. Retokenize into LLaDA, rebuild the alignment
each step, and use the same mass-preserving 50/50 mixture and
fixed per-step commit quota. Always write Dream's native top
token from the current canvas. Keep the reserved answer block
native. There is no fitted threshold or learned scorer.

In one job, evaluate both native scaffolded trajectories,
Dream with the original max-fused ranking, Dream with ranking
by support for the written token, and a two-sample Dream
control. That control reuses the greedy Dream trajectory and
adds a temperature-0.3 draw, selecting by answer-token
stability using the existing sampler rank. Both live position
rules use the same projection and forward-budget limits.
Report weighted processed tokens and per-model forwards. The
two-sample control has a similar nominal budget. Verify the
actual cost ratio before treating it as compute matched.

Freeze 32 GSM8K training questions with seed 20261123 after
excluding the earlier 192-question training-development set.
The reserved GSM8K test IDs remain unused here. The primary
outcome is paired flexible exact match of commit support
against Dream. Report paired gain and loss counts against
the other three controls, and log position changes and how
often the fused top token disagrees with the one written.
Treat the 32-question estimate as exploratory. An accuracy
claim requires a fresh larger comparison against both native
models and the single-model control.
[Commit-support report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_support_report.py)

```bash
sbatch --time=02:00:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh support commit_support
```

Submitted all five arms in one Slurm job, 430959.

Job 430959 completed all 32 questions in 22.3 minutes of reported
lm-eval time across the five arms. The saved responses and flexible
correctness flags match lm-eval sample records for all 32 frozen IDs.

| Arm | Correct | Gains / losses vs Dream | Gains / losses vs LLaDA |
|---|---:|---:|---:|
| Dream | 24/32 | — | 6 / 8 |
| LLaDA | 26/32 | 8 / 6 | — |
| Two-sample Dream | 26/32 | 3 / 1 | 6 / 6 |
| Fused max-probability rank | 28/32 | 6 / 2 | 6 / 4 |
| Fused support for the written token | 26/32 | 6 / 4 | 5 / 5 |

Fused max ranking gained five answers and lost three versus
two-sample Dream, a net gain of 6.25 percentage points (paired 95%
bootstrap interval [−9.38, +25.00]). Its gain versus LLaDA was
6.25 points [−12.50, +25.00]. The two standalone trajectories
collectively solved every question in hindsight.

Commit support cut writes that disagreed with the fused top token
from 332 to 217. On these diverging trajectories it recovered three
answers max ranking missed and lost five answers max ranking solved.
It ties both LLaDA and two-sample Dream in accuracy. It used 98.5%
of two-sample Dream's parameter-weighted processed tokens. Max ranking
used 93.9% of that proxy budget. Actual per-arm latency was not measured.

The outcome motivates replication of the existing Dream-owned max
rank at its fixed 50/50 mixture. Use fresh questions and compare
against both standalone models and two-sample Dream. The pilot
selected the candidate to replicate and is excluded from the
replication estimate.
[Paired report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/support/commit_support/430959/commit-support-report.json)

##### Fused position ranking replication

The `fused_position_replication` run freezes the 50/50,
Dream-owned max-fused ranking from job 430959. It reuses the
existing projection, commit rule, forward limits and answer
scaffold. Score the same four arms on 768 GSM8K training
questions sampled with seed 20261124, excluding the earlier 192
and 32 training-development IDs. Keep the reserved GSM8K test
split for a separate benchmark evaluation. Exclude job 430959
from all replication estimates.

With a two-sided exact McNemar test at 0.05, the 768-question
sample has about 93% power against two-sample Dream and 86%
against LLaDA under the pilot's 6.25-point gain and respective
25% and 31.25% discordance. For a five-point gain at the same
discordance rates, these estimates fall to 78% and 68%.
These are per-comparison calculations under assumed effect sizes.

The primary comparison is flexible exact match versus two-sample
Dream at a similar nominal inference budget. Report paired gains,
losses, 95% bootstrap intervals and two-sided exact McNemar
p-values also versus both standalone models. A claim of beating
all controls requires a positive effect and p < 0.05 in each
comparison. Report the ratio of parameter-weighted processed
tokens against two-sample Dream and require it to be no more
than 1.05 for the inference-budget claim. Analyze the entire fixed
set once complete. Queue one GPU job and use the existing progress log.

```bash
sbatch --time=10:00:00 --exclude=gpu-37,gpu-62 /home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_actions_slurm.sh replication fused_position_replication
```

Use [the paired report](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tie/commit_support_report.py)
with the new trajectory and `--replication`.

Submitted the four-arm replication as Slurm job 431372.

Job 431372 completed all 768 frozen development questions in
5.68 hours of reported lm-eval time. The split is disjoint
from the two earlier training-development sets. The stored
responses and flexible correctness flags match all 768 lm-eval
samples and the main result. The pilot gain did not replicate.

| Comparator | Comparator correct | Fused gains / losses | Fused Δ (percentage points) | Paired 95% interval | Exact p |
|---|---:|---:|---:|---:|---:|
| Two-sample Dream (primary) | 556/768 | 110 / 115 | −0.65 | [−4.43, +3.13] | 0.790 |
| LLaDA | 575/768 | 103 / 127 | −3.13 | [−6.90, +0.78] | 0.129 |
| Dream | 560/768 | 104 / 113 | −1.17 | [−4.95, +2.60] | 0.587 |

Fused ranking scored 551/768 (71.74%). It passed the cost
criterion at 94.6% of two-sample Dream's parameter-weighted
processed tokens and used 1.83× LLaDA's. It failed the accuracy
criterion. No comparison had a positive point estimate.
All three intervals exclude a five-point gain on this split.
This is an independent decoding comparison on GSM8K training
questions. The reserved test split remains unused by this method.

Projection covered 68.0% of active position observations and
displaced 12.7% of reasoning commits. Pilot rates were 67.2%
and 12.3%. Where both standalone models were correct, fusion
lost 85 of 454 answers. Where both standalones failed, it
recovered 25 of 87 answers. Twenty-one of the latter were
also missed by two-sample Dream. The native hindsight oracle
reaches 681/768. These counts show both useful changes and loss
of already-correct answers. Treat fixed 50/50 max-fused
position ranking as a negative replication and retain the
reserved test for a method with stronger development evidence.
[Replication report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/mdlm-gsm8k-llada-dream-commit-actions/replication/fused_position_replication/431372/fused-position-replication-report.json)

### Synthetic position correspondence: `ctca-toy-position-correspondence-20261123`

This CPU diagnostic tests one denoising step with eight master slots and known
auxiliary correspondences. The two synthetic models have different token IDs
and embedding spaces. A probability-weighted sum of token profiles gives each
position a distribution over shared anchors. Score every master–auxiliary pair
by cosine similarity. Hold vocabulary transfer exact via the synthetic token
ID permutation so the experiment isolates position correspondence.

Compare same-index fusion, per-row argmax, and strictly increasing one-to-one
matching by dynamic programming. Try both similarity-based matchers over the
full score matrix and within `|i − j| ≤ 4`, using generation-relative positions.
The matchers discard scores at or below 0.5. The dynamic program assigns each
match a reward of `cosine − 0.5` and can skip positions on either side at zero cost.
Mix matched distributions 50/50 and use the master at unmatched slots. Include
master-only, oracle-aligned fusion, and oracle-aligned peer predictions with
master fallback.

The frozen cases cover identical layouts, a local crossing decoy with one
master slot missing its counterpart, a five-token offset, and an overconfident
peer on an identical layout. Record gold-pair precision and recall, crossings
and collisions, top-1 token accuracy, paired gains and losses against the master,
and mean gold-token negative log likelihood. The proposed mechanism passes the
local case if monotonic matching recovers more gold pairs and correct tokens
than same-index fusion and independent argmax. The offset and overconfidence
cases test failure modes of the window and of fixed-weight fusion, respectively.

[Toy script](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_position_correspondence.py)

```bash
source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
conda activate dllm
python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_position_correspondence.py
```

Completed on CPU. Each table cell gives gold pairs recovered and then correct
master tokens. Master-only gave 5/8 correct tokens in all four cases.

| Fixture | Same index | Row argmax ±4 | Monotonic ±4 | Monotonic full | Oracle fusion |
|---|---:|---:|---:|---:|---:|
| Identical layouts | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 |
| Local crossing | 1/7, 2/8 | 6/7, 7/8 | 7/7, 8/8 | 7/7, 8/8 | 7/7, 8/8 |
| Five-token offset | 0/8, 0/8 | 0/8, 5/8 | 0/8, 5/8 | 8/8, 8/8 | 8/8, 8/8 |
| Overconfident peer | 8/8, 4/8 | 8/8, 4/8 | 8/8, 4/8 | 8/8, 4/8 | 8/8, 4/8 |

In the local case, independent matching preferred auxiliary slot 4 for
master slot 1, crossing the match from master slot 2 to auxiliary slot 3
(zero-based). Monotonic matching picked auxiliary slot 2 for master slot 1,
skipped the unmatched master slot, and corrected three master predictions.
Oracle-aligned peer predictions with master fallback got 7/8 in this case.
The five-token offset excluded all gold pairs from the ±4 window. Oracle
fusion with the overconfident peer corrected three tokens and spoiled four.
The fixtures use engineered anchor profiles and exact vocabulary transfer, so
the outcomes establish behavior on one synthetic step. The script constructs
the full score matrix for all variants and measures candidate eligibility for
the window experiment.

An exact rerun reproduced the report. Exhaustive enumeration of eligible
monotonic matchings confirmed the dynamic-programming optimum in each case.

[CPU report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-toy-position-correspondence-20261123/report.json)

### LLaDA–Dream position matching with top-k: `ctca-gsm8k6-llada-dream-position-topk-20261123`

Replay saved native distributions from the six exploratory CTCA traces
(docs 3, 6, 9, 11, 18, and 23). Use refinement steps 0 and 4 of blocks 0, 1,
4, 8, and 15. Compare each active master mask against the saved native
auxiliary masks. The reference pair is the exact overlap from CTCA's canvas
rendering, which copies each master mask to one auxiliary mask and retokenizes
surrounding visible text.

Project each model's native top 8, 16, 32, or 64 probabilities through the
same cached 2,048 shared-anchor profiles at temperature 0.01. Verify cache
vocabulary signatures against the tokenizer snapshots. Use original native
probabilities and omit mass from decoded strings with ambiguous token IDs.
Record native and usable retained mass at each cutoff. Compare same-index
matching, independent argmax within ±4, and the toy's monotonic matcher with
and without the ±4 constraint. Count scaffold agreement, matches proposed,
and slot decisions unchanged relative to the saved top-64 reference.

The audit measures how well distribution scores recover a constructed
scaffold on real logits. The six traces were previously inspected and their
tails beyond top 64 were unavailable. Testing independent model trajectories,
token fusion quality, and end-to-end decoding time requires live forwards.
[CPU driver](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/offline_position_topk.py)

Completed on CPU with 1,440 reference masks across 60 checkpoints. The ±4
window contains 1,296 true pairs. The other 144 have auxiliary index five
slots to the right. Same-index matching recovers 1,089 reference pairs.

| Native top-k | Mean retained mass, master / auxiliary | 10th-percentile mass, master / auxiliary | Correct monotonic ±4 pairs / proposed | Correct unbanded monotonic pairs / proposed |
|---:|---:|---:|---:|---:|
| 8 | 88.2% / 88.9% | 56.2% / 53.2% | 43 / 104 | 36 / 114 |
| 16 | 92.6% / 93.2% | 72.1% / 71.1% | 41 / 105 | 35 / 114 |
| 32 | 95.7% / 96.2% | 84.0% / 84.1% | 41 / 110 | 36 / 122 |
| 64 | 97.4% / 97.9% | 90.5% / 91.8% | 44 / 115 | 36 / 127 |

At top-64, only 81 of 1,440 true pairs exceeded cosine 0.5. Even before
applying that threshold, the true auxiliary mask had the highest cosine
within the band for only 187 of the 1,296 eligible reference slots. This
points to weak discrimination under the tested anchor profiles and cosine
score. Among the 351 shifted reference slots, banded monotonic top-64
correctly recovered 10 and unbanded top-64 recovered seven. Larger k within
the saved range changes little in correct matches.
The band admits 11,831 of 471,552 scored position pairs. The CPU driver
materializes all of them for the controlled comparison.

A separate warm-cache, single-thread CPU check measured a gather of the
cached profiles, conversion to float32, and weighted batched multiplication
for 32 native rows from doc 23 at block 0, step 0. Median time for that
operation was about 0.8 ms per model at top-16, 1.8 ms at top-32, and 4.4 ms
at top-64. This timing covers position-profile construction from already
available probabilities. A direct weighted sum also verified the sparse
mixtures and retained masses at all four k values.
[CPU report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k6-llada-dream-position-topk-20261123/report.json)

#### ±5 window check

Ran the same 60 checkpoints and cosine threshold with `--radius 5`. The
larger window admitted all 1,440 reference pairs and increased eligible
position comparisons from 11,831 to 14,285 (+20.7%). No monotonic variant
recovered any of the 144 newly eligible +5 reference pairs.

| Native top-k | Correct / proposed at ±4 | Correct / proposed at ±5 |
|---:|---:|---:|
| 8 | 43 / 104 | 42 / 106 |
| 16 | 41 / 105 | 41 / 107 |
| 32 | 41 / 110 | 41 / 113 |
| 64 | 44 / 115 | 44 / 116 |

The same-index and unbanded results reproduced exactly. A controlled +5
pair with identical profiles verified that the radius admits the boundary
match when the score supports it. [±5 report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k6-llada-dream-position-topk-20261123/report-window5.json)

### Independent reconciliation: `ctca-llada-dream-independent-reconciliation-pilot-20261123`

One-GPU pilot on three hand-written masked reasoning passages.
Use pinned LLaDA‑8B‑Instruct and Dream‑7B‑Instruct, BF16, greedy confidence,
and Dream's native sampling-logit shift. The fixture's hidden reference span
determines each tokenizer's initial mask width and provides the evaluation
label. It is hidden from the model except for gaps excluded at preflight,
which become visible and are excluded from results. Preflight requires at
least one eligible gap with unequal native widths. This setup measures
reconciliation with known gap boundaries and widths.

1. At each synchronization, render the same visible text in both tokenizers.
   Record the native indices of each persistent gap ID. Allow two native
   denoising steps with one greedy write per model per step and keep the
   original gap IDs through those in-place writes.
2. Read complete proposed spans by gap ID. Compare distinct texts on the
   same pre-round context and require exact contextual tokenization in both
   models. Allow native paths of up to three tokens, resizing the tested gap
   for a candidate when necessary. Score each path left-to-right, average
   over its native tokens, and weight both models equally. Leave a partially
   filled gap masked unless a complete proposal for it is accepted.
3. The joint arm rerenders the reconciled text and immediately fills one
   remaining common gap. For a one-token gap in both views, compare exact
   strings appearing in both native top-32 lists. For a wider gap, cross-score
   each native greedy span as text. Rerender the accepted text in both views
   and repeat for up to eight rounds, stopping when no gaps or no newly
   accepted choices remain. Gap identity defines the candidates, with no
   fixed native-index window.

Compare native independent completion, periodic committed-span reconciliation,
and reconciliation plus immediate unmasking. Report exact hidden-span matches,
coverage, conflict decisions, partial drafts, native paths and widths, plus
forward and processed-token counts by model and phase. Score split-width gaps
separately. Use these three fixtures to validate mechanics before evaluating
alignment from unknown boundaries, estimated widths, or open-ended generation.

Submit the live pilot with Slurm.

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_independent_reconciliation_slurm.sh
```

Tokenizer-only checks can use the CPU partition.

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/independent_reconciliation.py --preflight
'
```

[Pilot driver](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/independent_reconciliation.py)

#### First live run (job 431131)

The program ran all three passages on `gpu-59` and wrote the final report.
The stderr log contains checkpoint-loading progress and a Transformers
generation-config warning about `temperature`. This pilot calls the models
directly for logits and uses greedy writes. Slurm accounting was unavailable
from the current session.

| Arm | Exact hidden spans | Gaps filled | Model forwards |
|---|---:|---:|---:|
| LLaDA alone | 17/19 | 19/19 | 27 |
| Dream alone | 16/19 | 19/19 | 28 |
| Reconcile commits | 13/19 | 14/19 | 89 |
| Reconcile and immediately unmask | 18/19 | 19/19 | 152 |

In the boxes passage, commit-only stopped after round two with five gaps
open. Both models spent their two writes on partial three-token numbers.
The arm only accepts complete proposals and stops when a round adds no
shared text. The joint arm filled `12`, `36`, and `31` by immediately
unmasking complete spans, and finished all six gaps. All eight immediate
unmasks across the three passages matched their reference spans. The joint
arm's one exact-span error was a reconciled commit in the distance passage.

The joint arm also immediately unmasked ` centimeters every` in two
LLaDA tokens and three Dream tokens. Exponentiating the recorded mean
log scores times each native path length gives 0.486 and 0.258. A gate
requiring a complete-path score above 0.8 in both models would have
deferred this correct split-width fill at that checkpoint. This was one
split-width fixture and an ungated run.
[Final report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-independent-reconciliation-pilot-20261123/431131/report.json)

### Reconciliation confidence: `ctca-llada-dream-reconciliation-confidence-pilot-20261123`

Follow-up on the same three passages, checkpoints, and two native
writes per round as job 431131. Freeze the original native width of each gap
in both tokenizers for every arm. This gives the gates the same proposal
and scoring conditions as the new ungated controls. The original pilot
allowed candidate-specific resizing, so its results serve as context.

| Comparison | Arms |
|---|---|
| Confidence gating | Joint reconciliation and immediate unmasking with no gate, a complete-path gate, or a per-token geometric-mean gate |
| Partial writes | Ungated commit-only with a fresh native draft each round, or with each model's incomplete writes carried forward by gap ID and native offset |
| Native controls | Independent LLaDA and Dream completion |

The gate applies to both reconciled commits and immediate unmasking. Both
models must assign the exact text a score strictly above 0.8. The path arm
uses the product of the left-to-right native conditional probabilities.
The geometric-mean arm applies the same cutoff on a per-token scale. Both
keep the original proposal ranking among eligible candidates. Gates use
common-context rescoring. Every independent write also records its original
chosen ID, probability, native top-32, gap ID, offset, and write order, with
the round's shared text and starting draft available for context replay.
Carried tokens stay private to their owning model until a full candidate is
verified and accepted as shared text.

Report exact fills, final coverage, stop reasons, and model forwards per
arm. Separately apply both gates to the same decisions made on the ungated
joint trajectory. Count wrong ungated proposals blocked, correct proposals
deferred, and correct or wrong replacements by phase, scenario, and native
width pair. These matched-context counts isolate the immediate effect of the
gate. The live arms also include changes to subsequent decoding contexts.
The original fixtures contain one known unequal-width gap and serve as a
mechanics and failure-mode check.

Submit the GPU comparison with the existing one-GPU wrapper.

```bash
sbatch --job-name=ctca-reconcile-confidence /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_independent_reconciliation_slurm.sh --confidence-study --confidence-threshold 0.8
```

Reports will be written under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-reconciliation-confidence-pilot-20261123/<job-id>/report.json`.

Run the focused tests on the CPU partition.

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  cd /home/juan.guevara/naruto/dllm-epiplexity
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_reconciliation_confidence.py
'
```

#### Live results (job 431454)

The run on `gpu-23` wrote the complete report for all three passages and
all seven arms. The known native widths were fixed across arms.

| Arm | Exact hidden spans | Gaps filled | Passages stalled | Model forwards |
|---|---:|---:|---:|---:|
| LLaDA alone | 17/19 | 19/19 | 0 | 27 |
| Dream alone | 16/19 | 19/19 | 0 | 28 |
| Commit-only, reset drafts | 13/19 | 14/19 | 1 | 89 |
| Ungated reconciliation and immediate unmasking | 18/19 | 19/19 | 0 | 152 |
| Complete-path gate above 0.8 | 14/19 | 14/19 | 3 | 174 |
| Per-token geometric-mean gate above 0.8 | 14/19 | 14/19 | 3 | 174 |
| Commit-only, carry partial drafts | 18/19 | 19/19 | 0 | 110 |

The two gate arms produced identical traces. On the 19 matched ungated
action contexts, each gate kept 14 actions, deferred four exact-reference
fills, and blocked the one exact-match miss. There were no replacement
actions. The blocked text was ` long` in `That is six thousand meters long.`,
where the fixture expected ` altogether`. The run scores exact text. Both
gates deferred the correct two-to-three-token ` centimeters every` fill
and three correct one-token fills. For example, both models had independently
written ` remain`, but the shared-context scores were 0.645 and 0.917.

Carryover recovered all six boxes gaps through commit reconciliation, up
from one with reset drafts. After a round that added no shared text, it
retained the partial native writes for `31` and completed that gap in the
next round. Across the three passages it produced exactly the same text as
ungated immediate unmasking with 42 fewer forwards (27.6%). These known-gap
fixtures favor carryover as the next comparison baseline. They provide one
unequal-width example and one exact-match miss, so gate calibration needs
more labeled conflicts and width pairs.
[Final report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-reconciliation-confidence-pilot-20261123/431454/report.json)

### Expanded disagreement pilot: `ctca-llada-dream-reconciliation-disagreements-pilot-20261124`

Diagnostic using the same pinned LLaDA and Dream checkpoints as job
431454. The primary question is whether deferring an actual disagreement
avoids wrong shared commits without blocking too many correct ones. The
comparison holds partial-draft carryover, known gaps, and native gap widths
fixed. There is no fitted probability cutoff.

The frozen bank has 18 passages and 44 requested gaps before preflight.
Eight inventory passages have arithmetic labels and an incorrect ledger
claim. Ten transcription passages have an official source and a competing
draft. The bank includes shared-width gaps and unequal-width cases such as
` Toul`, ` centimeters every`, and a punctuation mark followed by a newline.
Preflight requires exact contextual tokenization of each answer and checks
that each controlled foil fits the same native width in both models. It
records every excluded gap and each rejected foil it tries. Require at least
two unequal-width groups with valid controlled pairs.

Run two live carryover arms. Each model makes up to two private greedy
writes and logs its original top-32. Reconcile full proposals on the same
pre-round context. Score a candidate text in each tokenizer by filling its
native path from left to right and averaging its conditional log
probabilities per token. The joint arm chooses the largest equal-weight
average across the two models. The unanimity arm accepts that winner on a
two-candidate disagreement only if both models strictly prefer it to the
other valid proposal. Both arms keep partial writes private for the next
round. Single eligible proposals and exact agreements follow the ordinary
commit rule. No immediate-unmask step runs in either arm.

Record natural conflicts and which ones survive the fixed-width verification.
On the joint arm's original contexts, count which otherwise accepted
candidates unanimity would veto, grouped by whether they were correct,
wrong, or unlabeled. Compare the full live trajectories separately, including
coverage, stalls, forwards, and processed tokens. Native LLaDA and Dream
completion are additional controls.

For the controlled comparison, supply one predeclared valid foil and the
reference as contenders for the same fully masked starting context. Score
both with the same native paths as the live arms. Report each model's choice,
the joint choice, unanimity's choices and deferrals, and how often the two
models prefer different candidates. These labeled comparisons test the rule
when both contender texts are available. They contribute no proposals to the
live trajectories and are reported by native width pair.

Arithmetic output is labeled by parsed integer value. Other formats remain
unlabeled. The transcription instructions require literal copying. Native
mask widths are obtained from the hidden references, so this design tests
reconciliation with known correspondence and width. It does not estimate
alignment accuracy on free generation.

Submit the one-GPU study:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_reconciliation_disagreements_slurm.sh
```

The report is written to
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-reconciliation-disagreements-pilot-20261124/<job-id>/report.json`.
The GPU driver runs tokenizer preflight before it loads model weights. To
validate the fixtures and unit tests separately, request the CPU partition:

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  cd /home/juan.guevara/naruto/dllm-epiplexity
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_reconciliation_confidence.py &&
  python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/reconciliation_disagreements.py --preflight
'
```

[Pilot driver](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/reconciliation_disagreements.py)

#### Live results (job 432583)

The one-GPU run on `gpu-47` completed all 18 passages. All 44 requested
gaps passed tokenizer preflight. The split widths covered `(1, 2)` for four
gaps, `(2, 1)` for two, and `(2, 3)` for four, in LLaDA/Dream order.

| Live arm | Correct gaps | Gaps filled | Model forwards | Processed tokens |
|---|---:|---:|---:|---:|
| LLaDA alone | 44/44 | 44/44 | 90 | 6,948 |
| Dream alone | 44/44 | 44/44 | 96 | 7,940 |
| Carryover with joint ranking | 44/44 | 44/44 | 359 | 28,745 |
| Carryover with unanimity veto | 44/44 | 44/44 | 359 | 28,745 |

The two live carryover traces were identical. Of the 44 accepted fills,
31 came from matching complete drafts, six had only one complete draft
with the peer fully masked, and seven had only one complete draft with
the peer partly filled. No round had competing complete native proposals.
Each carryover arm used 173 forwards for native writes and 186 for commit
rescoring. All rescoring occurred without a conflict eligible for the veto.

Both models preferred the reference on every one of the 42 controlled
pairs, including all ten unequal-width pairs. Two numeric foils changed
the native width and were excluded from the controlled comparison. The
closest pair was `37` versus `38` in `stock_8`, with a positive reference
margin of 0.083 mean log-probability units in Dream and 1.271 in LLaDA.

These results establish correct transfer and continued decoding on
known-width gaps. They leave the accuracy and coverage effects of the
disagreement veto unmeasured. A decision audit can use saved checkpoints
with observed, labeled, comparable conflicts. Estimate how often those
conflicts occur separately on an unselected sample.
[Final report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-reconciliation-disagreements-pilot-20261124/432583/report.json)

### Archived real conflicts: `ctca-llada-dream-real-conflict-audit-20261124`

CPU audit of the saved shared-canvas text replays from block-end
pilot 423957, mid-block pilot 423992, mid-block fit 424098, and mid-block
validation 424099. Report each run separately because checkpoint budgets
differ and the two pilots revisit the same questions. The main development
comparison uses the two mid-block splits. Earlier coverage reports counted
16 text disagreements on fit and four on validation. Include every saved
distinct text proposal with a paired replay that passes verification, without
filtering on continuation correctness or score margins. Report coverage
over the archived checkpoint population as well as the paired subset.

The checkpoints came from both native owner trajectories. Each peer
proposed text on a retokenized view of its counterpart's checkpoint. Recheck
that both candidates preserve the gap's two visible boundaries and fit the
same native width within each tokenizer. Verify that the saved replay keeps
all scheduled positions and other token writes, changes only the gap text,
and continues with the same owner model and budget. Log any excluded replay.
Use the archived flexible GSM8K final-answer correctness of the two
continuations as outcome labels.

Normalize each saved peer-minus-owner conditional log-score margin by its
native path width. Compare keeping the owner, always taking the peer on an
eligible pair, and joint ranking by the mean of both margins. Ties in joint
ranking keep the owner. Unanimity commits a choice only when the margins
have the same strict sign. Report correct and incorrect joint choices that
unanimity would defer, and whether each deferred incorrect choice had a
correct saved alternative. The archive supplies no continuation for leaving
the gap open. Report keeping the owner after a veto as an explicit fallback
policy with its own paired outcome. Slice by draft owner and native width
pair. With different greedy one-token proposals, both models ordinarily
favor their own choice on the respective scoring canvases, so unanimity can
defer by construction in that stratum.

The audit loads tokenizers and saved artifacts with zero new model
forwards. Record the original scoring work implied by the saved algorithm,
two conditional paths per model for a verified pair, and the verification
work after the peer has already proposed its text. The latter is two owner
paths plus one peer path. Record the original archive's aggregate proposal
scoring cost separately because it includes other actions and checkpoints.

Submit to the CPU partition:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_archived_reconciliation_conflicts_slurm.sh
```

The report is written to
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-real-conflict-audit-20261124/<job-id>/report.json`.
The focused tests can run on the same CPU partition:

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  cd /home/juan.guevara/naruto/dllm-epiplexity
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_archived_reconciliation_conflicts.py
'
```

[Audit driver](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/archived_reconciliation_conflicts.py)

#### Archived audit results (job 432656)

The CPU job on `cn-02` wrote a complete report for all four archives. All
45 saved disagreements passed the pinned-tokenizer boundary checks and
paired replay verification. The main mid-block development splits contained
20 verified pairs across 1,608 owner checkpoints under the archived gap
eligibility rule. Each table outcome counts a checkpoint continuation.

| Archive | Verified pairs | Keep owner correct | Take peer correct | Joint correct | Unanimity defers | Unanimity with owner fallback correct |
|---|---:|---:|---:|---:|---:|---:|
| Block-end pilot | 24 | 15 | 17 | 17 | 23 | 15 |
| Mid-block pilot | 1 | 1 | 1 | 1 | 1 | 1 |
| Mid-block fit | 16 | 12 | 10 | 11 | 16 | 12 |
| Mid-block validation | 4 | 3 | 3 | 3 | 4 | 3 |

On mid-block fit, joint ranking switched to the peer seven times and
incurred one answer loss with zero gains relative to keeping the owner.
It missed the one available peer recovery. Validation had four pairs with
identical answer-correctness labels across branches, so it supplies no
outcome-changing test of the ranking rule. Across both development splits,
only four pairs changed answer correctness. All 20 were deferred by
unanimity, including 14 correct and six incorrect joint choices. Four of
those six incorrect choices had two incorrect continuations. Keeping the
owner after every veto reproduced the owner baseline.

The older block-end pilot's joint rank recovered both available peer
gains, whereas unanimity deferred both. Across all four archives,
unanimity deferred all 35 single-token pairs, eight of nine two-token
pairs, and the sole unequal-width pair. The latter had two correct
continuations. This gives narrow coverage for decisions on unequal widths.
The audit contains no rollout for holding a gap open.

The original algorithm used 92 conditional scoring forwards for the
20 development pairs, including 69 verification forwards after the peer
proposals were available. The completed audit itself ran zero model
forwards. Its stderr contained a CUDA/NVML availability warning from the
PyTorch import on the CPU node.
[Final report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-llada-dream-real-conflict-audit-20261124/432656/report.json)

### Periodic JS ensemble: `ctca-gsm8k-llada-dream-periodic-js-ensemble-pilot-20261124`

End-to-end comparison on 32 GSM8K training questions drawn with seed
202611241 outside the existing `fresh`, `support`, and `replication` splits.
Keep the GSM8K test set unused. Use the pinned BF16 LLaDA and Dream weights,
native prompt templates, greedy scoring, and the existing TIE scaffold with
256 generation slots, 128 steps, 16-slot blocks, and a 16-slot answer block.

Compare native LLaDA, native Dream, periodic exact-text probability fusion,
and the same fusion with a JS gate. All branches use the same nominal decode
schedule. The periodic arms preserve LLaDA's selected positions and alter
only token choices. The LLaDA answer block completes natively in both arms.

1. Fork two private reasoning steps from the shared LLaDA round start.
   LLaDA writes its usual scheduled tokens, saving their original rows. Use
   LLaDA's initial masked-gap draft to infer Dream's native span widths.
   Render committed text and masks in Dream. The proposed text used to infer
   widths stays hidden from Dream. Dream makes its own greedy private writes
   over the mapped masks with the same per-step write budget when enough
   mapped slots are available.
2. At synchronization, look up verified Dream spans for the LLaDA writes.
   Mask all compared Dream spans again while retaining the other private
   writes as context. Score candidates in each model's native context.
   For single-token Dream spans, use a union of native top-32 candidates
   and include Dream's private write if complete. For two-token spans,
   score conditional paths with at most two prefix forwards per span and
   four per round. Prefer the private Dream path and the LLaDA incumbent
   when compatible. Keep only paths that transfer as exact single LLaDA
   tokens in context. Wider or unverified spans keep LLaDA's write.
3. Put the certified common events and all remaining mass into categorical
   distributions and compute equal-weight JS divided by `log(2)`. Both
   fusion arms require at least 0.5 common mass from each model. The JS
   arm also requires divergence at most 0.20. Keep the omitted mass in an
   explicit bin for this coarse divergence estimate. For accepted rows,
   choose the argmax of `(1 - 0.5m)p_LLaDA + 0.5r_Dream`, where the exact
   projected Dream subdistribution has mass `m`. Keep LLaDA on low coverage,
   excessive divergence in the gated arm, or an invalid synchronized canvas.
   Start another private round from the accepted shared state.

Report paired flexible GSM8K final-answer accuracy and gains/losses against
both native baselines. Compare the gate directly with the ungated fusion
arm on the same questions. Record model forwards and processed tokens,
coverage, accepted token changes, and support and divergence by Dream span
width and projected top-1 agreement. Both policies make the same type of
scoring calls. Their actual contexts and costs can diverge after different
writes. The pilot measures the cost and quality of a real decoding policy on
a development sample. Shape limits and a fixed schedule keep the experiment
focused on cross-tokenizer token choice before scheduling changes.

Submit the one-GPU pilot:

```bash
sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh
```

Each run writes `report.json`, the frozen split, and the raw trajectory under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k-llada-dream-periodic-js-ensemble-pilot-20261124/<job-id>/`.
Unit tests can run on the CPU partition:

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  cd /home/juan.guevara/naruto/dllm-epiplexity
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_tie.py -k periodic_js
'
```

[Ensemble sampler](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tie/periodic_js_ensemble.py)

#### Live results for job 432695

The trajectory and both lm-eval scoring filters contain the same 32 document
IDs as the frozen split. The lm-eval summary reports the full training set size
of 7,473 in `n-samples`. The ledger validates the effective count against the
frozen split and trajectory. The run artifacts record the checkout revision
and working-tree status.

| Arm | Correct | Flexible exact match | Model forwards | Processed tokens |
|---|---:|---:|---:|---:|
| Native LLaDA | 24/32 | 75.00% | 4,096 | 1,372,416 |
| Native Dream | 25/32 | 78.13% | 4,096 | 1,394,816 |
| Periodic fusion | 21/32 | 65.63% | 9,838 | 3,296,229 |
| Periodic fusion with JS | 25/32 | 78.13% | 9,903 | 3,318,173 |

The JS arm gained seven correct answers and lost three compared with ungated
fusion, a net gain of 12.5 percentage points (paired 95% bootstrap interval
−6.25 to +31.25). It gained one and lost zero against LLaDA. Against Dream it
gained four and lost four. Both comparisons with native decoding leave equal
accuracy plausible on this sample.

Each fusion arm scheduled 7,680 reasoning writes. The JS arm rejected 1,534
scored positions for high divergence and changed 24 LLaDA tokens. Ungated
fusion changed 630 tokens along its own trajectories. Both arms only changed
tokens at one-token Dream spans. The JS arm scored 75 two-token Dream paths and
passed 48 through the fusion gates. None changed the LLaDA choice.

The JS arm used 2.42 times as many forwards as either native arm. Its 5,807
Dream forwards break down into 3,774 for private drafts, 1,917 for verification,
and 116 for conditional paths. The JS check happens after those calls, so it
can protect writes while the draft and verification costs still accrue. The
pilot supports retaining JS as a write filter for future tests. A routing test
would need to make its decision before the Dream calls and compare accuracy
and cost against both native arms.

[Paired report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k-llada-dream-periodic-js-ensemble-pilot-20261124/432695/report.json)
[Run log](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-periodic-js_432695.out)

### Selective JS ensemble pilot

`ctca-gsm8k-llada-dream-selective-js-ensemble-pilot-20261124` compares
standalone LLaDA, standalone Dream, always-call JS fusion, and selective JS
fusion on the same 64 zero-shot GSM8K training questions. The frozen split
uses seed `202611242` and excludes the `fresh`, `support`, `replication`, and
`periodic_js` IDs. The GSM8K test set stays reserved. The pinned BF16 models,
greedy decoding, 256 slots, 128 steps, 16-slot blocks, and native 16-slot
answer block follow the periodic pilot.

After the two normal LLaDA reasoning steps, route the whole round to Dream if
the minimum top-token probability among LLaDA's scheduled writes is below
0.80. Use the probabilities from the existing LLaDA calls. Skipped rounds keep
all LLaDA writes and perform no Dream forwards. Routed rounds apply the same
private drafting, exact-text projection, 0.50 common-mass floor, and 0.20
normalized JS ceiling as the always-call arm. Routing controls access to the
whole round. The fusion rules continue to evaluate each write at its original
LLaDA position.

In job 432695, at least 625 of 1,920 rounds had a recorded write below 0.80.
All four scheduled writes were recorded and at or above 0.80 in 1,001 rounds.
The remaining 294 rounds leave the estimated call fraction between 32.6% and
47.9% on the archived trajectory. All 24 accepted edits fell in the 625 known
call rounds. These counts describe the periodic trajectory. Selective decoding
can visit different states after its own writes.

Compare paired flexible exact match against both standalone arms and the
always-call arm, with gains, losses, and 95% paired bootstrap intervals. Report
model forwards, Dream forward savings, processed tokens, generation time per
arm, and routed, skipped, and actually contacted rounds. The Slurm log records
GPU type and peak PyTorch allocated memory for the job. The pilot target is
more correct answers than both standalone arms with at least 50%
fewer Dream forwards than always-call fusion. Report accuracy against the
always-call arm even if the compute target is met. The development pilot tests
one prespecified routing rule and leaves a larger held-out evaluation for
estimating a reliable gain.

```bash
sbatch --job-name=ctca-selective-js /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh --selective
```

The Slurm stdout and stderr are
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-selective-js_<job-id>.out`
and `/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-selective-js_<job-id>.err`.
The four-arm report, split, raw trajectory, and lm-eval artifacts are under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k-llada-dream-selective-js-ensemble-pilot-20261124/<job-id>/`.

```bash
srun -p cscc-cpu-p --cpus-per-task=4 --time=00:10:00 bash -c '
  cd /home/juan.guevara/naruto/dllm-epiplexity
  source ~/.zshrc 2>/dev/null || source /apps/local/conda_init.sh
  conda activate dllm
  python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_tie.py -k "periodic_js or selective_js"
'
```

#### LLaDA-main results for job 432899

All 64 trajectory IDs and both scoring filters match the frozen split. The
lm-eval summary lists 7,473 in `n-samples`, the full training-set size. The
selective arm's flexible exact match was 81.25% ± 4.92 percentage points
(bootstrap standard error). Strict exact match was 0.00% ± 0.00 under the
filter that expects `The answer is <number>.` The four-arm comparison uses
flexible exact match.

| Arm | Correct | Flexible exact match | Total forwards | Dream peer forwards | Generation time |
|---|---:|---:|---:|---:|---:|
| Standalone LLaDA | 54/64 | 84.38% | 8,192 | 0 | 313.7 s |
| Standalone Dream | 48/64 | 75.00% | 8,192 | 0 | 307.5 s |
| Always-call JS | 52/64 | 81.25% | 19,774 | 11,582 | 863.4 s |
| Selective JS | 52/64 | 81.25% | 12,155 | 3,963 | 501.0 s |

Selective routing reproduced all 64 always-call token sequences and all 59
edits. It routed 1,321 of 3,840 rounds, made Dream forwards in 1,314 rounds,
and skipped 2,519 rounds. It saved 65.8% of Dream forwards, 38.5% of total
forwards, and 42.0% of measured generation time against always-call fusion.
Generation time above is per arm. The full four-arm lm-eval ran for 2,015.6 s
on one NVIDIA A100-SXM4-40GB with peak PyTorch allocation of 29.27 GiB.

Each fusion arm gained zero answers and lost two against standalone LLaDA
(−3.125 percentage points, paired 95% bootstrap interval −7.81 to 0.00).
Against Dream, each gained 11 and lost seven (+6.25 points, interval −6.25
to +18.75). The peer-forward reduction met the pilot target. The improvement
over both native arms remains unobserved.

Both regressions against LLaDA contain two accepted edits. On document 3787
the answer moved from 46 to 66. On 5310 it moved from 6 to 60. All four
replacement tokens had LLaDA probability below 0.50 and passed the JS gate.
This identifies cases to use when reviewing candidate-level confidence.
All edits across the run used one-token Dream spans. Always-call and
selective fusion scored 140 and 47 two-token Dream spans respectively.

[Paired report](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k-llada-dream-selective-js-ensemble-pilot-20261124/432899/report.json)
[Slurm stdout](/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-selective-js_432899.out)

### Dream-main selective JS pilot

`ctca-gsm8k-dream-llada-selective-js-ensemble-pilot-20261124` uses the same
frozen 64 questions and protocol as the LLaDA-main pilot (job 432899). The
Dream-main run was submitted as job 432963. Dream
chooses the commit positions, fixes the reasoning canvas in its own tokenizer,
and generates the final answer block. LLaDA supplies a private draft and
probabilities for matching one- and two-token spans. The peer projection and
JS gate apply in Dream's vocabulary. The always-call arm consults LLaDA at
each eligible two-step round. The selective arm asks only when at least one
scheduled Dream write has top-token probability below 0.80. Both keep Dream's
write when coverage is below 0.50 or normalized JS exceeds 0.20.

Compare each Dream-main arm with standalone Dream and with the corresponding
LLaDA-main arm, paired by document ID. Record actual peer-forward counts and
the number of scored and edited spans by width. The same numerical threshold
can produce different call rates because each model has its own confidence
distribution. The report records generation time, processed tokens, and model
forwards under the same schedule. The report CLI accepts `--selective --owner
dream --llada-owner-trace` and a completed LLaDA-main trajectory for direct
paired role comparisons.

```bash
sbatch --job-name=ctca-selective-js-dream /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_periodic_js_ensemble_slurm.sh --selective-dream
```

Run artifacts are under
`/home/juan.guevara/naruto/dllm-epiplexity/.logs/ctca-gsm8k-dream-llada-selective-js-ensemble-pilot-20261124/<job-id>/`.
The same CPU smoke-test command above covers both ownership directions.
