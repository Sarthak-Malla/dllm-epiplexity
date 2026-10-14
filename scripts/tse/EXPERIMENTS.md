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

Use `<method>-<models>-<YYYYMMDD>` for experiment IDs and
`<protocol-id>/<experiment-id>` as the W&B group/name prefix. A scientific
experiment can have multiple scheduler attempts, but every attempt must point to
the same immutable config or explicitly start a new experiment.
