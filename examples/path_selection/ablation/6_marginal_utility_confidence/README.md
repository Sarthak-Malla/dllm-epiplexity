# Ablation 6: Marginal utility with confidence weighting

Test higher marginal thresholds and confidence weighting separately. The default
is a diagnostic sweep over the first 128 GSM8K questions. Token temperature stays
0.0; candidate seed position temperature stays 1.0. All configurations use four
soft-full candidates, entropy-drop lookahead, per-token scoring, and maximum
action size 64 within a 64-token block. Entropy-budget stopping is disabled.

| Array task | Confidence exponent | Marginal threshold τ |
|---|---:|---:|
| 0 | 0.0 | 1.0 |
| 1 | 0.0 | 2.0 |
| 2 | 0.0 | 4.0 |
| 3 | 1.0 | 0.0 |
| 4 | 1.0 | 0.5 |
| 5 | 1.0 | 1.0 |
| 6 | 1.0 | 2.0 |
| 7 | 1.0 | 4.0 |

Confidence exponent one multiplies entropy-weighted attention utility by the
predicted token's probability; anchor support is added separately. The exponent
affects seed ranking and candidate growth. The committed-anchor confidence
threshold remains 0.8. Seeds are always accepted, so τ is not a hard minimum
confidence requirement.

Submit the diagnostic sweep:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/6_marginal_utility_confidence/run_gsm8k_marginal_utility_confidence.slurm.sh
```

Each configuration requests two GPUs, 32 CPUs, 64 GB RAM, and nine hours. The
array runs at most two configurations concurrently (four GPUs total). Override
with `--array=0-7%1` for one configuration at a time, or select task indices.

After inspecting the diagnostic sweep, evaluate selected configurations on full
GSM8K. For example, task 5 uses confidence exponent one and τ=1:

```bash
sbatch --array=5 --export=ALL,ABLATION6_LIMIT=full /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/6_marginal_utility_confidence/run_gsm8k_marginal_utility_confidence.slurm.sh
```

`ABLATION6_LIMIT` accepts a positive integer or `full`. To inspect all resolved
configurations after sourcing the shell setup and activating `dllm`:

```bash
for task in 0 1 2 3 4 5 6 7; do
    ABLATION2_DRY_RUN=1 SLURM_ARRAY_TASK_ID=${task} bash /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/6_marginal_utility_confidence/run_gsm8k_marginal_utility_confidence.slurm.sh
done
```

Diagnostic results and caches are isolated under
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/6_marginal_utility_confidence/marginal_confidence_v1/gsm8k_cot/confidence_exponent<eta>/marginal_utility_tau<tau>/limit128/seed42`.
Full runs omit the `limit128` component. W&B names identify both parameters and
the sample limit. `ABLATION6_RUN_TAG` overrides the run tag. Each run keeps the
five-shot prompt, 256-token generation length, proposal seed 42, and evaluation
seed tuple `0,1234,1234,1234` used by the earlier ablations.

Compare accuracy, first-action size and entropy, next-pass consistency, and model
calls. Existing Ablation 4 runs supply exponent-zero references at τ=0 and 0.5;
the completed entropy-budget-2 reference remains at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42`.
For the diagnostic sweep, compare the same first 128 documents from those saved
samples, rather than their full-dataset aggregate scores.
