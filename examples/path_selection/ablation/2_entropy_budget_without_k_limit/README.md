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
