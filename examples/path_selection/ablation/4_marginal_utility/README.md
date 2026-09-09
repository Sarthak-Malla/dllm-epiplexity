# Ablation 4: Marginal-Utility Stopping

This ablation replaces entropy-budget stopping with the marginal gain in the
`soft_full` objective. Each candidate retains its seed and accepts further
positions only while the best remaining marginal gain exceeds the threshold.
The three thresholds are `0.0`, `0.25`, and `0.5`; `0` is accepted as `0.0`.
The entropy budget is not used to stop growth in these runs.

All runs use full GSM8K with five-shot prompting, four candidates, maximum action
size 64, entropy-drop lookahead, per-token scoring, generation seed 42, and the
same two-GPU allocation as Ablation 2. The reference is the completed run at
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/2_entropy_budget_without_k_limit/vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42`.

Submit all three jobs:

```bash
for tau in 0.0 0.25 0.5; do
    sbatch --job-name="abl4-tau${tau}" \
        --export="ALL,ABLATION4_UTILITY_THRESHOLD=${tau}" \
        /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/4_marginal_utility/run_gsm8k_marginal_utility.slurm.sh
done
```

Results and response caches are separated by threshold under
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/4_marginal_utility/marginal_utility_v1/gsm8k_cot/marginal_utility_tau<tau>/seed42`.
W&B names also contain the threshold. `ABLATION4_RUN_TAG` overrides the run tag.
Completed runs are skipped using the existing runtime completion marker.

Inspect a configuration without loading the model or creating directories:

```bash
ABLATION2_DRY_RUN=1 ABLATION4_UTILITY_THRESHOLD=0.25 bash /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/4_marginal_utility/run_gsm8k_marginal_utility.slurm.sh
```

Compare flexible-extract exact match, model calls, action sizes, and stopping
reasons against the reference. No baseline rerun is required.

The completed September 8 audit finds that every marginal run commits more than
half of the first block immediately, collapsing flexible exact match to
3.11–3.49%. See the [diagnosis and measurements](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/analysis_marginal_utility_20260908/report.md).
Reproduce the measurements on CPU with
`python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/4_marginal_utility/analyze_results.py`
after sourcing the shell setup and activating the `dllm` environment.
