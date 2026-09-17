# Ablation 3: Eight Dependency Candidates

This ablation changes only the number of dependency candidates evaluated at
each decoding action. It compares eight candidates against the completed
four-candidate reference while retaining:

- GSM8K with five-shot prompting;
- entropy-drop lookahead verification;
- entropy-budget cardinality with budget `2.0`;
- maximum action size `64` (no cap below the decoding block size);
- per-token size-aware verifier scoring;
- sequential candidate evaluation with candidate chunk size one;
- dependency generation seed `42`; and
- two data-parallel GPU ranks.

The shared runner defaults to four candidates, so all Ablation 2 launchers keep
their previous behavior. This launcher sets `PATH_ABLATION_CANDIDATE_BUDGET=8`
without modifying the candidate generator or verifier algorithm.

Submit with:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/3_candidate_budget/run_gsm8k_candidate_budget8.slurm.sh
```

The default output directory is:

```text
/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/3_candidate_budget/candidate_budget8_v1/gsm8k_cot/entropy_budget2.0/seed42
```

## Maximum confidence without lookahead

Use `PATH_ABLATION_SAMPLER_TYPE=max_confidence` together with
`PATH_ABLATION_SEED_STRATEGY=incoming` and
`PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0` for eight corrected IE candidates,
entropy budget `2.0`, and maximum action size `64`. The winning group maximizes
mean token confidence from the current pass; candidate chunk size is unused.

The [N=4 and N=8 maximum-confidence launch commands](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/README.md#maximum-confidence-without-lookahead-entropy-budget-20-n4-or-n8)
write separate results and caches from the entropy-lookahead experiments.

## Parallel rerun

Set `PATH_ABLATION_CANDIDATE_CHUNK_SIZE=8` to evaluate all eight candidates in
one batched lookahead forward at each decoding step. The shared runner passes
this through as `candidate_chunk_size=8` and gives the run a separate output
directory, response cache, and W&B name. The default remains sequential.

For the corrected IE seed, also set `PATH_ABLATION_SEED_STRATEGY=incoming`
and `PATH_ABLATION_SEED_ENTROPY_WEIGHT=1.0`; the I variant uses weight `0.0`.
Seed selection is separate from companion scoring and entropy-budget stopping.

See the [N=4 and N=8 parallel IE rerun commands](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/README.md#parallel-candidate-lookahead-entropy-budget-20-n4-or-n8)
for the uncapped, entropy-budget-2.0 comparison and separate output paths.

The [FP32 entropy retry instructions](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/2_entropy_budget_without_k_limit/README.md#fp32-entropy-in-small-chunks)
keep all eight candidate sequences in one model forward while limiting entropy
intermediates to 64 token distributions at a time. They include focused checks,
an optional eight-example GPU run, and fresh run tags for the full reruns.
