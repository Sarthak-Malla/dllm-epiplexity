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
