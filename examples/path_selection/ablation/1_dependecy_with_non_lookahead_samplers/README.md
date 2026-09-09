# Ablation 1: Dependency Candidates Without Lookahead

This ablation keeps the frozen attention-based dependency candidate generator
but replaces entropy-drop lookahead with a statistic from the current base pass.
It matches the stronger Phase-8 entropy-budget setup: four dependency candidates
are generated per active step, and each candidate contains 1--4 positions whose
cumulative entropy does not exceed 2.0 (apart from its always-included seed).

The three jobs differ only in how the winning candidate is selected:

- `max_confidence`: maximize the mean argmax-token probability.
- `min_entropy`: minimize the mean categorical entropy.
- `min_top2_margin`: minimize the mean difference between the largest and
  second-largest token probabilities.

The implementation averages over candidate positions, which makes the selector
fair across variable candidate sizes. No counterfactual/lookahead model pass is
made. The matched reference is `dependency_entropy_budget_n4`; therefore the
only intended difference is entropy-drop lookahead versus a simple base-pass
selector.

Submit the three entrypoints with `sbatch`:

- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/run_max_confidence.slurm.sh`
- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/run_min_entropy.slurm.sh`
- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/1_dependecy_with_non_lookahead_samplers/run_min_top2_margin.slurm.sh`

All three use GSM8K with five-shot prompting, two data-parallel ranks, seed 42,
and the same dependency configuration as the reduced Phase-8 evaluation.
