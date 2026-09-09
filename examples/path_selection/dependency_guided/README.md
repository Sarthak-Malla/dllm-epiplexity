# Dependency-Guided Entropy-Drop Evaluation

This directory contains only the frozen reduced Phase-8 evaluation plan. Earlier
phase-specific proxy collection, calibration, ablation, and smoke-analysis code
has been removed because it is not part of the final execution path.

## Canonical evaluation

The frozen configuration is
`/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/dependency_guided/p8_reduced_full_plan.json`.
It evaluates the following two methods on GSM8K and HumanEval:

- `dependency_fixed_k4_n4`: four dependency-guided candidates, committing four
  positions from the entropy-drop winner.
- `dependency_entropy_budget_n4`: four dependency-guided candidates with a
  per-candidate entropy budget of `2.0`, maximum action size `4`, and entropy-drop
  per committed token as the final score.

The entropy-budget method can realize action sizes `1`, `2`, `3`, or `4`. The
budgeted constructor grows candidates one position at a time; it does not enforce
the older experimental `1|2|4` action-size list.

Use one of these task-specific entrypoints:

- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_gsm8k_full_dependency_fixed_k4.slurm.sh`
- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_gsm8k_full_dependency_entropy_budget.slurm.sh`
- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_humaneval_full_dependency_fixed_k4.slurm.sh`
- `/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_humaneval_full_dependency_entropy_budget.slurm.sh`

All four entrypoints call
`/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/run_p8_full_task.sh`.

## Runtime code

The final algorithm is split into a small set of reusable components:

- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/dependency.py`:
  reconstruct and filter the attention-derived dependency map.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/candidates.py`:
  shared candidate representation and dependency utility.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/parallel_candidates.py`:
  fixed-size soft-conflict candidates and anchor state.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/adaptive_cardinality.py`:
  entropy-budget candidate growth and size-aware scoring.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/batched_lookahead.py`:
  counterfactual entropy-drop evaluation.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/dependency_guided.py`:
  connect dependency proposals to the decoder.
- `/home/sarthak.malla/dllm-selection-ensemble/dllm/core/samplers/entropy_drop.py`:
  iterative reveal loop and final token commitment.

These components are retained even when they were first written during an earlier
phase because the final Phase-8 algorithm imports them directly.
