# 07 — Search for affordable companions

Use the environment and analyzer from the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** stopping at the first unaffordable companion wastes usable entropy
budget. Compare `affordable_companions` against `reference_cheap`, retaining the
same entropy budget 2, cap four, original seeds, confidence exponent zero,
attention representation, conflict penalty, anchors, and selector.

After the always-admitted seed, restrict the next choice to candidates whose
entropy fits the remaining budget. Choose the best current conflict-adjusted
utility and recompute against only accepted companions. Stop when none fit or
the cap is reached. A seed already above budget remains a singleton, preserving
the original exception; this experiment does not change seed admission.

Larger realized groups and fewer iterations are plausible, but correctness is
unresolved until measured. If evaluations fall while accuracy deteriorates beyond
the intended tradeoff, the budget is not sufficient protection against committing
more tokens. A benefit supports improving growth search, not changing attention
orientation or claiming independence.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/07_affordable_companions/slurm.sh
```

The launcher requests two GPUs for four hours and resumes completed documents.
Use the same run tag as stage 01 to compare with its complete cheap baseline;
W&B logging follows the suite settings.
