# 04 — Preserve absolute attention mass

Use the environment and shared states from the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** selected-key normalization inflates dependency estimates when little
attention reaches the selected response keys. Reconstruct full-key softmax, retain
the selected entries without row renormalization, and apply the same sink mask
derived from the original conditional representation. Preserve existing conflict
normalization, seed utility, companion rules, and cheap selection.

The diagnostic compares both representations on the same candidate positions and
actual seed-induced prediction changes. It reuses precedence probes and branches.
Retain regional attention mass so selected, masked, prompt, and available-context
contributions remain inspectable. Turning off only the old reconstruction flag
would be insufficient because sink filtering previously renormalized rows again.

Large conditional edges shrinking is expected mathematically, not evidence of
better decoding. Retain the change only if it predicts stability more usefully
beyond confidence or improves the measured complete-policy tradeoff.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/04_attention_mass/slurm.sh
```

Compare the benchmark with `reference_cheap`. Its runtime includes constructing
the reference sink mask as well as preserving the full-key mass.

The launcher requests two GPUs for 24 hours. Running it after stage 02 permits
reuse of matching probes and continuations. Its complete benchmark runs separately
from those diagnostic forks. Completed work resumes and W&B logging is enabled.
