# 06 — Confidence admission with optional attention ranking

Use the shared environment and benchmark conventions in the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** incoming attention adds useful ranking information once confidence
controls admission. At thresholds 0.80, 0.90, and 0.95, admit qualified positions,
reveal at most four, and use the highest-confidence singleton when none qualify.
The baseline ranks by confidence and performs no attention capture or lookahead.

The attention arm ranks qualified positions by
`confidence[i] * sum_j A[j,i] * H[j]`, where `j` ranges over currently eligible
masked positions and `A` retains full-key-normalized attention mass. Uncertain
positions below the admission threshold still contribute as possible dependents.
Both arms commit the same type of current-pass token predictions. There is no
additional outgoing penalty, learned gate, or conflict threshold.

Compare at each threshold before choosing a development operating point. Count
attention reconstruction latency even though it adds no model forward. Comparable
accuracy at more cost does not justify attention; a useful result improves the
accuracy–evaluation or accuracy–latency tradeoff beyond confidence alone.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/06_confidence_groups/slurm.sh
```

All twelve combinations of the three thresholds, block sizes 64/256, and rankings
`confidence`/`incoming` have explicit arm names. Each command runs only its named
arm; block256/incoming is a later interaction comparison, not the initial clean
attention ablation. Benchmark `seed_first` only after reviewing stage 02.

The launcher selects only the six clean block64 arms. Its array runs one task at
a time, using two GPUs and four hours per task. Completed confidence baselines
from stage 05 are reused. W&B progress and final local analysis distinguish each
threshold and ranking.

For the four-job submission limit, use
[the combined single-job launcher](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_06_sequential.slurm.sh)
documented in
[stage 05](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_block_scope/README.md).
It runs both stages sequentially in one queue slot and generates each shared
confidence baseline once. No separate stage-06 submission is needed.
