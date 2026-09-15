# 03 — Proposal quality at fixed size

Use the common environment, corpus, and analyzer from the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** the original dependency pool omits useful actions available from
confidence-based construction. At each frozen state, form four fixed-four dependency
candidates and four fixed-four confidence candidates from the same predictions.
The confidence construction reuses the existing confidence-seed/companion generator.
It is a candidate-pool control, not the direct threshold sampler of stage 06.

Continue every distinct action under the same cheap policy. Report each pool's
best attainable outcome and candidate-level branch outcomes, reusing identical
states and actions. This is ground-truth-assisted analysis after generation;
ground truth never selects an inference-time action.

A confidence-pool success when every dependency candidate fails supports proposal
failure. A successful dependency candidate overlooked by a selector supports
scoring failure. Both pools failing does not identify proposal failure: the shared
model or continuation may be responsible. Equal size/count do not match entropy;
candidate summaries preserve that difference for interpretation.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/03_proposals/slurm.sh
```

Optional complete fixed-size reference benchmarks use the original seed mechanics:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/03_proposals/fixed4.slurm.sh
```

The diagnostic requests two GPUs for 24 hours. The optional controls use a
two-task array, one active task at a time, with two GPUs and six hours per task.
W&B logs each worker under the shared suite group; local analysis merges both.
