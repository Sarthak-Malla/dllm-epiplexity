# 05 — Block restriction versus full-response eligibility

Prepare the environment using the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** full-response selection exposes useful confident positions that
64-token blocks postpone. Compare the confidence-only direct sampler with blocks
64 and 256, matching confidence threshold, minimum one, maximum four, and all token
settings. This comparison has no attention reconstruction or anchor history, so
block size primarily changes which masked positions can be revealed.

The underlying model already receives the entire response canvas. A larger block
does not automatically lower each forward's sequence cost or resolve simultaneous
token dependencies. Possible gains are larger useful groups or better ordering;
possible losses are premature later-reasoning commitments. Measure complete final
accuracy, action sizes, total evaluations, and synchronized generation time.

The two-GPU launcher runs six array tasks, covering all three thresholds at both
block sizes. Each task requests four hours. The array runs one task at a time,
so it uses two GPUs in total. Threshold selection sees both block sizes.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_block_scope/slurm.sh
```

With a four-job submission limit, use the single-job launcher for stages 05 and
06 instead. Slurm counts all six array tasks toward that limit even with `%1`.

```bash
sbatch --export=ALL,TF_RUN_TAG=training_free_v1_05_06_two_gpu,WANDB_RUN_GROUP=training_free_v1_05_06_two_gpu \
    /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/05_06_sequential.slurm.sh
```

This requests two GPUs for 24 hours and runs nine distinct configurations in
sequence: all six stage-05 configurations, followed by the three incoming-attention
stage-06 configurations. Stage 06 shares stage 05's confidence baselines. It uses
one queue slot, records W&B logs, and applies no node exclusions. Do not also
submit the original 05/06 arrays against this output directory. If the allocation
expires, resubmit the same command to resume completed problem units.
