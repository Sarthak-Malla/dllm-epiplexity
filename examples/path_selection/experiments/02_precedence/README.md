# 02 — Conflict, value stability, and seed precedence

First collect the shared corpus using the
[suite README](/home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/README.md).

**Hypothesis:** low estimated conflict predicts companions that retain their values
after the seed is revealed. Test every candidate, retaining directional attention,
symmetric conflict, its normalization scale, confidence weighting, and actual penalty.
Compare token flips, probability loss, total variation, and Jensen–Shannon divergence
within confidence/entropy/margin, seed-confidence, size, and stage strata.

For the union of the entropy and cheap winners, freeze positions and compare
simultaneous commitment with seed-first refreshed values. Natural pairs also use
reverse order. Larger groups additionally provide a labeled seed/first-grown-companion
pair with simultaneous and both sequential commitments. Pair results are not
misrepresented as the original larger action.

All branches use baseline anchor-confidence metadata and the same next decision
index, isolating value changes. Probes keep companions masked and do not advance
the carrier's RNG. One actual seed-only forward can supply predictions for several
groups. A singleton shortcut is never cached as if such a forward occurred.

If refresh improves final correctness, simultaneous commitment is harmful. If
values change without correctness gains, instability exists but the extra forward
is not justified. Low conflict predicting stability only through high confidence
does not establish added attention value. Stable values can still be wrong.

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/02_precedence/slurm.sh
```

If the diagnostic supports refresh, benchmark its deployed version independently.
This version uses refreshed anchor confidence and includes the extra model work:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/experiments/02_precedence/seed_first.slurm.sh
```

Both scripts use two GPUs. The diagnostic requests 24 hours for its many paired
continuations; the optional deployed benchmark requests six hours. Neither script
submits the other. Resubmission resumes completed units and W&B run identities.

## Finding to retain from the completed diagnostic

On the 100-problem development corpus, attention conflict predicted initial
companion-value instability beyond the coarse confidence controls inspected.
Among fixed-four initial candidates, the lowest/highest conflict quarters had
33.3%/60.3% companion flip rates, with similar mean confidence (8.89%/8.47%).
This supports retaining conflict as a stability signal; it does not establish
that choosing minimum-conflict companions improves final answer accuracy.
The current soft_full construction already includes a conflict penalty.

All 400 adaptive initial candidates were singletons. Refreshing selected natural
adaptive groups changed none of 503 post-commit states; fixed-four refresh had
25 flexible-correctness improvements and 25 regressions across 519 interventions.
These are same-state interventions with a common continuation, not a full-policy
seed-first benchmark. Companion observations overlap across candidates and pools.

Source results:
[experiment 02](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/experiments/training_free_v1_02_two_gpu).
Filter diagnostics to `experiment == "precedence"`; this directory also contains
attention-mass artifacts from experiment 04.

## Future track: decrease the entropy budget during decoding

**Status:** proposed for future exploration; no implementation or run is scheduled.

**Hypothesis:** a higher initial entropy budget could permit more simultaneous
token commitments and reduce generation cost, while decreasing the budget toward
2 could limit later uncertainty. Test whether this schedule improves the
speed–accuracy tradeoff relative to a constant budget.

| Condition | Budget schedule |
|---|---|
| Reference | Constant 2 |
| Higher-budget control | Constant `B_high` |
| Dynamic budget | Start at `B_high`, decrease toward 2 as decoding progresses |

Define progress by the fraction of response positions revealed, rather than raw
decision count: changing action sizes changes the number of decisions. Freeze
`B_high`, the decay shape, and its endpoint before the confirmation evaluation.
Choose the higher budget after inspecting seed-plus-companion entropy thresholds
on development states, and verify that it actually increases early action sizes.
The seed is always accepted; its entropy counts toward the budget for admitting
companions, so a modest increase may still leave initial actions as singletons.

Keep the model, prompts, seeds, candidate count, maximum action size, selector,
attention-conflict penalty, and commitment mode identical across conditions.
Evaluate changes to companion ranking separately. The first comparison should
retain the experiment 02 maximum action size of four.

Measure actual action sizes and singleton rates by decoding stage, companion
stability, paired final-answer correctness, total model evaluations, and generation
latency. A same-state comparison of larger-budget actions against budget-2 actions
with a common continuation can diagnose local effects; complete-policy runs are
needed to establish the accumulated accuracy and cost of the schedule. Use the
development corpus to select the schedule, then confirm on previously unused
problems without retuning.

The earlier constant-budget ablation at cap 64 scored 68.16% with budget 4 versus
71.19% with budget 2, with fewer model calls at budget 4. This motivates testing
the schedule directly: those results do not identify when the accuracy loss
occurred and do not test a decreasing budget. See the
[ablation report](/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/analysis_20260907/report.md).

Experiment 02 establishes initial instability and an attention-conflict stability
signal, but does not establish that larger initial commitments preserve accuracy.
Equal aggregate correctness between original and refreshed values at fixed
positions is a different comparison from increasing the number of committed
positions. Tightening the budget later cannot revise tokens already committed
under this decoder.
