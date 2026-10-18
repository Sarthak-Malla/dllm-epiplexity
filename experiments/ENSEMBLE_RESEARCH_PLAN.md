# Ensemble research plan for masked diffusion language models

## Objective

Determine whether probability-level ensembling improves masked-diffusion
generation, why it helps when it does, and whether the improvement justifies its
additional compute. The central hypothesis is that member diversity provides a
useful signal for both token prediction and the order in which masked positions
are committed.

This document defines research questions rather than runnable experiments. Before
launching a study, create a named experiment and immutable configuration following
the conventions in
[`/home/sarthak.malla/dllm-epiplexity/scripts/tse/EXPERIMENTS.md`](/home/sarthak.malla/dllm-epiplexity/scripts/tse/EXPERIMENTS.md),
and record it in
[`/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json`](/home/sarthak.malla/dllm-epiplexity/scripts/tse/experiments.json).

## Common experimental principles

- Compare methods only under the same protocol, prompt, generation budget,
  decoding temperature, suppression rules, and evaluation set.
- Use the native single-model runs as controls. Report each member separately,
  not only the stronger member.
- Separate **token selection** from **position selection**. An ensemble can alter
  the predicted token, the position committed next, or both.
- Analyze trajectories by remaining-mask fraction or normalized diffusion time so
  runs with different step counts can be compared.
- Treat lower entropy as greater confidence, not automatically as lower
  uncertainty error. Always pair confidence measurements with correctness,
  calibration, or final task success.
- Use teacher-forced corruptions of known reference sequences for unambiguous
  token-level analyses. In free generation, where several rationales or programs
  may be valid, use task success and final-answer correctness rather than raw
  reference-token equality.
- Report uncertainty intervals and the efficiency fields required by the canonical
  experiment ledger. Do not tune and report on the same examples.

## Primary research questions

### Q1. Do we need distributional agreement, or is maximum fused confidence sufficient?

**Meaning.** The fused distribution can provide both a candidate token and a
confidence score. It is unclear whether a separate cross-member agreement term
adds useful information when deciding which position to commit. Here, “maximum
fused confidence” means

\[
c_{i,t}=\max_v q_{i,t}(v),
\]

where \(q\) is the fused token distribution. “Distributional agreement” means a
bounded agreement score derived from the member distributions, such as one minus
normalized Jensen–Shannon divergence. The current locked parity baseline commits
using maximum fused confidence and records agreement only as a diagnostic.

**How to test it.** Keep the fused token distribution fixed and change only the
position score:

1. maximum fused confidence, \(c_{i,t}\);
2. confidence times agreement, \(c_{i,t}a_{i,t}\);
3. agreement alone, as a diagnostic ablation;
4. a shuffled or mismatched agreement control that preserves the marginal score
   distribution but destroys position-specific information.

Use identical commitment budgets and deterministic decoding. Compare final task
quality, the accuracy of newly committed tokens on teacher-forced data, the time
of the first incorrect commitment, and the fraction of errors committed at high
scores. Also measure the rank correlation between confidence and agreement; if
they induce nearly identical position rankings, agreement is redundant.

**Decision criterion.** Agreement is useful only if it improves held-out task or
commitment accuracy beyond maximum fused confidence, consistently across seeds or
deterministic repeats, without merely reducing the number of committed tokens.

### Q2. Does ensembling reduce useful uncertainty during early diffusion?

**Meaning.** Early denoising has little visible context and may contain the most
epistemic uncertainty. Ensembling may sharpen correct shared beliefs and expose
contested positions. Merely lowering entropy is not sufficient: a confidently
wrong ensemble is worse calibrated, not less uncertain in a useful sense.

**How to test it.** At fixed mask fractions, evaluate each member and the ensemble
on the same masked canvases. Divide the trajectory into early, middle, and late
phases and record:

- predictive entropy and top-1 margin;
- token negative log-likelihood and Brier score on teacher-forced references;
- expected calibration error and risk-coverage curves;
- top-1 token accuracy;
- member disagreement and fused confidence;
- correctness of tokens actually selected for commitment.

Run both an on-policy analysis using generated canvases and an off-policy analysis
using shared reference corruptions. The latter prevents different trajectories
from creating incomparable contexts.

**Decision criterion.** The ensemble reduces useful early uncertainty if it
improves proper scoring rules or selective-prediction risk in the early phase. A
drop in entropy without improved correctness or calibration does not support the
hypothesis.

### Q3. Does cross-member disagreement predict commitment error beyond single-model confidence?

**Meaning.** This asks whether disagreement contains an independent error signal,
rather than being a more complicated restatement of low confidence. A positive
result motivates agreement-aware commitment or deferral.

**How to test it.** For every candidate commitment, log member confidence, fused
confidence, entropy, margin, agreement or divergence, diffusion time, position,
and eventual correctness. Then:

1. bin examples by single-model or fused confidence and compare error rates for
   low- and high-disagreement positions within every bin;
2. fit a held-out error predictor using confidence and diffusion time;
3. add disagreement features and measure the improvement in log loss, Brier
   score, AUROC/AUPRC, and calibration;
4. repeat by diffusion phase and task to detect conditional effects.

Use grouped splits by prompt so tokens from one generation never occur in both
training and evaluation splits. Prefer a simple calibrated logistic model first;
a complex predictor could manufacture apparent value from overfitting.

**Decision criterion.** Disagreement has incremental value if it improves
held-out error prediction and separates error rates within matched confidence
bins. Statistical significance alone is insufficient; the effect must be large
enough to change position ranking or improve generation.

### Q4. Does ensembling help particular tasks or task characteristics?

**Meaning.** Complementarity may depend on the type of generation. Mathematical
reasoning may benefit from semantic or arithmetic diversity, code may be harmed
by small syntax disagreements, and knowledge-heavy tasks may benefit when members
have different coverage.

**How to test it.** Start with the already locked GSM8K and HumanEval protocols.
For each task, compare both native members with the ensemble using matched
generation settings. Report absolute ensemble gain over the stronger member, not
only over the weaker member. Annotate or stratify examples by observable
properties such as:

- reasoning length or generated length;
- early member agreement;
- whether members propose different final answers;
- code versus natural-language structure;
- formatting or premature-termination errors;
- problem difficulty, using a predeclared proxy.

If GSM8K and HumanEval show different behavior, add one held-out task per missing
category—such as knowledge QA, instruction following, or constrained generation—
before making a general claim. Choose and record the benchmark before inspecting
ensemble results.

**Decision criterion.** Claim task-specific benefit only when the ensemble beats
the stronger matched native member and the effect repeats on a second benchmark
with the same proposed task characteristic.

## Supporting research questions

### Q5. Is there enough member complementarity to make ensembling worthwhile?

**Meaning.** An ensemble cannot help much when both members make the same errors.
This question measures the available headroom before optimizing a fusion rule.

**How to test it.** Compute member accuracy, error overlap, disagreement rate, and
oracle accuracy obtained by selecting a correct member whenever either succeeds.
At token level, perform the analysis on shared teacher-forced canvases; at sequence
level, use final task correctness. Stratify the oracle gap by task and diffusion
phase.

**Decision criterion.** A small oracle-over-stronger-member gap indicates that
fusion is unlikely to justify its cost. A large oracle gap paired with weak actual
fusion indicates a selection, calibration, or alignment problem.

### Q6. Do gains come from token fusion, commitment scheduling, or their interaction?

**Meaning.** Probability fusion can change the selected token, while a fused or
agreement-aware score can change reveal order. These are different mechanisms.
For a same-model temperature ensemble, temperature preserves within-position
token ranking, so its primary possible effect is commitment order rather than new
token knowledge.

**How to test it.** Run a factorial ablation:

| Token distribution | Position score | Interpretation |
|---|---|---|
| Master only | Master confidence | Native trajectory baseline |
| Fused | Master confidence | Token-fusion effect |
| Master only | Ensemble score | Scheduling effect |
| Fused | Ensemble score | Combined method and interaction |

Keep the number of commits per step fixed. Compare final task quality and trace
how often each condition changes token identity or commitment order relative to
the native trajectory.

**Decision criterion.** Attribute the method’s benefit only to components whose
controlled ablations improve the matched baseline.

### Q7. Does the most reliable member change across diffusion time or positions?

**Meaning.** Static averaging assumes that member quality is constant. Dynamic
weighting is justified only if relative reliability changes predictably across
the trajectory or across token types.

**How to test it.** On shared canvases, measure per-member token accuracy,
calibration, and proper scoring rules by mask fraction, position class, and task.
Measure how often the identity of the better member changes within a trajectory.
Test whether early-step signals predict which member will be correct on held-out
data.

**Decision criterion.** Explore dynamic weights only if reliability switches are
common and predictable. Otherwise retain static weights as the simpler baseline.

### Q8. Are the results driven by calibration or cross-tokenizer projection artifacts?

**Meaning.** CTCA combines heterogeneous models only after mapping the auxiliary
distribution into the master vocabulary. Apparent agreement, disagreement, or
quality changes may arise from projection loss rather than model knowledge.

**How to test it.** Measure shared-token coverage, probability mass retained by
projection, top-1 preservation, round-trip consistency, and divergence before and
after projection on controlled text. Compare raw and calibrated member
probabilities. Where possible, include a same-tokenizer heterogeneous-model
control to separate model diversity from tokenizer alignment.

**Decision criterion.** Interpret CTCA as model complementarity only when the
effect survives calibration and is not concentrated in low-fidelity projection
cases.

### Q9. Does the quality improvement justify the additional compute?

**Meaning.** A two-member ensemble approximately doubles model forward work and
may require an additional GPU. Accuracy alone is therefore an incomplete result.

**How to test it.** Report wall time, GPU-hours, peak memory, total forward passes,
and task quality. Compare the ensemble against stronger single-model decoding at
matched wall-time and matched forward-pass budgets, for example by giving the
native model more denoising steps.

**Decision criterion.** Present a quality-compute frontier. Call the ensemble
efficient only if it is competitive with native decoding under a matched resource
budget; otherwise describe it as test-time scaling for quality.

## Recommended execution order

1. Measure oracle complementarity and projection fidelity (Q5, Q8).
2. Collect shared-canvas uncertainty trajectories (Q2, Q3, Q7).
3. Run the token-fusion versus scheduling factorial ablation (Q6).
4. Compare maximum fused confidence with agreement-aware scoring (Q1).
5. Complete matched task studies on GSM8K and HumanEval, then expand only when
   their results justify a task-generalization study (Q4).
6. Construct the quality-compute frontier for every promising method (Q9).

This order avoids spending full-generation compute on mechanisms for which there
is no measurable complementarity or reliable online selection signal.
