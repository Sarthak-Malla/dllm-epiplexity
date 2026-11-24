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
[EXPERIMENTS.md](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/EXPERIMENTS.md),
and record it in
[experiments.json](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/experiments.json).

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

## Focused follow-up after the CTCA results audit

The paired GSM8K results disfavor ranking 50/50 fused proposals by normalized
JS agreement under the tested setup. HumanEval also shows a net loss from
static CTCA fusion in both master directions. The endpoints solve different
GSM8K questions, so model diversity still merits testing. The immediate
question is whether a second denoiser can identify a **local change worth
making to a partially decoded sequence**. The current benchmarks mix
changes to token identity with changes to commitment order. [Results audit](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/EXPERIMENTS.md)

### What the implementation measures

- The auxiliary canvas retokenizes each committed run and puts one auxiliary
  mask at each open master slot. This preserves a logical position only. A
  single mask for a two-token native spelling asks a different conditional
  question. Committing and merging runs may also shift downstream auxiliary
  indices. For GSM8K, LLaDA's five-shot prompt and Dream's zero-shot prompt
  provide different conditioning even when sampled text matches. [Canvas construction](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/ctca/canvas.py:212), [prompt construction](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/eval.py:343)
- `exact_shared` overwrites shared ordinary vocabulary rows using raw
  token strings. Unshared and special rows keep the anchor-based fallback.
  Spatial weighting, many-to-one mappings, and subsequent normalization can
  distort confidence. The `centimeters` → `kilometers` trace identifies a
  concrete invalid substitution. The smaller-anchor replays corrected the
  unit but still gave the wrong answer, showing that valid mappings and
  useful reasoning interventions need their own measurements. [Exact-shared policy](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/ctca/relative.py:33), [projection trace](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/EXPERIMENTS.md:1366)
- JS compares master probabilities with projected auxiliary probabilities
  before master-token suppression, while the fused proposal and its
  confidence use the suppressed distribution. It changes the rank of
  positions and leaves the greedy fused proposal at each position intact
  for a fixed canvas. The schedule still requires a fixed number of commits
  per step. In particular, a low score offers no master fallback when a
  contested position must be filled. [Fusion and ranking](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/ctca_sampler.py:692), [fixed-count selection](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/selection.py:56)
- The current live probe supports a single proposal substitution on
  master-selected positions and then continues under the master policy. The
  first admissible disagreement is a feasibility target. Its two
  continuations demonstrate trajectory sensitivity, including changed
  rationale quality with unchanged numeric exact match. To measure useful
  disagreement, sample target states across mask fraction and disagreement
  strata using only features available at intervention time. [Live probe](/home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/live_one_step_ablation.py:57)
- The original same-checkpoint temperature blend is a scheduling control.
  When both members receive identical logits, each positive softmax
  temperature preserves token order, and an arithmetic mix of the two
  probability vectors preserves that order as well. Across-position
  confidence can still change. Also, the same-tokenizer sampler currently
  records JS as a diagnostic and ranks by fused confidence. A clean test
  of model diversity needs genuinely different checkpoints. [Same-tokenizer sampler](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/sampler.py:185)

### Where the related methods act

| Method | Unit of combination | Implication for this study |
|---|---|---|
| AR vocabulary alignment | DEEPEN maps next-token distributions through relative representations. EVA aligns vocabulary distributions and filters suspect contributors. | Use approximate vocabulary alignment as a baseline and measure it independently from commitment order. [DEEPEN](https://proceedings.neurips.cc/paper_files/paper/2024/file/d8a6eb79f8ccaacbe7198a5caf3a0323-Paper-Conference.pdf), [EVA](https://arxiv.org/abs/2404.09492) |
| AR string and byte alignment | Wicks et al. enforce surface agreement during generation. Phan et al. derive next-byte conditionals. SweetSpan cross-scores spans extending a prefix. | Shared text and native rescoring transfer as design ideas. Exact next-byte conversion relies on the AR prefix factorization. [Token-level Ensembling](https://arxiv.org/abs/2502.21265), [Exact Byte-Level Probabilities](https://arxiv.org/abs/2410.09303), [SweetSpan](https://arxiv.org/abs/2409.18583) |
| Output selection | Self-consistency votes over answers from multiple samples. LLM-Blender ranks and fuses complete candidate outputs. | Keep a same-budget stronger-model multi-sample baseline and a heterogeneous output selector that uses a frozen answer-agnostic rule. [Self-Consistency](https://arxiv.org/abs/2203.11171), [LLM-Blender](https://arxiv.org/abs/2306.02561) |
| Masked diffusion ensembles and search | TIE cross-scores partial trajectories and relays one winner wholesale. UnMaskFork branches on denoising actions and can switch tokenizers across models. | Compare against a matched trajectory relay. TIE reports 83.62% GSM8K for entropy relay versus 78.77% for its best constituent. Its setup uses LLaDA-1.5 and Dream, so its published scores serve as evidence for the baseline rather than a comparison with our runs. [TIE](https://arxiv.org/html/2606.16281v1#S5), [UnMaskFork](https://arxiv.org/html/2602.04344v2) |
| Masked diffusion planning and verification | Path Planning separates position planning from denoising. LookUM evaluates unmasking paths with lookahead uncertainty. Plan-Verify-Fill selectively verifies tentative commitments against the decoding backbone. BMC reconstructs reasoning and answers for single-model self-verification. | Test position policy and lookahead in isolation. The contribution would need to come from the joint use of heterogeneous, cross-tokenizer evidence about the same gap. [Path Planning](https://arxiv.org/abs/2502.03540), [LookUM](https://arxiv.org/abs/2511.05563), [Plan, Verify and Fill](https://aclanthology.org/2026.findings-emnlp.1119/), [BMC](https://proceedings.mlr.press/v306/ruan26a.html) |

Commitment delay and counterfactual repair also have direct precedents.
Answer First, Reason Later studies delaying answer positions. LOCKR uses
targeted remasking and counterfactual continuations to repair lock-in.
Deferral and lookahead therefore need equally budgeted single-model controls.
[Answer First, Reason Later](https://arxiv.org/abs/2608.05687), [LOCKR](https://arxiv.org/abs/2609.27220)

### Proposed method

Use **cross-model verification of local masked spans**, with an explicit
choice to commit the proposed replacement, keep the master proposal, or
defer the gap. The research hypothesis is that a cross-model verifier can
identify edits with positive downstream value that a master-only verifier
misses at matched compute. The targeted literature search found trajectory
relay, action branching, and self-verification as precedents. A possible
contribution is cross-tokenizer verification of competing local edits to the
same partially observed text under a fixed commitment budget.

1. **Measure probability mass on exact matches.** Start with ordinary,
   context-verified single-token matches and exclude special tokens from
   auxiliary evidence. Keep unmatched mass explicit. For master distribution
   `p`, exact projected subdistribution `r` of total mass `m`, and auxiliary
   weight `λ`, use `q = (1 − λm)p + λr`. A row with no exact support equals
   the master distribution. Measure retained mass and changed selected
   proposals by model direction, block, mask fraction, and token class.
2. **Compare completions of the same bounded gap.** Start with short,
   currently masked regions with verifiable text boundaries and at most the
   scheduled master commit count. Build a small union of native candidates and
   always include the master completion. Recreate identical visible text on
   both canvases and represent the gap with each tokenizer's required mask
   count. Keep candidate lengths matched within each model in the first test.
   Check neighboring token boundaries by in-context round trip. Score
   candidate-relative margins in each model using a predeclared within-gap
   reveal order and calibrate them on development data. A sum of conditional
   log scores along one order is a path score. The score can vary with reveal
   order and mask count.
3. **Test what the candidate changes.** Branch from identical pre-commit
   canvases. Replace a span atomically only when the verifier accepts it.
   For deferral, spend the same number of slots on other eligible positions
   and use the master choice when the budget forces a decision. In the
   diagnostic, roll out each branch with fresh forwards under the same
   continuation policy and label its final task result. As an online feature,
   test a bounded lookahead that checks remaining masks after inserting each
   candidate. Compare it with native gap scores before adding extra forwards
   to the production policy. Calibrate acceptance against the observed
   probability of helping minus the probability of harming, relative to the
   master branch. Compare rules at the same intervention rate.

### Evidence required to keep developing the method

1. **Reproduce the controls.** Pin the code and checkpoint revisions,
   batch composition, precision, prompt bytes, suppression, and decode
   schedule. Check selected positions and text at the first divergence.
   Use both true single-model controls and both CTCA endpoint controls.
   The endpoint path currently forwards both models even at zero auxiliary
   weight. Evaluate with matched semantic prompts as well as the current
   model-specific prompt policy. [CTCA forward path](/home/juan.guevara/naruto/dllm-epiplexity/dllm/pipelines/tse/ctca_sampler.py:335)
2. **Use paired interventions to identify the source of any gain.** Compare
   master, exact-mass fallback, local span scoring with the master alone,
   and cross-model span scoring, initially with master-selected positions.
   Then cross proposal choices with fixed-budget deferral. Include a
   shuffled-deferral control at the same rate. Report candidate eligibility,
   retained mass, intervention rate, gains and harms per override, and
   first divergent commitments. Put every state from the same prompt on the
   same side of the calibration split. Use teacher-forced corruptions for
   lexical diagnostics and final outcomes to evaluate free-generation edits.
3. **Run one frozen evaluation once the signal survives the diagnostic.**
   Use untouched GSM8K questions and confirm on an external reasoning
   benchmark. Reserve HumanEval as a separate code outcome with its own
   fixed protocol. Primary outcome is paired flexible exact match against
   the stronger native model and a compute-matched output selection control.
   Compare with tuned confidence CTCA, JS rank, and TIE or a documented
   budget-matched relay adaptation. Report protocol-specific confidence
   intervals, forward calls by model, processed tokens, latency, and
   GPU-hours. A small pilot can reject a mechanism or estimate the size of a
   confirmatory study. Test-set tuning cannot establish an accuracy gain.
