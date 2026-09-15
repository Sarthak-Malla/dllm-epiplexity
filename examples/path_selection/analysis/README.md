# Historical answer diversity and voting

This analysis compares all seven methods in the 1,319-example report table:
original MDLM, fixed-four dependency decoding, budget-two/cap-four entropy
selection, its maximum-confidence selector, budget-two/cap-64, budget-four/cap-64,
and budget-two/cap-64 with eight candidates. It does not mix in the 300-example
seed experiments or the new 100-example diagnostic suite.

The analyzer reuses existing sample validation and requires all 1,319 document
IDs, identical document/prompt/target contents and hashes, identical extraction
and matching rules, and exact reproduction of reported individual accuracies.
It records input hashes and model arguments. Historical implementations differ;
this comparison does not isolate any single mechanism.

## Measurements

- Strict and flexible individual accuracy, including uniquely solved questions.
- Pairwise normalized-answer agreement, response-text equality, both correct,
  both wrong, and questions solved by only one member of the pair.
- Number of methods correct on each question and at-least-one-correct coverage.
- Seven-method plurality accuracy, true-majority coverage, and all abstentions.
- Paired vote wins/losses and 95% document-bootstrap intervals against every
  method. These are exploratory intervals, not seed-to-seed variation estimates.

An answer receives one vote per method. Invalid extractions abstain. The largest
group wins; ties follow the fixed order above, regardless of correctness. A true
majority needs at least four votes. Normalization follows the saved exact-match
rules; there is no additional numeric-equivalence rule. Voting all seven methods
requires their combined generation cost. Exact total calls remain unavailable
because the historical cheap-selector call count is missing.

Exact token-reveal path diversity cannot be reconstructed for all seven methods.
The MDLM baseline lacks path traces, the cheap selector saved empty step records,
and compact dependency traces omit selected positions. Answer diversity is
reported explicitly as a separate measurement.

## Execute

The user submits the CPU-only job; no model inference or new GPU job is needed:

```bash
sbatch /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/analysis/diversity_vote.slurm.sh
```

The script prepares the conda environment, runs focused validation tests, then
analyzes saved results. It requests two CPUs, 16 GB RAM, and 30 minutes, without
node exclusions. It does not modify the experiment suite or its frozen sources.

Reports are written under
`/home/sarthak.malla/dllm-selection-ensemble/eval_results/path_selection/ablation/diversity_vote_table_job<JOB_ID>`:

- `report.md`: individual results, pairwise comparisons, and voting results.
- `report.tex`: a short LaTeX subsection and table.
- `summary.json`: complete measurements, source/input hashes, and completion flag.
- `votes.jsonl`: every normalized answer, vote, tie, and correctness outcome.

Job logs are under
`/home/sarthak.malla/dllm-selection-ensemble/.logs/path-diversity-vote_<JOB_ID>.out`.
Existing nonempty report directories are rejected. Source evaluation files are
never overwritten. This setup has not run the tests or analysis on the login node.
