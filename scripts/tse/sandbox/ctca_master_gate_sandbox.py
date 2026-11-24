"""Run the CPU agreement/uncertainty study (no model checkpoints).

    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/sandbox/ctca_master_gate_sandbox.py

Reports are synthetic diagnostics, not benchmark accuracy or threshold advice.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tse.divergence import entropy, jensen_shannon_divergence
from dllm.pipelines.tse.fusion import fuse_probabilities
from dllm.pipelines.tse.selection import select_positions


STUDY = "ctca-toy4-mastergate-20261118"
VERSION = 2
EXAMPLES = {
    "confident_correct_agreement": ([.9, .1, 0, 0], [.8, .2, 0, 0], 0),
    "confident_conflict": ([.8, .2, 0, 0], [0, .95, .05, 0], 0),
    "identical_uncertainty": ([.25] * 4, [.25] * 4, None),
    "uncertain_master_correct_assistant": ([.2, .3, .25, .25], [.95, .05, 0, 0], 0),
    "confident_master_uncertain_assistant": ([.9, .1, 0, 0], [.25] * 4, 0),
    "correct_minority_assistant": ([.1, .8, .1, 0], [.95, .05, 0, 0], 0),
    "shared_confident_mistake": ([.05, .95, 0, 0], [.1, .9, 0, 0], 0),
}


def validate(probabilities):
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("Expected [positions, 4] probabilities")
    if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]))):
        raise ValueError("Probabilities must be finite, nonnegative, and normalized")


def measures(master, auxiliary):
    """Return symmetric agreement and local entropy in common coordinates."""
    validate(master)
    validate(auxiliary)
    if master.shape != auxiliary.shape:
        raise ValueError("Distribution shapes must match")
    return {
        "js": (1 - jensen_shannon_divergence(master, auxiliary, .5) / math.log(2)).clamp(0, 1),
        "overlap": torch.minimum(master, auxiliary).sum(-1).clamp(0, 1),
        "master_uncertainty": entropy(master) / math.log(4),
        "auxiliary_uncertainty": entropy(auxiliary) / math.log(4),
    }


def gate(master, auxiliary, metric="js", rule="both", agreement_threshold=.5,
         uncertainty_threshold=.5):
    """Gate whole distributions; equality accepts and failure uses master."""
    if metric not in {"js", "overlap"} or rule not in {"both", "one", "agreement_only"}:
        raise ValueError("Unknown metric or rule")
    if not (0 <= agreement_threshold <= 1 and 0 <= uncertainty_threshold <= 1):
        raise ValueError("Thresholds must be in [0, 1]")
    scores = measures(master, auxiliary)
    accepted = scores[metric] >= agreement_threshold
    if rule != "agreement_only":
        combine = torch.maximum if rule == "both" else torch.minimum
        accepted &= combine(scores["master_uncertainty"], scores["auxiliary_uncertainty"]) <= uncertainty_threshold
    output = torch.where(accepted[:, None], fuse_probabilities(master, auxiliary, .5), master)
    return output, accepted, scores


def tied(probabilities):
    return torch.isclose(probabilities, probabilities.max(-1, keepdim=True).values,
                         atol=1e-7, rtol=0).sum(-1) > 1


def simplex():
    return torch.tensor([values for values in itertools.product(range(11), repeat=4)
                         if sum(values) == 10], dtype=torch.float32) / 10


def projection_maps():
    identity = torch.eye(4)
    return {"identity": identity, "swapped": identity[[1, 0, 2, 3]],
            "collapsed": torch.tensor([[1., 0, 0, 0]] * 4)}


def correlated_cases():
    """Balance every labelled template across all 24 token permutations.

    Models share a known answer before mapping corruption. Equal template
    weights are a controlled stress test, not a realistic error frequency.
    """
    cases = []
    for name, (p, q, answer) in EXAMPLES.items():
        if answer is None:
            continue
        for permutation in itertools.permutations(range(4)):
            cases.append({"case": name, "permutation": list(permutation),
                          "master": [p[index] for index in permutation],
                          "auxiliary": [q[index] for index in permutation],
                          "answer": permutation.index(answer)})
    return cases


def outcome_counts(master, blend, output, accepted, answers):
    """Compare all policies on the same untied cases; count ties separately."""
    eligible = ~tied(master) & ~tied(blend)
    master_correct = master.argmax(-1) == answers
    blend_correct = blend.argmax(-1) == answers
    final_correct = output.argmax(-1) == answers
    return {
        "labelled_comparable_cases": int(eligible.sum()),
        "labelled_excluded_ties": int((~eligible).sum()),
        "labelled_master_correct": int((eligible & master_correct).sum()),
        "labelled_ungated_correct": int((eligible & blend_correct).sum()),
        "labelled_gated_correct": int((eligible & final_correct).sum()),
        "labelled_helpful_overrides": int((eligible & ~master_correct & final_correct).sum()),
        "labelled_harmful_overrides": int((eligible & master_correct & ~final_correct).sum()),
        "labelled_prevented_harm": int((eligible & master_correct & ~blend_correct & ~accepted).sum()),
        "labelled_rejected_corrections": int((eligible & ~master_correct & blend_correct & ~accepted).sum()),
    }


def example_report():
    reports = []
    for name, (p, q, answer) in EXAMPLES.items():
        master, native = torch.tensor([p]), torch.tensor([q])
        for mapping, matrix in projection_maps().items():
            auxiliary = native @ matrix
            for metric, rule in itertools.product(("js", "overlap"), ("agreement_only", "both", "one")):
                output, accepted, scores = gate(master, auxiliary, metric, rule)
                token = int(output.argmax(-1)[0])
                master_token = int(master.argmax(-1)[0])
                is_tied = bool(tied(output)[0])
                # No correctness judgement is made for ambiguous maxima.
                comparable = answer is not None and not is_tied and not bool(tied(master)[0])
                reports.append({"case": name, "mapping": mapping, "metric": metric,
                                "rule": rule, "answer": answer, "master": p,
                                "auxiliary_native": q, "auxiliary_projected": auxiliary[0].tolist(),
                                "auxiliary_native_uncertainty": float(entropy(native)[0] / math.log(4)),
                                **{k: float(v[0]) for k, v in scores.items()},
                                "accepted": bool(accepted[0]), "output": output[0].tolist(),
                                "token": token, "argmax_tie": is_tied,
                                "correct": token == answer if comparable else None,
                                "helpful_override": comparable and token == answer and master_token != answer,
                                "harmful_override": comparable and token != answer and master_token == answer})
    return reports


def canvas_report():
    master = torch.tensor([[.6, .4, 0, 0], [.65, .35, 0, 0], [.7, .3, 0, 0]])
    auxiliary = torch.tensor([[0, .99, .01, 0], [.7, .3, 0, 0], [.7, .3, 0, 0]])
    output, accepted, _ = gate(master, auxiliary, agreement_threshold=.75,
                               uncertainty_threshold=1)
    positions = torch.tensor([[0, 0], [0, 1], [0, 2]])
    result = {"accepted": accepted.tolist(), "commit_budget": 2,
              "agreement_threshold": .75, "uncertainty_threshold": 1}
    for name, distribution in (("ungated", (master + auxiliary) / 2), ("gated", output)):
        confidence, tokens = distribution.max(-1)
        selected, selected_tokens = select_positions(confidence, tokens, positions, torch.tensor([2]))
        result[name] = {"proposals": tokens.tolist(), "confidence": confidence.tolist(),
                        "selected_positions": selected.tolist(), "selected_tokens": selected_tokens.tolist()}
    return result


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = simplex()
    master = grid.repeat_interleave(len(grid), dim=0)
    native = grid.repeat((len(grid), 1))
    rows = []
    for mapping, matrix in projection_maps().items():
        auxiliary = native @ matrix
        scores = measures(master, auxiliary)
        blend = fuse_probabilities(master, auxiliary, .5)
        changed = blend.argmax(-1) != master.argmax(-1)
        blend_ties, master_ties = tied(blend), tied(master)
        labelled = correlated_cases()
        labelled_master = torch.tensor([value["master"] for value in labelled])
        labelled_native = torch.tensor([value["auxiliary"] for value in labelled])
        labelled_auxiliary = labelled_native @ matrix
        answers = torch.tensor([value["answer"] for value in labelled])
        labelled_blend = fuse_probabilities(labelled_master, labelled_auxiliary, .5)
        for metric in ("js", "overlap"):
            for rule in ("agreement_only", "both", "one"):
                for ai in range(21):
                    for ui in ([None] if rule == "agreement_only" else range(11)):
                        accepted = scores[metric] >= ai / 20
                        if ui is not None:
                            combine = torch.maximum if rule == "both" else torch.minimum
                            accepted &= combine(scores["master_uncertainty"], scores["auxiliary_uncertainty"]) <= ui / 10
                        final_ties = torch.where(accepted, blend_ties, master_ties)
                        example_output, example_accepted, _ = gate(
                            labelled_master, labelled_auxiliary, metric, rule,
                            ai / 20, 1 if ui is None else ui / 10)
                        rows.append({"mapping": mapping, "metric": metric, "rule": rule,
                                     "agreement_threshold": ai / 20,
                                     "uncertainty_threshold": "" if ui is None else ui / 10,
                                     "pairs": len(master), "fusion_rate": float(accepted.float().mean()),
                                     "changed_token_rate": float((accepted & changed).float().mean()),
                                     "tie_rate": float(final_ties.float().mean()),
                                     **outcome_counts(labelled_master, labelled_blend, example_output,
                                                      example_accepted, answers)})
        scores["auxiliary_native_uncertainty"] = entropy(native) / math.log(4)
        rows_summary = {key: float(value.mean()) for key, value in scores.items()}
        (output_dir / f"{mapping}_summary.json").write_text(json.dumps(rows_summary, indent=2) + "\n")
    with (output_dir / "thresholds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"study_id": STUDY, "version": VERSION, "device": "cpu", "grid_distributions": len(grid),
              "pairs_per_mapping": len(master), "default_example_thresholds": {"agreement": .5, "uncertainty": .5},
              "tie_policy": "lowest token index; tied cases excluded from correctness comparisons",
              "limitations": ["Synthetic grid is not a model distribution or benchmark.",
                              "No threshold is recommended from this study.",
                              "Projection can create false confidence; mask lengths are not modelled."],
              "labelled_design": "Six templates, each under all 24 vocabulary permutations; fixed maps applied afterward.",
              "grid_design": "Unlabelled independent pairs: permutation-invariant sanity check, not mapping robustness evidence.",
              "examples_design": "Canonical illustrative examples only; use balanced labelled counts for comparisons.",
              "correlated_cases": correlated_cases(),
              "examples": example_report(), "canvas": canvas_report()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"study_id": STUDY, "output_dir": str(output_dir),
                      "distributions": len(grid), "threshold_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".logs" / STUDY / "v2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir.resolve())
