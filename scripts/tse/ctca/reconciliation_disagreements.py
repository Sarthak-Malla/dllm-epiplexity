"""Run the expanded reconciliation study on the Slurm GPU allocation.

    sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_reconciliation_disagreements_slurm.sh

For tokenization-only checks, activate dllm on cscc-cpu-p and run this file
with --preflight. The live study loads both pinned models on one GPU.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tie.relay import RelayError, Vocabulary
from dllm.pipelines.tie.sampler import TIEMember
from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_model, get_tokenizer
from scripts.tse.ctca.independent_reconciliation import CHECKPOINTS, INDEPENDENT_STEPS, Fixture, Pilot, select_proposal


STUDY = "ctca-llada-dream-reconciliation-disagreements-pilot-20261124"
INVENTORIES = ((4, 9, 7), (6, 7, 8), (5, 8, 6), (3, 8, 9),
               (8, 6, 5), (7, 5, 8), (4, 7, 9), (5, 9, 8))
REGISTERS = (
    ("train", "train", " Toul", " Pau", " north", " south"),
    ("parcel", "parcel", " Pau", " Toul", " south", " north"),
    ("shuttle", "shuttle", " Toul", " Pau", " west", " east"),
    ("freight", "freight", " Pau", " Toul", " east", " west"),
)
LABELS = (
    ("tape", "paper tape", " centimeters every", " centimeters per", " green", " blue"),
    ("cord", "measuring cord", " centimeters per", " centimeters every", " blue", " green"),
    ("ruler", "wooden ruler", " centimeters each", " centimeters per", " red", " black"),
    ("scale", "digital scale", " centimeters every", " centimeters each", " black", " red"),
)
LINE_ENDINGS = (
    ("engine_log", "engine", "period", "exclamation mark", ".\n", "!\n"),
    ("alert_log", "door", "exclamation mark", "period", "!\n", ".\n"),
)


@dataclass(frozen=True)
class Truth:
    text: str
    foils: tuple[str, ...]
    literal: bool = False

    def label(self, text):
        if text is None:
            return "unfilled"
        if self.literal:
            return "correct" if text == self.text else "wrong"
        if re.fullmatch(r"[+-]?[0-9]+", text.strip()):
            return "correct" if int(text) == int(self.text) else "wrong"
        return "unlabeled"


@dataclass(frozen=True)
class Case:
    fixture: Fixture
    truth: tuple[Truth, ...]
    foils: dict[int, str]
    skipped_foils: dict[int, dict[str, str]]

    def eligible(self):
        return next(iter(self.fixture.widths.values()))

    def width_pair(self, gap):
        return tuple(self.fixture.widths[name][gap] for name in CHECKPOINTS)


def case_bank():
    """Declare passage contexts and answer/foil pairs before running either model."""
    for index, (packs, count, removed) in enumerate(INVENTORIES, 1):
        total = packs * count
        left = total - removed
        yield (
            f"stock_{index}",
            f"A shop has {packs} trays with {count} tickets per tray and sells {removed} tickets. "
            f"An unchecked note claims that {left + 1} remain. Calculate from the tray counts and use Arabic digits.",
            ("There are", " tickets at the start. Selling", " tickets leaves", " tickets in stock."),
            (Truth(f" {total}", (f" {total - 1}",)), Truth(f" {removed}", (f" {removed + 1}",)),
             Truth(f" {left}", (f" {left + 1}",))),
        )
    for name, vehicle, city, wrong_city, direction, wrong_direction in REGISTERS:
        yield (
            name,
            f"The official {vehicle} register lists the town '{city.strip()}' and heading '{direction.strip()}'. "
            f"An old draft says '{wrong_city.strip()}' and '{wrong_direction.strip()}'. "
            "Copy the official register verbatim into the gaps.",
            (f"The {vehicle} leaves from", " and heads", " according to the official register."),
            (Truth(city, (wrong_city, " Pau", " Toul"), True), Truth(direction, (wrong_direction,), True)),
        )
    for name, object_name, phrase, wrong_phrase, color, wrong_color in LABELS:
        yield (
            name,
            f"A {object_name} record says 'two{phrase} hour' and marks its color as '{color.strip()}'. "
            f"The draft used 'two{wrong_phrase} hour' and '{wrong_color.strip()}'. "
            "Copy the record's exact wording and color.",
            (f"The {object_name} record says two", " hour and lists the color as", "."),
            (Truth(phrase, (wrong_phrase, " centimeters an", " centimeters per", " centimeters every"), True),
             Truth(color, (wrong_color,), True)),
        )
    for name, object_name, correct_mark, wrong_mark, ending, wrong_ending in LINE_ENDINGS:
        yield (
            name,
            f"The official {object_name} log ends its first line with a {correct_mark} followed by a line break. "
            f"A draft used a {wrong_mark}. Copy the official punctuation exactly. The next check is at noon.",
            (f"The {object_name} is ready", "Next check takes place at", "."),
            (Truth(ending, (wrong_ending,), True), Truth(" noon", (" dusk",), True)),
        )


def prepare_cases(vocabularies):
    """Retain contextual round trips and match each foil's native mask widths."""
    cases = []
    for name, question, visible, labels in case_bank():
        fixture = Fixture(name, question, list(visible), [label.text for label in labels],
                          {name: {} for name in vocabularies}, {})
        case = Case(fixture, labels, {}, {})
        for gap, label in enumerate(labels):
            try:
                paths = fixture.candidates(vocabularies, gap, label.text)
            except RelayError as error:
                fixture.excluded[gap] = str(error)
                continue
            for name, ids in paths.items():
                fixture.widths[name][gap] = len(ids)
            for foil in dict.fromkeys(label.foils):
                if label.label(foil) != "wrong":
                    continue
                try:
                    alternative = fixture.candidates(vocabularies, gap, foil)
                    if any(len(ids) != len(paths[name]) for name, ids in alternative.items()):
                        raise RelayError("foil uses a different mask width")
                except RelayError as error:
                    case.skipped_foils.setdefault(gap, {})[foil] = str(error)
                    continue
                case.foils[gap] = foil
                break
        if case.eligible():
            cases.append(case)
        else:
            raise ValueError(f"No gap passed tokenizer preflight in {fixture.name}")
    paired_widths = {case.width_pair(gap) for case in cases for gap in case.foils}
    unequal = {widths for widths in paired_widths if widths[0] != widths[1]}
    if len(unequal) < 2:
        raise ValueError(f"Expected controlled foils in two unequal native-width groups, found {sorted(unequal)}")
    return cases


def ranking(proposals):
    """Rank by average native mean log probability and check both preferences."""
    winner = select_proposal(proposals)
    if len(proposals) < 2:
        return winner, None
    margins = {
        name: score - max(item["scores"][name] for item in proposals if item is not winner)
        for name, score in winner["scores"].items()
    }
    return winner, margins


class DisagreementPilot(Pilot):
    """Change the commit decision only when two distinct drafts can be compared."""

    require_unanimity = False

    def reconcile_commits(self, solved, drafts, gate=None):
        decisions = super().reconcile_commits(solved, drafts, gate)
        for decision in decisions:
            if decision["scenario"] != "both_disagree":
                continue
            _, margins = ranking(decision["scored"])
            decision["unanimity_margins"] = margins
            if self.require_unanimity and margins is not None and min(margins.values()) <= 0:
                decision["accepted"] = None
        return decisions

    def natural_runs(self):
        native = self.native_baselines()
        native_cost = self.cost_summary()
        arms = {}
        for name, require_unanimity in (("carry_joint", False), ("carry_unanimous", True)):
            self.require_unanimity = require_unanimity
            self.cost.clear()
            self.processed_tokens.clear()
            arms[name] = self.periodic(immediately_unmask=False, carry=True, trace_confidence=True)
        return {"native": native, "native_cost": native_cost, "arms": arms}

    def controlled_pairs(self, case):
        """Force only the contender list. Both models score a fully masked gap."""
        self.cost.clear()
        self.processed_tokens.clear()
        results = []
        for gap, foil in sorted(case.foils.items()):
            candidates = [self.verify({}, gap, text, "controlled_pair")
                          for text in (case.truth[gap].text, foil)]
            result = {"gap": gap, "native_widths": case.width_pair(gap), "candidates": candidates}
            if all(candidate is not None for candidate in candidates):
                winner, margins = ranking(candidates)
                scores = {name: candidates[0]["scores"][name] - candidates[1]["scores"][name] for name in self.members}
                choices = {name: ("correct" if margin > 0 else "wrong" if margin < 0 else "deferred")
                           for name, margin in scores.items()}
                joint_tie = candidates[0]["score"] == candidates[1]["score"]
                choices["joint"] = "deferred" if joint_tie else case.truth[gap].label(winner["text"])
                choices["unanimous"] = choices["joint"] if min(margins.values()) > 0 else "deferred"
                result.update(choices=choices, gold_minus_foil=scores,
                              joint_winner=None if joint_tie else winner["text"],
                              independently_ranked_differently=min(scores.values()) < 0 < max(scores.values()))
            else:
                result["status"] = "unscorable"
            results.append(result)
        return {"pairs": results, "cost": self.cost_summary()}


def manifest(case):
    fixture = case.fixture
    return {"name": fixture.name, "question": fixture.question, "visible": fixture.visible,
            "reference": fixture.gold, "widths": fixture.widths, "excluded": fixture.excluded,
            "label_types": ["literal" if item.literal else "integer" for item in case.truth],
            "controlled_foils": case.foils, "skipped_foils": case.skipped_foils}


def summarize_case(case, natural):
    groups = {}
    for name, output in natural["native"].items():
        groups[name] = (output, [row for row in natural["native_cost"] if row["model"] == name], None)
    for name, branch in natural["arms"].items():
        groups[name] = (branch["filled_gaps"], branch["cost"], branch["stop_reason"])
    summary = {}
    for name, (output, cost, stop) in groups.items():
        verdicts = Counter()
        widths = defaultdict(Counter)
        for gap in case.eligible():
            verdict = case.truth[gap].label(output.get(gap))
            verdicts[verdict] += 1
            widths[case.width_pair(gap)][verdict] += 1
        summary[name] = {"labels": dict(verdicts), "forwards": sum(row["forwards"] for row in cost),
                         "processed_tokens": sum(row["processed_tokens"] for row in cost), "stop_reason": stop,
                         "by_width": [{"native_widths": pair, "labels": dict(counts)}
                                      for pair, counts in sorted(widths.items())]}
    conflicts = []
    for index, round_ in enumerate(natural["arms"]["carry_joint"]["rounds"], start=1):
        for decision in round_["committed"]:
            if decision["scenario"] != "both_disagree":
                continue
            gap = decision["gap"]
            winner = decision["accepted"]
            margins = decision["unanimity_margins"]
            conflicts.append({"round": index, "gap": gap, "native_widths": case.width_pair(gap),
                              "comparable": margins is not None,
                              "joint_label": case.truth[gap].label(winner["text"] if winner else None),
                              "unanimity_veto": margins is not None and min(margins.values()) <= 0})
    return summary, conflicts


def summarize_study(cases):
    outcomes = defaultdict(Counter)
    natural_widths = defaultdict(lambda: defaultdict(Counter))
    controlled = defaultdict(Counter)
    conflicts = defaultdict(Counter)
    for case in cases:
        for arm, scores in case["natural_summary"].items():
            outcomes[arm].update(scores["labels"])
            for group in scores["by_width"]:
                natural_widths[arm][tuple(group["native_widths"])].update(group["labels"])
            outcomes[arm]["forwards"] += scores["forwards"]
            outcomes[arm]["processed_tokens"] += scores["processed_tokens"]
            stop = scores["stop_reason"]
            if stop is not None:
                outcomes[arm][stop] += 1
        for item in case["natural_conflicts_on_joint_contexts"]:
            group = conflicts[tuple(item["native_widths"])]
            group["observed"] += 1
            group["comparable"] += item["comparable"]
            if item["unanimity_veto"]:
                group[f"vetoed_{item['joint_label']}"] += 1
        for pair in case["controlled"]["pairs"]:
            group = controlled[tuple(pair["native_widths"])]
            group["pairs"] += 1
            if "choices" in pair:
                group["model_disagreed"] += pair["independently_ranked_differently"]
                for name, choice in pair["choices"].items():
                    group[f"{name}_{choice}"] += 1
            else:
                group["unscorable"] += 1
    return {"natural": dict(outcomes),
            "natural_by_width": {arm: [{"native_widths": widths, **counts}
                                       for widths, counts in sorted(groups.items())]
                                 for arm, groups in natural_widths.items()},
            "natural_conflicts_by_width": [{"native_widths": widths, **counts}
                                           for widths, counts in sorted(conflicts.items())],
            "controlled_by_width": [{"native_widths": widths, **counts}
                                    for widths, counts in sorted(controlled.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.preflight and not torch.cuda.is_available():
        parser.error("The live study requires the Slurm GPU allocation")
    configs = {name: ModelArguments(model_name_or_path=str(path), dtype="bfloat16") for name, path in CHECKPOINTS.items()}
    tokenizers = {name: get_tokenizer(config) for name, config in configs.items()}
    cases = prepare_cases({name: Vocabulary.from_tokenizer(tokenizer) for name, tokenizer in tokenizers.items()})
    report = {"study_id": STUDY, "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
              "width_order": list(CHECKPOINTS), "writes_per_model_round": INDEPENDENT_STEPS,
              "unanimity_margin_exclusive": 0, "fixtures": [manifest(case) for case in cases], "cases": []}
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    if torch.cuda.mem_get_info(0)[0] < 36 * 1024 ** 3:
        raise RuntimeError("The BF16 pilot needs at least 36 GiB of free GPU memory")
    members = {}
    for name, config in configs.items():
        print(f"Loading {name}", flush=True)
        model = get_model(config, device_map={"": "cuda:0"}).eval()
        members[name] = TIEMember(name, model, tokenizers[name], right_shift_logits=(name == "dream"))
    output = args.output or ROOT / ".logs" / STUDY / os.environ.get("SLURM_JOB_ID", "manual") / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for case in cases:
            pilot = DisagreementPilot(case.fixture, members, fixed_width=True)
            natural = pilot.natural_runs()
            summary, conflicts = summarize_case(case, natural)
            report["cases"].append({"name": case.fixture.name, "natural": natural, "natural_summary": summary,
                                    "natural_conflicts_on_joint_contexts": conflicts,
                                    "controlled": pilot.controlled_pairs(case)})
            report["summary"] = summarize_study(report["cases"])
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
            print(f"{case.fixture.name}: {summary}", flush=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
