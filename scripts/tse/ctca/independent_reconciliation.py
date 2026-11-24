"""Run the LLaDA–Dream reconciliation pilot inside a Slurm GPU allocation.

    sbatch /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/ctca_independent_reconciliation_slurm.sh

Pass --confidence-study through the Slurm wrapper to run the matched ablation.
For tokenizer checks, activate dllm on a cscc-cpu-p allocation and run
this file with --preflight. The full pilot expects a CUDA device.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import chain
import json
import math
import os
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dllm.pipelines.tie.commit_actions import contextual_ids
from dllm.pipelines.tie.relay import RelayError, Vocabulary
from dllm.pipelines.tie.sampler import TIEMember
from dllm.utils.configs import ModelArguments
from dllm.utils.models import get_model, get_tokenizer


STUDY = "ctca-llada-dream-independent-reconciliation-pilot-20261123"
CONFIDENCE_STUDY = "ctca-llada-dream-reconciliation-confidence-pilot-20261123"
SNAPSHOTS = Path("/home/juan.guevara/.cache/huggingface/hub")
CHECKPOINTS = {
    "llada": SNAPSHOTS / "models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07",
    "dream": SNAPSHOTS / "models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
}
FIXTURES = {
    "candle": (
        "A candle is burning from noon to 4 PM and melts at 2 cm per hour. Explain the amount lost.",
        "The{ candle} burns from{ noon} until{ four} in the afternoon. It loses two{ centimeters every} hour."
        " Over{ four} hours it becomes{ eight} centimeters{ shorter} overall.",
    ),
    "boxes": (
        "There are three boxes with 12 pencils each. Five pencils are used. Explain how many are left.",
        "There are{ three} boxes with{ 12} pencils per box. Multiplying gives{ 36} pencils altogether."
        " After{ five} pencils are used,{ 31} pencils{ remain} unused.",
    ),
    "distance": (
        "A walk is 3 kilometers each way. Explain the total distance in meters.",
        "It takes{ three} kilometers to walk there and{ three} kilometers to return."
        " The distance is{ six} kilometers in{ total}. That is{ six thousand} meters{ altogether}.",
    ),
}
INDEPENDENT_STEPS = 2
TOP_K = 32
MAX_PATH_WIDTH = 3
MAX_ROUNDS = 8


def select_proposal(proposals, gate=None, threshold=0.8):
    """Retain the original ranking among proposals that both models support."""
    if gate is not None:
        proposals = [item for item in proposals if min(item[gate].values()) > threshold]
    return max(proposals, key=lambda item: item["score"], default=None)


def draft_scenario(drafts):
    complete = [draft["text"] for draft in drafts.values() if draft["text"] is not None]
    partial = any(0 < draft["filled"] < len(draft["ids"]) for draft in drafts.values())
    if len(complete) == 2:
        return "both_agree" if complete[0] == complete[1] else "both_disagree"
    if complete:
        return "one_complete_peer_partial" if partial else "one_complete_peer_masked"
    if partial:
        return "partial_only"
    return "no_usable_text" if any(draft["filled"] for draft in drafts.values()) else "both_masked"


@dataclass(frozen=True)
class Fixture:
    name: str
    question: str
    visible: list[str]
    gold: list[str]
    widths: dict[str, dict[int, int]]
    excluded: dict[int, str]

    def candidates(self, vocabularies, gap, text):
        """Find each model's native encoding for the same bounded surface span."""
        if not text or not text.strip() or "\ufffd" in text:
            raise RelayError("empty or incomplete candidate text")
        paths = {name: contextual_ids(vocab, self.visible[gap], text, self.visible[gap + 1])
                 for name, vocab in vocabularies.items()}
        if any(len(ids) > MAX_PATH_WIDTH for ids in paths.values()):
            raise RelayError("candidate exceeds the native path width budget")
        return paths


def prepare(vocabularies):
    """Define checkpoint layouts from reference widths, before any model forward."""
    prepared = []
    for name, (question, marked) in FIXTURES.items():
        parts = re.split(r"\{([^{}]+)\}", marked)
        fixture = Fixture(name, question, parts[::2], parts[1::2],
                          {name: {} for name in vocabularies}, {})
        for gap, gold in enumerate(fixture.gold):
            try:
                paths = fixture.candidates(vocabularies, gap, gold)
            except RelayError as error:
                fixture.excluded[gap] = str(error)
                continue
            for model, ids in paths.items():
                fixture.widths[model][gap] = len(ids)
        prepared.append(fixture)
    if not any(len({widths[gap] for widths in case.widths.values()}) > 1
               for case in prepared for gap in next(iter(case.widths.values()))):
        raise ValueError("The tokenizer preflight must find at least one split-width gap")
    if any(not next(iter(case.widths.values())) for case in prepared):
        raise ValueError("Each fixture must contain eligible gaps")
    return prepared


class Pilot:
    """Reconcile proposals on a shared text canvas with stable gap IDs."""

    def __init__(self, fixture, members, *, fixed_width=False, confidence_threshold=0.8):
        self.fixture = fixture
        self.members = members
        self.vocabularies = {name: member.vocabulary for name, member in members.items()}
        self.prompts = {
            name: member.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{fixture.question}\nComplete the reasoning passage."}],
                tokenize=True, add_generation_prompt=True,
            ) for name, member in members.items()
        }
        self.cost = Counter()
        self.processed_tokens = Counter()
        self.fixed_width = fixed_width
        self.confidence_threshold = confidence_threshold

    def render(self, name, solved, override=None):
        """Render native context and record the current indices of each gap ID."""
        vocab = self.vocabularies[name]
        ids = list(self.prompts[name])
        spans = {}
        visible = self.fixture.visible[0]
        for gap, right in enumerate(self.fixture.visible[1:]):
            if gap in self.fixture.excluded:
                visible += self.fixture.gold[gap] + right
            elif gap in solved:
                visible += solved[gap] + right
            else:
                ids.extend(self.visible_ids(vocab, visible))
                width = (override or {}).get(gap, self.fixture.widths[name][gap])
                spans[gap] = tuple(range(len(ids), len(ids) + width))
                ids.extend([vocab.mask_id] * width)
                visible = right
        ids.extend(self.visible_ids(vocab, visible))
        member = self.members[name]
        if not member.fits_context(len(ids)):
            raise ValueError(f"{name} canvas exceeds context length")
        return torch.tensor(ids, device=member.device, dtype=torch.long), spans

    @staticmethod
    def visible_ids(vocab, text):
        ids = vocab.encode(text)
        if vocab.decode(ids) != text or any(token in vocab.special_ids for token in ids):
            raise RelayError("visible run failed exact native round trip")
        return ids

    def logp(self, name, canvas, positions, phase):
        self.cost[phase, name] += 1
        self.processed_tokens[phase, name] += len(canvas)
        rows = self.members[name].log_probabilities(canvas, canvas.new_tensor(positions))
        terminal = [i for i in self.vocabularies[name].terminal_ids if i < rows.shape[-1]]
        rows[:, terminal] = -torch.inf
        return rows

    def denoise(self, name, canvas, spans, steps, phase, *, writes=None, carried=None):
        """Update each native canvas in place, then read complete gap proposals."""
        canvas = canvas.clone()
        vocab = self.vocabularies[name]
        locations = {position: (gap, offset) for gap, positions in spans.items()
                     for offset, position in enumerate(positions)}
        for gap, ids in (carried or {}).items():
            positions = spans[gap]
            if len(positions) != len(ids):
                raise ValueError("A carried draft must fit its original native gap width")
            canvas[list(positions)] = canvas.new_tensor(ids)
        if carried:
            pending = [position for position, token in zip(locations, canvas[list(locations)].tolist())
                       if token == vocab.mask_id]
        else:
            pending = list(locations)
        for step in range(1, min(steps, len(pending)) + 1):
            rows = self.logp(name, canvas, pending, phase)
            confidence, tokens = rows.max(dim=1)
            row = int(confidence.argmax())
            log_probability = float(confidence[row])
            if not math.isfinite(log_probability):
                break
            position = pending.pop(row)
            if writes is not None:
                values, ids = rows[row].topk(min(TOP_K, rows.shape[1]))
                gap, offset = locations[position]
                writes.append({"step": step, "gap": gap, "offset": offset, "position": position,
                               "chosen_id": int(tokens[row]), "log_probability": log_probability,
                               "probability": math.exp(log_probability),
                               "top_k": list(zip(ids.tolist(), values.exp().tolist()))})
            canvas[position] = tokens[row]
        drafts = {gap: canvas[list(positions)].tolist() for gap, positions in spans.items()}
        return {
            gap: {"ids": ids, "filled": sum(i != vocab.mask_id for i in ids),
                  "text": vocab.decode(ids) if all(i not in vocab.special_ids for i in ids) else None}
            for gap, ids in drafts.items()
        }

    def path_score(self, name, solved, gap, ids, phase):
        canvas, spans = self.render(name, solved, {gap: len(ids)})
        logp = 0.0
        for position, token in zip(spans[gap], ids):
            row = self.logp(name, canvas, [position], phase)[0]
            if token >= row.numel():
                return -math.inf
            logp += float(row[token])
            canvas[position] = token
        return logp / len(ids)

    def verify(self, solved, gap, text, phase):
        try:
            paths = self.fixture.candidates(self.vocabularies, gap, text)
        except RelayError:
            return None
        if self.fixed_width and any(len(ids) != self.fixture.widths[name][gap]
                                    for name, ids in paths.items()):
            return None
        scores = {name: self.path_score(name, solved, gap, ids, phase) for name, ids in paths.items()}
        return self.proposal(text, paths, scores) if all(map(math.isfinite, scores.values())) else None

    def proposal(self, text, paths, scores):
        tokens = {name: self.members[name].tokenizer.convert_ids_to_tokens(ids)
                  for name, ids in paths.items()}
        return {"text": text, "paths": paths, "native_tokens": tokens, "scores": scores,
                "score": sum(scores.values()) / len(scores),
                "path_probability": {name: math.exp(score * len(paths[name])) for name, score in scores.items()},
                "token_geomean": {name: math.exp(score) for name, score in scores.items()}}

    def reconcile_commits(self, solved, drafts, gate=None):
        decisions = []
        for gap in next(iter(drafts.values())):
            proposals = {name: draft[gap] for name, draft in drafts.items()}
            texts = sorted({draft["text"] for draft in proposals.values() if draft["text"]})
            scored = [item for text in texts if (item := self.verify(solved, gap, text, "committed"))]
            winner = select_proposal(scored, gate, self.confidence_threshold)
            decisions.append({"gap": gap, "scenario": draft_scenario(proposals), "independent": proposals,
                              "scored": scored, "accepted": winner})
        return decisions

    def unmask_candidates(self, solved):
        """Score exact text candidates for an immediate shared fill."""
        predictions = {}
        for name in self.members:
            canvas, spans = self.render(name, solved)
            if not spans:
                return []
            rows = self.logp(name, canvas, list(chain.from_iterable(spans.values())), "unmask")
            predictions[name] = dict(zip(spans, rows.split([len(span) for span in spans.values()])))
        proposals = []
        for gap in next(iter(predictions.values())):
            rows = {name: prediction[gap] for name, prediction in predictions.items()}
            if all(len(row) == 1 for row in rows.values()):
                options = {}
                for name, row in rows.items():
                    vocab = self.vocabularies[name]
                    values, ids = row[0].topk(min(TOP_K, row.shape[1]))
                    options[name] = {
                        token: value for token, value in zip(ids.tolist(), values.tolist())
                        if math.isfinite(value) and token not in vocab.special_ids
                    }
                first = next(iter(rows))
                for text in sorted({self.vocabularies[first].decode([token]) for token in options[first]}):
                    try:
                        paths = self.fixture.candidates(self.vocabularies, gap, text)
                    except RelayError:
                        continue
                    if all(len(ids) == 1 and ids[0] in options[name] for name, ids in paths.items()):
                        scores = {name: options[name][ids[0]] for name, ids in paths.items()}
                        proposals.append({"gap": gap, **self.proposal(text, paths, scores)})
            else:
                texts = sorted({self.vocabularies[name].decode(row.argmax(1).tolist()) for name, row in rows.items()})
                for text in texts:
                    if item := self.verify(solved, gap, text, "unmask_split"):
                        proposals.append({"gap": gap, **item})
        return proposals

    def cost_summary(self):
        return [{"phase": phase, "model": name, "forwards": count,
                 "processed_tokens": self.processed_tokens[phase, name]}
                for (phase, name), count in sorted(self.cost.items())]

    def native_baselines(self):
        baselines = {}
        for name in self.members:
            canvas, spans = self.render(name, {})
            draft = self.denoise(name, canvas, spans, len(canvas), "native_baseline")
            baselines[name] = {gap: item["text"] for gap, item in draft.items()}
        return baselines

    def periodic(self, *, immediately_unmask, gate=None, carry=False, trace_confidence=False):
        solved = {}
        carried = {name: {} for name in self.members}
        rounds = []
        stop_reason = "round_limit"
        for _ in range(MAX_ROUNDS):
            views = {name: self.render(name, solved) for name in self.members}
            layouts = {name: spans for name, (_, spans) in views.items()}
            if not any(layouts.values()):
                stop_reason = "all_filled"
                break
            writes = {name: [] for name in self.members} if trace_confidence else {}
            drafts = {name: self.denoise(name, *view, INDEPENDENT_STEPS, "independent",
                                         writes=writes.get(name), carried=carried[name])
                      for name, view in views.items()}
            committed = self.reconcile_commits(solved, drafts, gate)
            updated = {**solved, **{item["gap"]: item["accepted"]["text"]
                                    for item in committed if item["accepted"]}}
            candidates = self.unmask_candidates(updated) if immediately_unmask else []
            unmasked = select_proposal(candidates, gate, self.confidence_threshold)
            if unmasked is not None:
                updated[unmasked["gap"]] = unmasked["text"]
            after = {name: self.render(name, updated)[1] for name in self.members}
            next_carried = {
                name: {gap: draft["ids"] for gap, draft in native.items()
                       if carry and gap not in updated and 0 < draft["filled"] < len(draft["ids"])}
                for name, native in drafts.items()
            }
            record = {"start_native_mask_layout": layouts, "committed": committed,
                      "immediate_unmask": unmasked, "end_native_mask_layout": after}
            if trace_confidence:
                record.update(shared_before=solved, original_writes=writes, carried_in=carried,
                              carried_out=next_carried, unmask_candidates=candidates)
            rounds.append(record)
            progress = updated != solved or next_carried != carried
            solved, carried = updated, next_carried
            if not any(after.values()):
                stop_reason = "all_filled"
                break
            if not progress:
                stop_reason = "no_progress"
                break
        return {"filled_gaps": solved, "rounds": rounds, "stop_reason": stop_reason,
                "cost": self.cost_summary()}

    def run(self, *, confidence_study=False):
        native = self.native_baselines()
        result = {"native": native, "native_cost": self.cost_summary(), "periodic": {}}
        arms = {
            "commits_only": {"immediately_unmask": False},
            "commits_and_unmask": {"immediately_unmask": True},
        }
        if confidence_study:
            arms.update({
                "path_gate": {"immediately_unmask": True, "gate": "path_probability"},
                "token_geomean_gate": {"immediately_unmask": True, "gate": "token_geomean"},
                "commits_only_carry": {"immediately_unmask": False, "carry": True},
            })
        for arm, options in arms.items():
            self.cost.clear()
            self.processed_tokens.clear()
            result["periodic"][arm] = self.periodic(trace_confidence=confidence_study, **options)
        return result


def summarize(fixture, results):
    widths = next(iter(fixture.widths.values()))
    counts = {}
    outputs = {**results["native"], **{name: arm["filled_gaps"] for name, arm in results["periodic"].items()}}
    for name, guesses in outputs.items():
        counts[name] = {
            label: {"correct": sum(guesses.get(gap) == fixture.gold[gap] for gap in gaps),
                    "filled": sum(guesses.get(gap) is not None for gap in gaps), "total": len(gaps)}
            for label, gaps in {
                "all": list(widths),
                "split_width": [gap for gap in widths if len({w[gap] for w in fixture.widths.values()}) > 1],
            }.items()
        }
    return counts


def summarize_actions(fixture, rounds):
    counts = Counter()
    for record in rounds:
        for decision in record["committed"]:
            drafts = list(decision["independent"].values())
            texts = [draft["text"] for draft in drafts if draft["text"]]
            counts["independent_conflicts"] += len(set(texts)) > 1
            counts["partially_filled_native_gaps"] += sum(
                0 < draft["filled"] < len(draft["ids"]) for draft in drafts
            )
            if proposal := decision["accepted"]:
                counts["accepted_commits"] += 1
                counts["correct_commits"] += proposal["text"] == fixture.gold[decision["gap"]]
                counts["split_width_commits"] += len({len(path) for path in proposal["paths"].values()}) > 1
        if proposal := record["immediate_unmask"]:
            counts["immediately_unmasked"] += 1
            counts["correct_immediate_unmasks"] += proposal["text"] == fixture.gold[proposal["gap"]]
            counts["split_width_immediate_unmasks"] += len({len(path) for path in proposal["paths"].values()}) > 1
    return dict(counts)


def decision_points(rounds):
    for record in rounds:
        for decision in record["committed"]:
            yield ("committed", decision["scenario"], decision["gap"], decision["scored"], decision["accepted"])
        if candidates := record.get("unmask_candidates"):
            yield ("immediate", "both_masked", None, candidates, record["immediate_unmask"])


def scored_group(fixture, phase, scenario, gap):
    return phase, scenario, fixture.widths["llada"][gap], fixture.widths["dream"][gap]


def group_rows(groups):
    return [{"phase": phase, "scenario": scenario, "llada_width": llada, "dream_width": dream, **counts}
            for (phase, scenario, llada, dream), counts in sorted(groups.items())]


def live_decisions(fixture, rounds):
    groups = {}
    for phase, scenario, gap, candidates, accepted in decision_points(rounds):
        reference = select_proposal(candidates)
        if phase == "immediate":
            gap = (accepted or reference)["gap"]
        counts = groups.setdefault(scored_group(fixture, phase, scenario, gap), Counter())
        counts["opportunities"] += 1
        if accepted is not None:
            counts["correct"] += accepted["text"] == fixture.gold[gap]
            counts["wrong"] += accepted["text"] != fixture.gold[gap]
        elif reference is not None:
            counts["gate_deferrals"] += 1
        else:
            counts["no_valid_candidate"] += 1
    return group_rows(groups)


def compare_on_ungated_contexts(fixture, rounds, threshold):
    """Measure gate decisions on identical contexts before trajectories diverge."""
    report = {}
    for gate in ("path_probability", "token_geomean"):
        groups = {}
        for phase, scenario, gap, candidates, reference in decision_points(rounds):
            if reference is None:
                continue
            if phase == "immediate":
                gap = reference["gap"]
            counts = groups.setdefault(scored_group(fixture, phase, scenario, gap), Counter())
            counts["ungated_actions"] += 1
            selected = select_proposal(candidates, gate, threshold)
            if selected == reference:
                counts["unchanged"] += 1
                continue
            was_correct = reference["text"] == fixture.gold[gap]
            counts["correct_ungated_action_deferred" if was_correct else "wrong_ungated_action_prevented"] += 1
            if selected is None:
                counts["abstained"] += 1
            else:
                replacement_gap = selected.get("gap", gap)
                label = "correct_replacement" if selected["text"] == fixture.gold[replacement_gap] else "wrong_replacement"
                counts[label] += 1
        report[gate] = group_rows(groups)
    return report


def summarize_study(cases):
    summary = {}
    for arm in cases[0]["accuracy"]:
        score = Counter()
        stops = Counter()
        forwards = 0
        for case in cases:
            score.update(case["accuracy"][arm]["all"])
            if arm in case["periodic"]:
                branch = case["periodic"][arm]
                stops[branch["stop_reason"]] += 1
                forwards += sum(row["forwards"] for row in branch["cost"])
            else:
                forwards += sum(row["forwards"] for row in case["native_cost"] if row["model"] == arm)
        summary[arm] = {**score, "stop_reasons": dict(stops), "forwards": forwards}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confidence-study", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be greater than zero and at most one")
    if not args.preflight and not torch.cuda.is_available():
        parser.error("The live pilot requires a Slurm GPU allocation")
    model_args = {name: ModelArguments(model_name_or_path=str(path), dtype="bfloat16")
                  for name, path in CHECKPOINTS.items()}
    tokenizers = {name: get_tokenizer(config) for name, config in model_args.items()}
    vocabularies = {name: Vocabulary.from_tokenizer(tokenizer) for name, tokenizer in tokenizers.items()}
    fixtures = prepare(vocabularies)
    study = CONFIDENCE_STUDY if args.confidence_study else STUDY
    report = {"study_id": study, "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
              "independent_steps": INDEPENDENT_STEPS, "top_k": TOP_K, "cases": []}
    if args.confidence_study:
        report.update(fixed_native_gap_widths=True, confidence_threshold_exclusive=args.confidence_threshold)
    for fixture in fixtures:
        reference_paths = {
            gap: {
                name: {"ids": ids, "tokens": tokenizers[name].convert_ids_to_tokens(ids)}
                for name, ids in fixture.candidates(vocabularies, gap, text).items()
            }
            for gap, text in enumerate(fixture.gold) if gap not in fixture.excluded
        }
        report["cases"].append({"name": fixture.name, "question": fixture.question, "gold": fixture.gold,
                                "visible": fixture.visible, "widths": fixture.widths, "excluded": fixture.excluded,
                                "reference_native_paths": reference_paths})
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    if torch.cuda.mem_get_info(0)[0] < 36 * 1024 ** 3:
        raise RuntimeError("The sequential BF16 pilot needs at least 36 GiB of free GPU memory")
    members = {}
    for name, config in model_args.items():
        print(f"Loading {name}", flush=True)
        model = get_model(config, device_map={"": "cuda:0"}).eval()
        members[name] = TIEMember(name, model, tokenizers[name], right_shift_logits=(name == "dream"))
    output = args.output or ROOT / ".logs" / study / os.environ.get("SLURM_JOB_ID", "manual") / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, (fixture, case) in enumerate(zip(fixtures, report["cases"]), start=1):
            pilot = Pilot(fixture, members, fixed_width=args.confidence_study,
                          confidence_threshold=args.confidence_threshold)
            results = pilot.run(confidence_study=args.confidence_study)
            actions = {name: summarize_actions(fixture, arm["rounds"])
                       for name, arm in results["periodic"].items()}
            case.update(results, accuracy=summarize(fixture, results), actions=actions)
            if args.confidence_study:
                case["live_decisions"] = {name: live_decisions(fixture, arm["rounds"])
                                          for name, arm in results["periodic"].items()}
                ungated = results["periodic"]["commits_and_unmask"]["rounds"]
                case["same_context_gate_comparison"] = compare_on_ungated_contexts(
                    fixture, ungated, args.confidence_threshold
                )
                report["summary"] = summarize_study(report["cases"][:index])
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
            print(f"{fixture.name}: {case['accuracy']}", flush=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
