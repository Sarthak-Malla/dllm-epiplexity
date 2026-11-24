"""On a cscc-cpu-p allocation, activate dllm and run:

    python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_reconciliation_confidence.py
"""

from types import SimpleNamespace

import torch

from scripts.tse.ctca.independent_reconciliation import (
    Fixture,
    Pilot,
    compare_on_ungated_contexts,
    select_proposal,
)
from scripts.tse.ctca.reconciliation_disagreements import Truth, ranking


def candidate(text, score, paths, geomeans):
    return {"text": text, "score": score, "path_probability": paths, "token_geomean": geomeans}


def test_confidence_gate_requires_both_models_and_keeps_ranking():
    overconfident = candidate("x", -0.01, {"llada": 0.99, "dream": 0.8}, {"llada": 0.99, "dream": 0.8})
    supported = candidate("y", -0.1, {"llada": 0.82, "dream": 0.83}, {"llada": 0.82, "dream": 0.83})
    assert select_proposal([overconfident, supported]) == overconfident
    assert select_proposal([overconfident, supported], "path_probability") == supported
    assert select_proposal([overconfident], "path_probability") is None


def test_same_context_report_counts_correct_deferred_split_gap():
    fixture = Fixture("test", "", ["", ""], ["abc"], {"llada": {0: 2}, "dream": {0: 3}}, {})
    split = candidate("abc", -0.21, {"llada": 0.69, "dream": 0.52}, {"llada": 0.83, "dream": 0.81})
    decisions = [{"committed": [{"gap": 0, "scenario": "one_complete_peer_masked",
                                 "scored": [split], "accepted": split}]}]
    result = compare_on_ungated_contexts(fixture, decisions, 0.8)
    path, = result["path_probability"]
    assert path["correct_ungated_action_deferred"] == path["abstained"] == 1
    assert (path["llada_width"], path["dream_width"]) == (2, 3)
    assert result["token_geomean"][0]["unchanged"] == 1


class ToyMember:
    device = torch.device("cpu")

    def __init__(self):
        self.vocabulary = SimpleNamespace(
            mask_id=0, special_ids={0, 1}, terminal_ids={1},
            encode=lambda text: [ord(char) - ord("a") + 4 for char in text],
            decode=lambda ids: "".join(chr(token - 4 + ord("a")) for token in ids),
        )
        self.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [3],
            convert_ids_to_tokens=lambda ids: self.vocabulary.decode(ids),
        )

    def fits_context(self, length):
        return True

    def log_probabilities(self, canvas, positions):
        logits = torch.full((len(positions), 8), -10.0)
        logits[torch.arange(len(positions)), positions + 3] = 10
        return logits.log_softmax(dim=1)


def test_carry_completes_three_slot_gap_after_two_write_round():
    fixture = Fixture("test", "", ["", ""], ["abc"], {"llada": {0: 3}, "dream": {0: 3}}, {})
    members = {name: ToyMember() for name in fixture.widths}
    reset = Pilot(fixture, members, fixed_width=True).periodic(immediately_unmask=False)
    carry = Pilot(fixture, members, fixed_width=True).periodic(
        immediately_unmask=False, carry=True, trace_confidence=True
    )
    assert reset["stop_reason"] == "no_progress" and not reset["filled_gaps"]
    assert carry["stop_reason"] == "all_filled" and carry["filled_gaps"] == {0: "abc"}
    first, second = carry["rounds"]
    assert first["carried_out"] == second["carried_in"]
    assert first["carried_out"]["llada"] == {0: [4, 5, 0]}
    assert len(second["original_writes"]["llada"]) == 1
    write = second["original_writes"]["llada"][0]
    assert (write["gap"], write["offset"], write["chosen_id"]) == (0, 2, 6)
    assert write["probability"] > 0.99


def test_fixed_width_rejects_a_candidate_that_requires_resizing():
    fixture = Fixture("test", "", ["", ""], ["ab"], {"llada": {0: 2}, "dream": {0: 2}}, {})
    pilot = Pilot(fixture, {name: ToyMember() for name in fixture.widths}, fixed_width=True)
    assert pilot.verify({}, 0, "abc", "committed") is None
    assert not pilot.cost


def test_disagreement_veto_uses_each_models_preference():
    gold = {"text": " 29", "score": -0.3, "scores": {"llada": -0.1, "dream": -0.5}}
    foil = {"text": " 30", "score": -0.5, "scores": {"llada": -0.9, "dream": -0.1}}
    winner, margins = ranking([gold, foil])
    assert winner == gold
    assert margins["llada"] > 0 > margins["dream"]
    assert ranking([gold]) == (gold, None)


def test_numeric_label_distinguishes_factual_errors_from_unparsed_text():
    truth = Truth(" 29", (" 30",))
    assert truth.label("29") == "correct"
    assert truth.label(" 30") == "wrong"
    assert truth.label(" twenty-nine") == "unlabeled"
    assert truth.label(None) == "unfilled"
    assert Truth(".\n", ("!\n",), literal=True).label("!\n") == "wrong"
