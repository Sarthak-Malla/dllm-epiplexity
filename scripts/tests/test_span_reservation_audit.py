"""Run: pytest scripts/tests/test_span_reservation_audit.py -q."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tse" / "ctca"))
from span_reservation_audit import select_with_protection


def decision(position, score, budget=2):
    return {"absolute_position": position, "selection_score": score,
            "scheduled_transfer_count": budget, "predicted_token": str(position)}


def test_protection_replaces_conflicting_commit_and_preserves_budget():
    decisions = [decision(10, .9), decision(11, .8), decision(12, .7), decision(13, .6)]
    proposals = [{"positions": [11, 12], "source_probability": .95}]
    selected, protected, accepted, rejected = select_with_protection(decisions, proposals)
    assert {row["absolute_position"] for row in selected} == {10, 13}
    assert protected == {11, 12}
    assert len(accepted) == 1 and not rejected


def test_protection_rejects_span_when_budget_cannot_be_filled():
    decisions = [decision(10, .9), decision(11, .8), decision(12, .7)]
    proposals = [{"positions": [11, 12], "source_probability": .95}]
    selected, protected, accepted, rejected = select_with_protection(decisions, proposals)
    assert {row["absolute_position"] for row in selected} == {10, 11}
    assert not protected and not accepted and len(rejected) == 1


def test_overlapping_proposals_share_protected_slots():
    decisions = [decision(10, .9), decision(11, .8), decision(12, .7),
                 decision(13, .6), decision(14, .5)]
    proposals = [{"positions": [11, 12], "source_probability": .9},
                 {"positions": [12, 13], "source_probability": .8}]
    selected, protected, accepted, rejected = select_with_protection(decisions, proposals)
    assert protected == {11, 12, 13}
    assert {row["absolute_position"] for row in selected} == {10, 14}
    assert len(accepted) == 2 and not rejected
