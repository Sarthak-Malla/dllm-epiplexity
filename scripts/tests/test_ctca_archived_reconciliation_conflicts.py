"""On a cscc-cpu-p allocation, activate dllm from the repository root and run:

    python -m pytest -q /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_ctca_archived_reconciliation_conflicts.py
"""

import pytest

from scripts.tse.ctca.archived_reconciliation_conflicts import choose_from_margins, summarize, verify_replay


def test_archive_rules_distinguish_veto_from_owner_fallback():
    assert choose_from_margins(-1, 2) == ("peer", "defer")
    assert choose_from_margins(-1, -0.3) == ("owner", "owner")
    assert choose_from_margins(0, 0) == ("owner", "defer")
    row = {"doc_id": 1, "correct": {"owner": True, "peer": False}, "joint_choice": "peer",
           "unanimous_choice": "defer", "pair_scoring_forwards": {"llada": 2, "dream": 4},
           "additional_verification_after_peer_proposal": 4}
    counts = summarize([row])
    assert counts["unanimous_correct"] == counts["unanimous_wrong"] == 0
    assert counts["wrong_joint_choices_deferred"] == counts["deferred_wrong_with_correct_alternative"] == 1
    assert counts["unanimous_or_keep_owner_correct"] == 1
    assert counts["pair_scoring_forwards"] == 6


def test_verify_replay_allows_only_the_paired_gap_to_change():
    baseline = {"member": "llada", "reasoning_ids": [7, 8, 0, 10, 11], "correct": False, "forward_calls": 8}
    check = {"gap": [2], "owner_ids_a": [10], "owner_ids_b": [20], "owner_margin": -1, "peer_margin": 2}
    branch = {"slots": [3, 2], "ids": [11, 20], "owner_score": -1, "peer_score": 2,
              "completion": {**baseline, "correct": True, "reasoning_ids": [7, 8, 0, 20, 11]}}
    event = {"default_slots": [3, 2], "default_ids": [11, 10], "draft": [[7], [8, 0, 0, 0]],
             "block": 1, "step": 4, "alternatives": {"peer_text": branch}}
    assert verify_replay(event, check, baseline, "llada") == (True, 4)
    branch["ids"] = [99, 20]
    with pytest.raises(ValueError, match="annotated gap"):
        verify_replay(event, check, baseline, "llada")
