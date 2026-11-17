"""Run with: pytest scripts/tests/test_span_audit.py -q."""

from scripts.tse.ctca.span_audit import (
    high_confidence_projection_mismatch,
    match_spans,
)


def forward(*tokens):
    return {"master": {"distribution": {"top_k": [
        {"token": token, "probability": probability}
        for token, probability in tokens
    ]}}}


def test_split_text_matches_only_the_complete_span():
    master = {69: forward((" cent", 0.06), (" kilometers", 0.1)),
              70: forward(("imeters", 0.02), (" every", 0.7))}
    matches, blocked = match_spans(" centimeters", 69, master, {})
    assert matches == [{"positions": [69, 70], "pieces": [" cent", "imeters"],
                        "master_product": 0.0012}]
    assert blocked == []


def test_committed_slot_blocks_split_text():
    master = {69: forward((" cent", 0.06))}
    matches, blocked = match_spans(
        " centimeters", 69, master,
        {70: {"predicted_token": " every", "step": [0, 0]}},
    )
    assert matches == []
    assert blocked == [{"position": 70, "committed_token": " every",
                        "committed_step": [0, 0], "prefix": " cent"}]


def test_prefix_does_not_become_a_longer_word_without_continuation():
    master = {0: forward((" trajectory", 0.4), (" traj", 0.3))}
    assert match_spans(" traj", 0, master, {})[0][0]["pieces"] == [" traj"]
    matches, blocked = match_spans(" trajectory", 0, {0: forward((" traj", 0.3))}, {})
    assert matches == []
    assert blocked == []


def test_candidate_absent_from_saved_top_k_is_unscored():
    matches, blocked = match_spans(" centimeters", 0,
                                   {0: forward((" kilometers", 0.9))}, {})
    assert matches == []
    assert blocked == []


def test_projection_mismatch_excludes_specials_and_weak_source_predictions():
    row = {"selected": True, "source_top_probability": 0.99,
           "source_top": " centimeters", "projected_auxiliary_top": " kilometers"}
    assert high_confidence_projection_mismatch(row)
    assert not high_confidence_projection_mismatch({**row, "source_top": "<|endoftext|>"})
    assert not high_confidence_projection_mismatch({**row, "source_top_probability": 0.2})
    assert not high_confidence_projection_mismatch({**row, "projected_auxiliary_top": " centimeters"})
