"""Validate historical voting on a compute node after activating dllm.

Run: python -m pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_historical_diversity_vote.py -q
The diversity_vote.slurm.sh analysis job runs these checks before analysis.
"""

import pytest

from examples.path_selection.analysis.diversity_vote import check_identity, normalize_answer, vote


METRIC = {"ignore_case": True, "regexes_to_ignore": [",", r"\$", r"(?s).*#### ", r"\.$"]}


def test_normalization_matches_harness_without_numeric_equivalence():
    assert normalize_answer("$1,234.", METRIC) == "1234"
    assert normalize_answer("1234.0", METRIC) != normalize_answer("1234", METRIC)
    assert normalize_answer("[invalid]", METRIC) is None
    assert normalize_answer("reasoning\n#### $18.", METRIC) == "18"


def test_multiclass_vote_can_have_no_majority_with_seven_methods():
    result = vote(["a", "b", "a", "b", "c", "d", "e"])
    assert result["answer"] == "a"
    assert result["tie"] and not result["strict_majority"]


def test_invalid_extractions_abstain_without_lowering_majority_threshold():
    result = vote([None, None, None, None, "18", "18", "9"])
    assert result["answer"] == "18" and result["votes"] == 2
    assert not result["strict_majority"]
    assert vote([None] * 7)["answer"] is None
    assert vote(["18"] * 4 + ["9"] * 3)["strict_majority"]


def test_matching_ids_do_not_hide_prompt_mismatch():
    from examples.path_selection.analysis.diversity_vote import helpers
    original = {key: "same" for key in helpers.IDENTITY_FIELDS}
    changed = dict(original, prompt_hash="different")
    with pytest.raises(ValueError, match="prompt_hash"):
        check_identity(original, changed, "toy")


def test_voting_accuracy_and_oracle_coverage_are_distinct():
    from examples.path_selection.analysis.diversity_vote import METHODS, analyze_filter
    predictions = [
        ["18", "9", "18", "9", "18", "18", "9"],
        ["9", "18", "9", "18", "8", "8", "7"],
        ["[invalid]"] * 7,
    ]
    runs = {}
    for index, (name, label) in enumerate(METHODS):
        rows = {doc_id: {"filtered_resps": [values[index]], "target": "18",
                         "exact_match": int(values[index] == "18"), "resps": [[values[index]]]}
                for doc_id, values in enumerate(predictions)}
        runs[name] = {"label": label, "samples": {"flexible-extract": rows}}
    result, records = analyze_filter(runs, METRIC, "flexible-extract", repeats=20, seed=42)
    assert result["ensemble"]["correct"] == 1
    assert result["oracle_any_correct"] == 2
    assert result["ensemble"]["ties"] == 1
    assert result["ensemble"]["no_valid_votes"] == 1
    assert result["ensemble"]["strict_majority_coverage"] == 1
    assert result["correct_method_count_histogram"] == {0: 1, 2: 1, 4: 1}
    assert result["pairs"][0]["left_only_correct"] == 1
    assert result["pairs"][0]["right_only_correct"] == 1
    assert records[1]["vote"]["answer"] == "9"


def test_changed_matching_rule_cannot_silently_change_individual_accuracy():
    from examples.path_selection.analysis.diversity_vote import METHODS, analyze_filter
    row = {"filtered_resps": ["18.0"], "target": "18", "exact_match": 1, "resps": [["18.0"]]}
    runs = {name: {"label": label, "samples": {"flexible-extract": {0: row}}} for name, label in METHODS}
    with pytest.raises(ValueError, match="does not reproduce"):
        analyze_filter(runs, METRIC, "flexible-extract", repeats=20, seed=42)
