"""Test compact diagnostic retention used by the final Phase-8 evaluation.

Run from the repository root:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_p8_evaluation.py -v
"""

import pytest

from dllm.core.eval.diagnostic_retention import retain_diagnostics


def test_compact_diagnostics_remove_candidate_pool_but_keep_cost_fields():
    diagnostics = [
        [
            {
                "lookahead_model_calls": 4,
                "captured_base_forward_count": 1,
                "candidates": [{"name": "candidate_0", "positions": [2, 4]}],
                "selected_candidate": {
                    "name": "candidate_0",
                    "positions": [2, 4],
                    "valid": True,
                },
            }
        ]
    ]

    compact = retain_diagnostics(diagnostics, "compact")

    assert "candidates" not in compact[0][0]
    assert "positions" not in compact[0][0]["selected_candidate"]
    assert compact[0][0]["lookahead_model_calls"] == 4
    assert compact[0][0]["captured_base_forward_count"] == 1
    assert retain_diagnostics(diagnostics, "full") is diagnostics
    assert retain_diagnostics(diagnostics, "none") == [[]]


def test_unknown_diagnostic_retention_is_rejected():
    with pytest.raises(ValueError, match="diagnostic_retention"):
        retain_diagnostics([], "summary")
