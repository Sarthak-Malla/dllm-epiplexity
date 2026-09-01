"""
Test dependency sink detection, filtering, and safe row normalization.

Run with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_sinks.py -v
"""

import torch

from dllm.core.samplers.dependency import (
    DependencyCaptureOutput,
    detect_dependency_sinks,
    filter_dependency_sinks,
)


def _dependency_output(
    directed: torch.Tensor,
    *,
    query_valid_mask: torch.Tensor | None = None,
    key_valid_mask: torch.Tensor | None = None,
) -> DependencyCaptureOutput:
    batch_size, query_count, key_count = directed.shape
    if query_valid_mask is None:
        query_valid_mask = torch.ones(
            (batch_size, query_count),
            dtype=torch.bool,
        )
    if key_valid_mask is None:
        key_valid_mask = torch.ones(
            (batch_size, key_count),
            dtype=torch.bool,
        )
    query_positions = torch.arange(query_count).expand(batch_size, -1).clone()
    key_positions = torch.arange(key_count).expand(batch_size, -1).clone()
    query_positions.masked_fill_(~query_valid_mask, -1)
    key_positions.masked_fill_(~key_valid_mask, -1)
    return DependencyCaptureOutput(
        directed=directed,
        query_positions=query_positions,
        key_positions=key_positions,
        query_valid_mask=query_valid_mask,
        key_valid_mask=key_valid_mask,
        layer_ids=(2, 3),
        renormalized_selected_keys=True,
        diagonal_zeroed=True,
    )


def test_dominant_incoming_column_is_detected_zeroed_and_renormalized():
    output = _dependency_output(
        torch.tensor(
            [
                [
                    [0.0, 0.8, 0.2],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.7, 0.1],
                ]
            ]
        )
    )

    filtered = filter_dependency_sinks(output, sink_quantile=0.75)

    assert torch.equal(filtered.sink_mask, torch.tensor([[False, True, False]]))
    assert torch.count_nonzero(filtered.directed[:, :, 1]) == 0
    assert torch.allclose(
        filtered.directed.sum(dim=-1),
        torch.ones((1, 3)),
    )
    assert torch.equal(output.directed[0, :, 1], torch.tensor([0.8, 0.8, 0.7]))


def test_explicit_threshold_ignores_padded_keys_and_invalid_queries():
    output = _dependency_output(
        torch.tensor(
            [
                [
                    [0.1, 0.9, 100.0],
                    [0.2, 0.8, 100.0],
                    [100.0, 100.0, 100.0],
                ]
            ]
        ),
        query_valid_mask=torch.tensor([[True, True, False]]),
        key_valid_mask=torch.tensor([[True, True, False]]),
    )

    sink_mask = detect_dependency_sinks(output, sink_threshold=0.75)
    filtered = filter_dependency_sinks(output, sink_threshold=0.75)

    assert torch.equal(sink_mask, torch.tensor([[False, True, False]]))
    assert torch.equal(filtered.sink_mask, sink_mask)
    assert torch.count_nonzero(filtered.directed[:, :, 1:]) == 0
    assert torch.equal(filtered.directed[0, :2, 0], torch.ones(2))
    assert torch.count_nonzero(filtered.directed[0, 2]) == 0


def test_all_zero_rows_remain_zero_and_finite_after_filtering():
    output = _dependency_output(torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))

    filtered = filter_dependency_sinks(output, sink_threshold=0.5)

    assert torch.equal(filtered.sink_mask, torch.tensor([[False, True]]))
    assert torch.count_nonzero(filtered.directed) == 0
    assert torch.isfinite(filtered.directed).all()


def test_filtering_disabled_returns_original_output_unchanged():
    output = _dependency_output(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))

    filtered = filter_dependency_sinks(
        output,
        enabled=False,
        sink_quantile=None,
        sink_threshold=None,
    )

    assert filtered is output
    assert filtered.sink_mask is None
    assert torch.equal(filtered.directed, output.directed)
