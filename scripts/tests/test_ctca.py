"""Run with: pytest scripts/tests/test_ctca.py -v"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from dllm.pipelines.tse.ctca.alignment import align_embeddings_procrustes
from dllm.pipelines.tse.ctca.cache import CTCACacheManager
from dllm.pipelines.tse.ctca.canvas import (
    build_canvas_overlap_matrix,
    build_model_canvas_view,
    encode_with_offsets,
    spatial_warp_probabilities,
)
from dllm.pipelines.tse.ctca.projection import (
    build_sparse_topk_vocab_projection,
    project_vocab_fused,
    project_vocab_sparse_topk,
)


class ToyTokenizer:
    def __init__(self, tokens, *, mask_token, pad_token):
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._tokens = list(tokens)
        self.mask_token_id = self._vocab[mask_token]
        self.pad_token_id = self._vocab[pad_token]
        self.eos_token_id = self.pad_token_id
        self.unk_token_id = 0
        self.all_special_ids = [self.mask_token_id, self.pad_token_id]

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, token_ids, **kwargs):
        return "".join(
            self._tokens[int(token_id)]
            for token_id in token_ids
            if int(token_id) not in self.all_special_ids
        )

    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        ids = []
        offsets = []
        position = 0
        ordinary = sorted(
            (
                token
                for token, token_id in self._vocab.items()
                if token_id not in self.all_special_ids
            ),
            key=len,
            reverse=True,
        )
        while position < len(text):
            token = next(
                (piece for piece in ordinary if text.startswith(piece, position)),
                None,
            )
            if token is None:
                position += 1
                continue
            ids.append(self._vocab[token])
            offsets.append((position, position + len(token)))
            position += len(token)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class SlowToyTokenizer(ToyTokenizer):
    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        if return_offsets_mapping:
            raise NotImplementedError("slow tokenizer")
        return super().__call__(text, return_offsets_mapping=False, **kwargs)


@pytest.mark.parametrize(
    ("auxiliary", "master", "expected"),
    [
        ([(0, 2)], [(0, 2)], [[1.0]]),
        ([(0, 2)], [(0, 1), (1, 2)], [[0.5, 0.5]]),
        ([(0, 1), (1, 2)], [(0, 2)], [[1.0], [1.0]]),
        ([(0, 4)], [(1, 3)], [[0.5]]),
    ],
)
def test_overlap_matrix_handles_identical_split_merged_and_partial_spans(
    auxiliary, master, expected
):
    overlap = build_canvas_overlap_matrix(auxiliary, master)

    assert overlap.is_sparse
    assert torch.allclose(overlap.to_dense(), torch.tensor(expected))


def test_spatial_warp_uses_auxiliary_to_master_overlap():
    probabilities = torch.tensor([[0.8, 0.2]])
    overlap = build_canvas_overlap_matrix([(0, 2)], [(0, 1), (1, 2)])

    warped = spatial_warp_probabilities(probabilities, overlap)

    assert torch.allclose(warped, torch.tensor([[0.4, 0.1], [0.4, 0.1]]))


def test_model_canvas_uses_native_mask_and_allows_different_length():
    master = ToyTokenizer(["ab", "c", "[PAD]", "[MASK]"], mask_token="[MASK]", pad_token="[PAD]")
    auxiliary = ToyTokenizer(
        ["a", "b", "c", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    view = build_model_canvas_view(
        [0],
        [0, 1, master.mask_token_id],
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_token_id=master.mask_token_id,
        model_mask_token_id=auxiliary.mask_token_id,
    )

    generated = view.input_ids[view.generation_slice]
    assert generated.tolist() == [0, 1, 2, auxiliary.mask_token_id]
    assert view.generation_length == 4
    assert view.overlap_matrix.shape == (4, 3)
    assert view.overlap_matrix.to_dense()[-1].tolist() == [0.0, 0.0, 1.0]


def test_slow_tokenizer_offsets_are_derived_from_decoded_prefixes():
    tokenizer = SlowToyTokenizer(
        ["ab", "c", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    token_ids, offsets = encode_with_offsets(tokenizer, "abc")

    assert token_ids == [0, 1]
    assert offsets == [(0.0, 2.0), (2.0, 3.0)]


def test_procrustes_recovers_synthetic_rotation():
    tokenizer = ToyTokenizer(
        ["a", "b", "c", "d", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )
    auxiliary = F.normalize(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.2, 0.3], [0.4, 0.5]]
        ),
        dim=-1,
    )
    expected_rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    master = auxiliary @ expected_rotation

    result = align_embeddings_procrustes(
        auxiliary,
        master,
        tokenizer,
        tokenizer,
        num_anchors=4,
        min_anchors=4,
    )

    assert torch.allclose(result.rotation, expected_rotation, atol=1e-5)


def test_procrustes_rejects_too_few_shared_anchors():
    auxiliary_tokenizer = ToyTokenizer(
        ["a", "x", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    master_tokenizer = ToyTokenizer(
        ["a", "y", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )

    with pytest.raises(ValueError, match="too few shared anchors"):
        align_embeddings_procrustes(
            torch.randn(4, 2),
            torch.randn(4, 2),
            auxiliary_tokenizer,
            master_tokenizer,
            num_anchors=2,
            min_anchors=2,
        )


def test_chunked_projection_matches_dense_and_normalizes_rows():
    spatial = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    auxiliary = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    master = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    temperature = 0.4
    vocabulary_map = torch.softmax(
        F.normalize(auxiliary, dim=-1) @ F.normalize(master, dim=-1).T / temperature,
        dim=-1,
    )
    expected = spatial @ vocabulary_map
    expected = expected / expected.sum(dim=-1, keepdim=True)

    actual = project_vocab_fused(
        spatial,
        auxiliary,
        master,
        temperature=temperature,
        chunk_size=2,
    )

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2))
    assert actual.shape == (2, 4)


def test_sparse_topk_projection_matches_dense_when_topk_covers_vocab():
    spatial = torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float32)
    auxiliary = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    master = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        dtype=torch.float32,
    )

    dense = project_vocab_fused(
        spatial,
        auxiliary,
        master,
        temperature=0.5,
        chunk_size=1,
    )
    top_indices, top_weights = build_sparse_topk_vocab_projection(
        F.normalize(auxiliary, dim=-1),
        F.normalize(master, dim=-1),
        top_k=master.shape[0],
        temperature=0.5,
        chunk_size=1,
    )
    sparse = project_vocab_sparse_topk(
        spatial,
        top_indices,
        top_weights,
        master_vocab_size=master.shape[0],
        chunk_size=1,
    )

    assert torch.allclose(sparse, dense, atol=1e-6)
    assert torch.allclose(sparse.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_cache_reuses_pair_signature_and_rejects_malformed_metadata(tmp_path):
    metadata = {"auxiliary_id": "aux", "master_id": "master", "anchors": 4}
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return torch.eye(2)

    manager = CTCACacheManager(tmp_path)
    first = manager.get_or_create(metadata, builder)
    second = CTCACacheManager(tmp_path).get_or_create(metadata, builder)
    assert torch.equal(first, second)
    assert calls == 1

    other_metadata = {"auxiliary_id": "other", "master_id": "master", "anchors": 4}
    assert manager.path_for(metadata) != manager.path_for(other_metadata)

    manager = CTCACacheManager(tmp_path)
    malformed_metadata = {"pair": "malformed"}
    torch.save(
        {"metadata": {"pair": "different"}, "rotation": torch.eye(2)},
        manager.path_for(malformed_metadata),
    )
    with pytest.raises(ValueError, match="Malformed CTCA cache metadata"):
        manager.get_or_create(malformed_metadata, builder)
