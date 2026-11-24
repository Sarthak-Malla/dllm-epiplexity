"""Run with: pytest scripts/tests/test_ctca.py -v"""

import pytest
import torch
import torch.nn.functional as F

from dllm.pipelines.tse.ctca.aligner import CrossTokenizerAligner
from dllm.pipelines.tse.ctca.cache import CTCACacheManager
from dllm.pipelines.tse.ctca.canvas import (
    build_canvas_overlap_matrix,
    build_model_canvas_view,
    encode_with_offsets,
    spatial_warp_probabilities,
)
from dllm.pipelines.tse.ctca.projection import (
    match_decoded_tokens,
    project_vocab_sparse_topk,
)
from dllm.pipelines.tse.ctca.relative import (
    build_sparse_topk_relative_anchor_projection,
    collect_relative_anchor_ids,
    project_vocab_relative_exact,
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


class CountingToyTokenizer(ToyTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_calls = 0

    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        self.encode_calls += 1
        return super().__call__(
            text, return_offsets_mapping=return_offsets_mapping, **kwargs
        )


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


def test_model_canvas_run_cache_preserves_view_outputs():
    master = ToyTokenizer(
        ["ab", "c", "d", "[PAD]", "[MASK]"],
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    auxiliary = ToyTokenizer(
        ["a", "b", "c", "d", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    uncached = build_model_canvas_view(
        [0],
        [0, 1, master.mask_token_id, 0, 1],
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_token_id=master.mask_token_id,
        model_mask_token_id=auxiliary.mask_token_id,
    )
    cached = build_model_canvas_view(
        [0],
        [0, 1, master.mask_token_id, 0, 1],
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_token_id=master.mask_token_id,
        model_mask_token_id=auxiliary.mask_token_id,
        run_cache={},
    )

    assert torch.equal(cached.input_ids, uncached.input_ids)
    assert cached.offsets == uncached.offsets
    assert cached.generation_start == uncached.generation_start
    assert torch.allclose(
        cached.overlap_matrix.to_dense(), uncached.overlap_matrix.to_dense()
    )


def test_model_canvas_run_cache_reuses_retokenized_committed_runs():
    master = ToyTokenizer(
        ["ab", "c", "[PAD]", "[MASK]"],
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    auxiliary = CountingToyTokenizer(
        ["a", "b", "c", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )
    run_cache = {}

    first = build_model_canvas_view(
        [0],
        [0, 1, master.mask_token_id, 0, 1],
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_token_id=master.mask_token_id,
        model_mask_token_id=auxiliary.mask_token_id,
        run_cache=run_cache,
    )
    assert auxiliary.encode_calls == 1

    second = build_model_canvas_view(
        [0],
        [0, 1, master.mask_token_id, 0, 1],
        master_tokenizer=master,
        model_tokenizer=auxiliary,
        master_mask_token_id=master.mask_token_id,
        model_mask_token_id=auxiliary.mask_token_id,
        run_cache=run_cache,
    )

    assert auxiliary.encode_calls == 1
    assert torch.equal(second.input_ids, first.input_ids)
    assert second.offsets == first.offsets


def test_slow_tokenizer_offsets_are_derived_from_decoded_prefixes():
    tokenizer = SlowToyTokenizer(
        ["ab", "c", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    token_ids, offsets = encode_with_offsets(tokenizer, "abc")

    assert token_ids == [0, 1]
    assert offsets == [(0.0, 2.0), (2.0, 3.0)]


def test_relative_anchor_selection_supports_auto_and_integer_caps():
    auxiliary_tokenizer = ToyTokenizer(
        ["a", "b", "c", "x", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )
    master_tokenizer = ToyTokenizer(
        ["a", "b", "c", "y", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )

    auto = collect_relative_anchor_ids(
        auxiliary_tokenizer,
        master_tokenizer,
        auxiliary_vocab_size=6,
        master_vocab_size=6,
        num_anchors="auto",
        min_anchors=3,
    )
    capped = collect_relative_anchor_ids(
        auxiliary_tokenizer,
        master_tokenizer,
        auxiliary_vocab_size=6,
        master_vocab_size=6,
        num_anchors=2,
        min_anchors=2,
    )

    assert auto.total_anchors == 3
    assert auto.selected_anchors == 3
    assert capped.selected_anchors == 2


def test_relative_anchor_rejects_too_few_shared_anchors():
    auxiliary_tokenizer = ToyTokenizer(
        ["a", "x", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    master_tokenizer = ToyTokenizer(
        ["a", "y", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )

    with pytest.raises(ValueError, match="too few shared anchors"):
        collect_relative_anchor_ids(
            auxiliary_tokenizer,
            master_tokenizer,
            auxiliary_vocab_size=4,
            master_vocab_size=4,
            num_anchors="auto",
            min_anchors=2,
        )


def test_relative_anchors_keep_case_and_boundary_markers():
    tokens = ["the", "The", "Ġthe", "<pad>", "<mask>"]
    auxiliary = ToyTokenizer(tokens, mask_token="<mask>", pad_token="<pad>")
    master = ToyTokenizer(tokens[::-1], mask_token="<mask>", pad_token="<pad>")

    anchors = collect_relative_anchor_ids(
        auxiliary,
        master,
        auxiliary_vocab_size=len(tokens),
        master_vocab_size=len(tokens),
        min_anchors=3,
    )

    assert anchors.auxiliary_anchor_ids.tolist() == [1, 0, 2]
    assert anchors.master_anchor_ids.tolist() == [3, 4, 2]


@pytest.mark.parametrize("projection_mode", ["exact", "sparse_topk"])
def test_projection_prefers_unambiguous_decoded_matches_and_preserves_mass(
    projection_mode,
):
    class AliasTokenizer(ToyTokenizer):
        def __init__(self, decoded_by_token):
            super().__init__(
                [*decoded_by_token, "<pad>", "<mask>"],
                mask_token="<mask>",
                pad_token="<pad>",
            )
            self.decoded_by_token = decoded_by_token

        def decode(self, ids, **kwargs):
            return "".join(
                self.decoded_by_token.get(self._tokens[int(index)], "")
                for index in ids
            )

    master = AliasTokenizer(
        {
            "duplicate": "word",
            "shared": "word",
            "Ġsaid": " said",
            "bad": "\ufffd",
        }
    )
    auxiliary = AliasTokenizer(
        {
            "ambiguous": "word",
            "shared": "word",
            "▁said": " said",
            "added_said": " said",
            "other_bad": "\ufffd",
            "unspaced": "said",
        }
    )
    matches = match_decoded_tokens(
        auxiliary, master, auxiliary_vocab_size=8, master_vocab_size=6
    )
    assert matches.auxiliary_ids.tolist() == [1, 2, 3]
    assert matches.master_ids.tolist() == [1, 2, 2]
    assert matches.unmatched_ids.tolist() == [0, 4, 5, 6]

    aligner = CrossTokenizerAligner(
        master, torch.eye(6), master_id="master", cache_dir=None
    )
    aligner.register_auxiliary_model(
        "auxiliary",
        auxiliary,
        torch.eye(8),
        projection_mode=projection_mode,
        projection_top_k=6,
        temperature=0.5,
        chunk_size=2,
        min_anchors=1,
    )
    probabilities = torch.zeros(3, 8)
    probabilities[0, 1:4] = torch.tensor([0.2, 0.3, 0.5])
    probabilities[1, 0] = 1
    probabilities[2] = 0.4 * probabilities[0] + 0.6 * probabilities[1]

    projected = aligner.project_spatial_probabilities("auxiliary", probabilities)

    assert torch.allclose(projected[0], torch.tensor([0, 0.2, 0.8, 0, 0, 0]))
    assert (projected[1, :master.mask_token_id] > 0).all()
    assert (projected[:, master.mask_token_id] == 0).all()
    assert torch.allclose(projected[2], 0.4 * projected[0] + 0.6 * projected[1])
    assert torch.allclose(projected.sum(dim=-1), torch.ones(3))


@pytest.mark.parametrize("projection_mode", ["exact", "sparse_topk"])
def test_projection_excludes_masks_and_unassigned_rows_before_ranking(projection_mode):
    master = ToyTokenizer(
        ["a", "b", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    auxiliary = ToyTokenizer(
        ["a", "b", "x", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    # A mask ID can be present even when the tokenizer omits it from this list.
    master.all_special_ids = [master.pad_token_id]
    auxiliary.all_special_ids = [auxiliary.pad_token_id]
    # The excluded master IDs are the best matches for "x" in relative space.
    master_embeddings = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]] + [[0.0, 1.0]] * 65
    )
    auxiliary_embeddings = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    aligner = CrossTokenizerAligner(
        master, master_embeddings, master_id="master", cache_dir=None
    )
    aligner.register_auxiliary_model(
        "aux",
        auxiliary,
        auxiliary_embeddings,
        projection_mode=projection_mode,
        min_anchors=2,
        chunk_size=8,
    )

    probabilities = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]
    )
    projected = aligner.project_spatial_probabilities("aux", probabilities)

    assert torch.allclose(projected[0, :3], torch.full((3,), 1 / 3))
    assert torch.count_nonzero(projected[:, 3:]) == 0
    assert torch.count_nonzero(projected[1]) == 0
    diagnostics = aligner.projection_diagnostics("aux")
    assert diagnostics["selected_anchors"] == 2
    assert diagnostics["relative_projection_rows"] == 2


@pytest.mark.parametrize("projection_mode", ["exact", "sparse_topk"])
def test_first_piece_projection_uses_all_mass_only_when_the_encoding_fits(
    projection_mode,
):
    master = ToyTokenizer(
        ["cent", "imeters", "kilo", "me", "ters", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )
    auxiliary = ToyTokenizer(
        ["centimeters", "kilometers", "cent", "kilo", "xmeters", "<pad>", "<mask>"],
        mask_token="<mask>",
        pad_token="<pad>",
    )
    rng = torch.Generator().manual_seed(2)
    aligner = CrossTokenizerAligner(
        master, torch.randn(7, 3, generator=rng), master_id="master", cache_dir=None
    )
    aligner.register_auxiliary_model(
        "aux",
        auxiliary,
        torch.randn(7, 4, generator=rng),
        projection_mode=projection_mode,
        temperature=0.5,
        min_anchors=2,
        chunk_size=2,
    )
    # The first three rows have the same input and different amounts of room.
    # xmeters can only be tokenized lossily by this master tokenizer.
    probabilities = torch.tensor([[0.5, 0.25, 0.25, 0, 0, 0, 0]] * 3)
    probabilities = torch.cat((probabilities, torch.eye(7)[4:5]))
    projected = aligner.project_spatial_probabilities(
        "aux", probabilities, masked_slots_to_right=torch.tensor([0, 1, 2, 5])
    )
    relative = aligner.project_spatial_probabilities("aux", probabilities)
    kilo_fallback = aligner.project_spatial_probabilities("aux", torch.eye(7)[1:2])[0]
    cent = torch.eye(7)[master.get_vocab()["cent"]]
    kilo = torch.eye(7)[master.get_vocab()["kilo"]]

    assert torch.allclose(projected[0], relative[0])
    assert torch.allclose(projected[1], 0.75 * cent + 0.25 * kilo_fallback)
    assert torch.allclose(projected[2], 0.75 * cent + 0.25 * kilo)
    assert torch.allclose(projected[3], relative[3])
    assert torch.allclose(projected.sum(dim=-1), torch.ones(4))
    assert aligner.projection_diagnostics("aux")["first_piece_matches"] == 2


def test_first_piece_requires_an_assigned_master_token_id():
    class UnassignedPieceTokenizer(ToyTokenizer):
        def __call__(self, text, **kwargs):
            if text == "centimeters":
                vocab = self.get_vocab()
                return {"input_ids": [len(vocab), vocab["imeters"]]}
            return super().__call__(text, **kwargs)

    master = UnassignedPieceTokenizer(
        ["cent", "imeters", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    auxiliary = ToyTokenizer(
        ["centimeters", "<pad>", "<mask>"], mask_token="<mask>", pad_token="<pad>"
    )
    matches = match_decoded_tokens(
        auxiliary,
        master,
        auxiliary_vocab_size=len(auxiliary.get_vocab()),
        master_vocab_size=len(master.get_vocab()) + 1,
    )

    assert matches.prefix_master_ids.numel() == 0
    assert 0 in matches.unmatched_ids


def test_relative_anchor_sparse_maps_matching_tokens_with_unequal_dimensions():
    auxiliary = F.normalize(
        torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    master = F.normalize(
        torch.tensor(
            [
                [3.0, 0.0],
                [0.0, 3.0],
                [2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    top_indices, top_weights = build_sparse_topk_relative_anchor_projection(
        auxiliary,
        auxiliary[:2],
        master,
        master[:2],
        top_k=1,
        anchor_temperature=0.01,
        projection_temperature=0.05,
        chunk_size=2,
    )

    assert top_indices[:2, 0].tolist() == [0, 1]
    assert top_weights.shape == (4, 1)


def test_relative_anchor_sparse_matches_exact_when_topk_covers_vocab():
    spatial = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    auxiliary = F.normalize(
        torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32),
        dim=-1,
    )
    master = F.normalize(
        torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 2.0]], dtype=torch.float32),
        dim=-1,
    )

    top_indices, top_weights = build_sparse_topk_relative_anchor_projection(
        auxiliary,
        auxiliary,
        master,
        master[:2],
        top_k=master.shape[0],
        anchor_temperature=0.05,
        projection_temperature=0.4,
        chunk_size=1,
    )
    sparse = project_vocab_sparse_topk(
        spatial,
        top_indices,
        top_weights,
        master_vocab_size=master.shape[0],
        chunk_size=1,
    )
    exact = project_vocab_relative_exact(
        spatial,
        auxiliary,
        auxiliary,
        master,
        master[:2],
        anchor_temperature=0.05,
        projection_temperature=0.4,
        chunk_size=1,
    )

    assert torch.allclose(sparse, exact, atol=1e-6)
    assert torch.allclose(sparse.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_cache_reuses_tensor_payloads(tmp_path):
    metadata = {"auxiliary_id": "aux", "master_id": "master", "anchors": 4}
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {
            "top_indices": torch.tensor([[0, 1]], dtype=torch.long),
            "top_weights": torch.tensor([[0.7, 0.3]], dtype=torch.float32),
        }

    manager = CTCACacheManager(tmp_path)
    first, first_rebuilt = manager.get_or_create_tensors(
        metadata,
        artifact="relative_sparse_topk",
        builder=builder,
        required_keys=("top_indices", "top_weights"),
    )
    second, second_rebuilt = CTCACacheManager(tmp_path).get_or_create_tensors(
        metadata,
        artifact="relative_sparse_topk",
        builder=builder,
        required_keys=("top_indices", "top_weights"),
    )

    assert first_rebuilt is True
    assert second_rebuilt is False
    assert calls == 1
    assert torch.equal(first["top_indices"], second["top_indices"])
    assert torch.equal(first["top_weights"], second["top_weights"])
