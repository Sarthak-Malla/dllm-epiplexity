"""Run with: pytest scripts/tests/test_ctca_sampler.py -v"""

from types import SimpleNamespace

import torch

from dllm.pipelines.tse import ctca_sampler as ctca_sampler_module
from dllm.pipelines.tse.ctca_sampler import CTCATSESampler
from dllm.pipelines.tse.sampler import TSESampler


class CharacterTokenizer:
    def __init__(self, ordinary_tokens, mask_token, pad_token):
        tokens = ordinary_tokens + [pad_token, mask_token]
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._tokens = tokens
        self.mask_token_id = self._vocab[mask_token]
        self.pad_token_id = self._vocab[pad_token]
        self.eos_token_id = self.pad_token_id
        self.unk_token_id = 0
        self.all_special_ids = [self.pad_token_id, self.mask_token_id]

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
        for index, character in enumerate(text):
            if character in self._vocab and self._vocab[character] not in self.all_special_ids:
                ids.append(self._vocab[character])
                offsets.append((index, index + 1))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FixedMaskedModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=3, preferred_token_id=0):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.preferred_token_id = preferred_token_id
        torch.manual_seed(vocab_size)
        torch.nn.init.normal_(self.embeddings.weight)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(
            *input_ids.shape,
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., self.preferred_token_id] = 8.0
        return SimpleNamespace(logits=logits)


def test_ctca_sampler_runs_unequal_vocabularies_and_commits_master_slots():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    master_model = FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2)
    auxiliary_model = FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3)
    sampler = CTCATSESampler(
        master_model,
        auxiliary_model,
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        num_anchors=3,
        min_anchors=3,
    )

    generated = sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        capture_logits=True,
    )

    output = generated[0, 1:]
    assert output.shape == (2,)
    assert not torch.any(output == master_tokenizer.mask_token_id)
    assert output.max() < len(master_tokenizer.get_vocab())
    assert sampler.last_aligned_probabilities
    master_probabilities = sampler.last_aligned_probabilities[0][1]
    auxiliary_probabilities = sampler.last_aligned_probabilities[0][2]
    assert master_probabilities.shape == auxiliary_probabilities.shape


def test_ctca_sampler_masks_unassigned_model_output_ids():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    master_vocab_size = len(master_tokenizer.get_vocab()) + 2
    auxiliary_vocab_size = len(auxiliary_tokenizer.get_vocab()) + 2
    sampler = CTCATSESampler(
        FixedMaskedModel(
            master_vocab_size,
            hidden_size=2,
            preferred_token_id=master_vocab_size - 1,
        ),
        FixedMaskedModel(
            auxiliary_vocab_size,
            hidden_size=3,
            preferred_token_id=auxiliary_vocab_size - 1,
        ),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        num_anchors=3,
        min_anchors=3,
    )

    generated = sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        capture_logits=True,
    )

    assigned_vocab_size = len(master_tokenizer.get_vocab())
    assert generated[0, 1:].max() < assigned_vocab_size
    _, master_probabilities, auxiliary_probabilities = (
        sampler.last_aligned_probabilities[0]
    )
    assert torch.all(master_probabilities[:, assigned_vocab_size:] == 0)
    assert torch.all(auxiliary_probabilities[:, assigned_vocab_size:] == 0)
    assert torch.allclose(master_probabilities.sum(dim=-1), torch.ones(2))
    assert torch.allclose(auxiliary_probabilities.sum(dim=-1), torch.ones(2))


def test_ctca_sampler_ranks_positions_by_fused_top_probability(monkeypatch):
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
        FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        num_anchors=3,
        min_anchors=3,
    )
    monkeypatch.setattr(
        ctca_sampler_module,
        "agreement_factor",
        lambda divergence, *args: torch.zeros_like(divergence),
    )
    fusion_payloads = []

    sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        trace_callback=lambda event, payload: fusion_payloads.append(payload)
        if event == "fusion"
        else None,
    )

    assert fusion_payloads
    payload = fusion_payloads[0]
    assert torch.allclose(
        payload["scores"], payload["fused_probabilities"].max(dim=-1).values
    )
    assert torch.equal(
        payload["consensus_scores"], torch.zeros_like(payload["scores"])
    )


def test_ctca_sampler_projects_only_current_block_for_static_weighting():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
        FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        num_anchors=3,
        min_anchors=3,
    )

    sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=4,
        steps=4,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    active_positions, master_probabilities, auxiliary_probabilities = (
        sampler.last_aligned_probabilities[0]
    )
    assert active_positions.tolist() == [[0, 1], [0, 2]]
    assert master_probabilities.shape == (2, len(master_tokenizer.get_vocab()))
    assert auxiliary_probabilities.shape == master_probabilities.shape


def test_ctca_sampler_runs_sparse_topk_projection_backend():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
        FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        projection_mode="sparse_topk",
        projection_top_k=2,
        num_anchors=3,
        min_anchors=3,
    )

    generated = sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    assert not torch.any(generated[0, 1:] == master_tokenizer.mask_token_id)
    assert sampler.last_aligned_probabilities[0][1].shape == (
        2,
        len(master_tokenizer.get_vocab()),
    )
    assert sampler.last_aligned_probabilities[0][2].shape == (
        2,
        len(master_tokenizer.get_vocab()),
    )


def test_ctca_sampler_runs_exact_relative_projection_backend():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
        FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        projection_mode="exact",
        num_anchors=3,
        min_anchors=3,
    )

    generated = sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    assert not torch.any(generated[0, 1:] == master_tokenizer.mask_token_id)
    assert sampler.last_aligned_probabilities[0][2].shape == (
        2,
        len(master_tokenizer.get_vocab()),
    )


def test_ctca_sampler_batches_canvas_and_projection_across_samples():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
        FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
        master_tokenizer,
        auxiliary_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        projection_temperature=0.5,
        projection_chunk_size=2,
        projection_mode="sparse_topk",
        projection_top_k=2,
        num_anchors=3,
        min_anchors=3,
    )

    generated = sampler.sample(
        [torch.tensor([0]), torch.tensor([1])],
        auxiliary_inputs=[torch.tensor([0]), torch.tensor([1])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    active_positions, master_probabilities, auxiliary_probabilities = (
        sampler.last_aligned_probabilities[0]
    )
    assert generated.shape == (2, 3)
    assert not torch.any(generated[:, 1:] == master_tokenizer.mask_token_id)
    assert active_positions.tolist() == [[0, 1], [0, 2], [1, 1], [1, 2]]
    assert master_probabilities.shape == (4, len(master_tokenizer.get_vocab()))
    assert auxiliary_probabilities.shape == master_probabilities.shape


def test_ctca_sampler_run_cache_matches_uncached_canvas_path(monkeypatch):
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )

    def make_sampler():
        return CTCATSESampler(
            FixedMaskedModel(len(master_tokenizer.get_vocab()), hidden_size=2),
            FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()), hidden_size=3),
            master_tokenizer,
            auxiliary_tokenizer,
            "cpu",
            "cpu",
            master_id="a",
            auxiliary_id="b",
            cache_dir=None,
            projection_temperature=0.5,
            projection_chunk_size=2,
            num_anchors=3,
            min_anchors=3,
        )

    cached_sampler = make_sampler()
    cached = cached_sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=4,
        steps=4,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    original_build_model_canvas_view = ctca_sampler_module.build_model_canvas_view

    def build_uncached_model_canvas_view(*args, **kwargs):
        kwargs["run_cache"] = None
        return original_build_model_canvas_view(*args, **kwargs)

    monkeypatch.setattr(
        ctca_sampler_module,
        "build_model_canvas_view",
        build_uncached_model_canvas_view,
    )
    uncached_sampler = make_sampler()
    uncached = uncached_sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=4,
        steps=4,
        block_size=2,
        selection_mode="tse",
        weighting_mode="static",
        capture_logits=True,
    )

    assert torch.equal(cached, uncached)
    assert len(cached_sampler.last_aligned_probabilities) == len(
        uncached_sampler.last_aligned_probabilities
    )
    for cached_step, uncached_step in zip(
        cached_sampler.last_aligned_probabilities,
        uncached_sampler.last_aligned_probabilities,
    ):
        for cached_tensor, uncached_tensor in zip(cached_step, uncached_step):
            assert cached_tensor.shape == uncached_tensor.shape


def test_homogeneous_sampler_regression_still_commits_equal_vocabularies():
    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    model_a = FixedMaskedModel(len(tokenizer.get_vocab()))
    model_b = FixedMaskedModel(len(tokenizer.get_vocab()))
    sampler = TSESampler(model_a, model_b, tokenizer, "cpu", "cpu")

    generated = sampler.sample(
        [torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
    )

    output = generated[0, 1:]
    assert output.tolist() == [0, 0]
    assert not torch.any(output == tokenizer.mask_token_id)
