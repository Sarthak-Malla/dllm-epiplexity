"""Run with: pytest scripts/tests/test_ctca_sampler.py -v"""

from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.tse import ctca_sampler as ctca_sampler_module
from dllm.pipelines.tse import eval as tse_eval
from dllm.pipelines.tse import loader
from dllm.pipelines.tse import scoring
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
        scoring,
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
        [torch.tensor([0, tokenizer.mask_token_id]), torch.tensor([1])],
        max_new_tokens=2,
        steps=2,
        block_size=1,
        selection_mode="tse",
        capture_logits=True,
    )

    assert generated.tolist() == [
        [0, tokenizer.mask_token_id, 0, 0],
        [1, 0, 0, tokenizer.pad_token_id],
    ]
    assert sampler.last_paired_logits[0][0].tolist() == [[0, 2], [0, 3], [1, 1], [1, 2]]


def test_sampler_counts_consecutive_right_masks_across_blocks():
    master = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary = CharacterTokenizer(["a", "b", "c", "ab"], "<mask>", "<pad>")
    sampler = CTCATSESampler(
        FixedMaskedModel(len(master.get_vocab())),
        FixedMaskedModel(len(auxiliary.get_vocab()), preferred_token_id=3),
        master,
        auxiliary,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        min_anchors=3,
        projection_temperature=0.5,
    )
    mask = master.mask_token_id
    canvas = torch.tensor(
        [
            [2, mask, mask, mask, mask, master.pad_token_id],
            [0, 1, mask, 1, mask, mask],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    trace = {}
    _, projected, positions = sampler._forward_aligned_probabilities(
        canvas,
        attention,
        [1, 2],
        [torch.tensor([2]), torch.tensor([0, 1])],
        max_new_tokens=4,
        projection_range=range(0, 3),
        temperature_master=1.0,
        temperature_auxiliary=1.0,
        fusion_device=torch.device("cpu"),
        trace_callback=lambda event, payload: trace.update(payload),
    )

    assert positions.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4]]
    assert trace["masked_slots_to_right"].tolist() == [3, 2, 1, 0, 1]
    assert (projected[[0, 1, 2, 4], 0] > projected[3, 0]).all()

    sampler.sample(
        [[2]],
        max_new_tokens=2,
        steps=2,
        block_size=1,
        capture_logits=True,
    )
    first_step, second_step = sampler.last_aligned_probabilities
    assert first_step[2][0, 0] > second_step[2][0, 0]


@pytest.mark.parametrize("ctca", [False, True])
@pytest.mark.parametrize("selection_mode", ["baseline", "tse"])
def test_sampler_resolves_all_slots_when_both_models_prefer_masks(ctca, selection_mode):
    master = CharacterTokenizer(["a", "b"], "[MASK]", "[PAD]")
    auxiliary = (
        CharacterTokenizer(["a", "b", "c"], "<mask>", "<pad>") if ctca else master
    )
    model_a = FixedMaskedModel(
        len(master.get_vocab()), preferred_token_id=master.mask_token_id
    )
    model_b = FixedMaskedModel(
        len(auxiliary.get_vocab()), preferred_token_id=auxiliary.mask_token_id
    )
    if ctca:
        sampler = CTCATSESampler(
            model_a,
            model_b,
            master,
            auxiliary,
            "cpu",
            "cpu",
            master_id="a",
            auxiliary_id="b",
            cache_dir=None,
            min_anchors=2,
        )
    else:
        sampler = TSESampler(model_a, model_b, master, "cpu", "cpu")
    generated = sampler.sample(
        [[master.mask_token_id, 0]],
        max_new_tokens=4,
        steps=4,
        block_size=2,
        selection_mode=selection_mode,
        baseline_model="b",
        temperature=0.8,
        capture_logits=True,
    )

    assert generated[0, :2].tolist() == [master.mask_token_id, 0]
    assert (generated[0, 2:] != master.mask_token_id).all()
    if ctca:
        for _, master_probs, auxiliary_probs in sampler.last_aligned_probabilities:
            assert (master_probs[:, master.mask_token_id] == 0).all()
            assert (auxiliary_probs[:, master.mask_token_id] == 0).all()


def test_dream_logits_predict_the_following_canvas_slot():
    class PositionModel(FixedMaskedModel):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(
                logits=torch.nn.functional.one_hot(
                    input_ids, num_classes=self.config.vocab_size
                ).float()
            )

    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    dream = PositionModel(len(tokenizer.get_vocab()))
    dream.config.model_type = "Dream"
    same_position = PositionModel(len(tokenizer.get_vocab()))
    sampler = TSESampler(dream, same_position, tokenizer, "cpu", "cpu")
    input_ids = torch.tensor([[0, 1, 2]])

    dream_logits, other_logits = sampler.forward_pair(input_ids, torch.ones_like(input_ids))

    assert dream_logits.argmax(dim=-1).tolist() == [[0, 0, 1]]
    assert torch.equal(other_logits.argmax(dim=-1), input_ids)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [("llada", [1, 2, 0, 1]), ("Dream", [0, 1, 2, 0])],
)
def test_single_checkpoint_greedy_matches_ctca_master_endpoint(
    monkeypatch, model_type, expected
):
    class PositionModel(FixedMaskedModel):
        def __init__(self, tokenizer):
            super().__init__(len(tokenizer.get_vocab()) + 1)
            self.config.model_type = model_type
            self.mask_id = tokenizer.mask_token_id
            self.calls = 0

        def forward(self, input_ids, attention_mask=None):
            self.calls += 1
            logits = torch.zeros(*input_ids.shape, self.config.vocab_size)
            positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
            logits.scatter_(-1, (positions % 3).unsqueeze(-1), 5.0)
            logits[..., self.mask_id :] = 50
            return SimpleNamespace(logits=logits)

    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    model = PositionModel(tokenizer)
    loaded = []

    def get_model(args, **kwargs):
        loaded.append(args.model_name_or_path)
        return model

    monkeypatch.setattr(loader, "get_model", get_model)
    monkeypatch.setattr(loader, "get_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(
        tse_eval.accelerate,
        "Accelerator",
        lambda: SimpleNamespace(process_index=0, num_processes=1),
    )
    harness = tse_eval.TSEEvalHarness(pretrained="fixture/model", device="cpu")
    sampling_kwargs = dict(max_new_tokens=4, steps=4, block_size=2)
    greedy = harness.sampler.sample(
        [[0]],
        **sampling_kwargs,
        selection_mode=harness.selection_mode,
        baseline_model=harness.baseline_model,
    )

    assert loaded == ["fixture/model"]
    assert model.calls == 4
    assert greedy[0, 1:].tolist() == expected

    aux_tokenizer = CharacterTokenizer(["a", "b", "c", "d"], "<mask>", "<pad>")
    endpoint = CTCATSESampler(
        model,
        FixedMaskedModel(len(aux_tokenizer.get_vocab())),
        tokenizer,
        aux_tokenizer,
        "cpu",
        "cpu",
        master_id="a",
        auxiliary_id="b",
        cache_dir=None,
        min_anchors=3,
    )
    assert torch.equal(endpoint.sample([[0]], **sampling_kwargs, alpha=1.0), greedy)


@pytest.mark.parametrize("checkpoint_kind", ["local", "unversioned", "hub_revision"])
def test_startup_rebuilds_projection_unless_disk_reuse_is_requested(
    tmp_path, monkeypatch, checkpoint_kind
):
    tokenizer = CharacterTokenizer(["a", "b"], "[MASK]", "[PAD]")
    model_a = FixedMaskedModel(len(tokenizer.get_vocab()))
    model_b = FixedMaskedModel(len(tokenizer.get_vocab()))
    cache_dir = tmp_path / "projections"
    names = ("org/model-a", "org/model-b")
    for model, name in zip((model_a, model_b), names):
        if checkpoint_kind == "local":
            path = tmp_path / name
            path.mkdir(parents=True)
            model.config._name_or_path = str(path)
        else:
            model.config._name_or_path = name
        if checkpoint_kind != "unversioned":
            model.config._commit_hash = "revision"

    def build_sampler(**kwargs):
        return CTCATSESampler(
            model_a, model_b, tokenizer, tokenizer, "cpu", "cpu",
            master_id="a", auxiliary_id="b", cache_dir=str(cache_dir), min_anchors=2,
            master_cache_id=names[0], auxiliary_cache_id=names[1],
            **kwargs,
        )

    first = build_sampler()
    assert first.ctca_projection_diagnostics["projection_version"] == 42
    assert first.ctca_projection_diagnostics["sparse_topk_cache_rebuilt"]["cpu"]
    assert build_sampler().ctca_projection_diagnostics["sparse_topk_cache_rebuilt"]["cpu"]

    def unexpected_rebuild(*args):
        pytest.fail("projection was rebuilt again inside the same run")

    monkeypatch.setattr(first.aligner, "_build_sparse_tables", unexpected_rebuild)
    first.aligner.warm_projection_cache(
        first._aligner_auxiliary_id, torch.device("cpu:0")
    )

    reused = build_sampler(force_rebuild=False)
    rebuilt = reused.ctca_projection_diagnostics["sparse_topk_cache_rebuilt"]["cpu"]
    assert rebuilt == (checkpoint_kind != "hub_revision")
    if checkpoint_kind == "hub_revision":
        assert len(list(cache_dir.glob("decoded_relative_sparse_topk_v42_*.pt"))) == 1
    else:
        assert not cache_dir.exists()
