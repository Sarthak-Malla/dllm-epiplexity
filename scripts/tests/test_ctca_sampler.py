"""Run with: pytest scripts/tests/test_ctca_sampler.py -v"""

import json
from types import SimpleNamespace

import torch

from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.tse import ctca_sampler as ctca_sampler_module
from dllm.pipelines.tse import sampler as sampler_module
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
        self.bos_token_id = 0
        self.unk_token_id = 0
        self.all_special_ids = [self.pad_token_id, self.mask_token_id]

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._tokens[int(token_id)]

    def decode(self, token_ids, **kwargs):
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        return "".join(
            self._tokens[int(token_id)]
            for token_id in token_ids
            if not skip_special_tokens or int(token_id) not in self.all_special_ids
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

    @property
    def device(self):
        return self.embeddings.weight.device

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(
            *input_ids.shape,
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., self.preferred_token_id] = 8.0
        return SimpleNamespace(logits=logits)


class PositionAwareDreamModel(FixedMaskedModel):
    """Small Dream stand-in whose winning token exposes positional alignment."""

    def __init__(self, vocab_size, hidden_size=3):
        super().__init__(vocab_size, hidden_size)
        self.config.model_type = "Dream"
        self.last_position_ids = None

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        self.last_position_ids = position_ids.detach().clone()
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -8.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for position in range(input_ids.shape[1]):
            logits[:, position, position % self.config.vocab_size] = 8.0
        return SimpleNamespace(logits=logits)


class PositionAwareMaskedModel(FixedMaskedModel):
    """Small MDLM stand-in with unique confidence at each canvas position."""

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(
            (*input_ids.shape, self.config.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        ordinary_vocab_size = self.config.vocab_size - 2
        for position in range(input_ids.shape[1]):
            preferred_token = position % ordinary_vocab_size
            logits[:, position, preferred_token] = 1.0 + position
        return SimpleNamespace(logits=logits)


def test_ctca_alpha_one_matches_native_mdlm_trajectory():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    master_model = PositionAwareMaskedModel(len(master_tokenizer.get_vocab()))
    auxiliary_model = FixedMaskedModel(len(auxiliary_tokenizer.get_vocab()))

    native = MDLMSampler(master_model, master_tokenizer).sample(
        [torch.tensor([0])],
        MDLMSamplerConfig(
            max_new_tokens=4,
            steps=4,
            block_size=2,
            temperature=0.0,
            probability_temperature=1.0,
            remasking="low_confidence",
            stochastic_transfer=False,
            cfg_scale=0.0,
            return_dict=True,
        ),
    )
    ctca_sampler = CTCATSESampler(
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
    ctca = ctca_sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=4,
        steps=4,
        block_size=2,
        temperature=0.0,
        remasking="low_confidence",
        stochastic_transfer=False,
        capture_history=True,
        selection_mode="tse",
        weighting_mode="static",
        alpha=1.0,
        temperature_a=1.0,
        temperature_b=1.0,
    )

    assert torch.equal(ctca, native.sequences)
    assert len(ctca_sampler.last_canvas_history) == len(native.histories)
    for ctca_canvas, native_canvas in zip(
        ctca_sampler.last_canvas_history, native.histories
    ):
        assert torch.equal(ctca_canvas, native_canvas.cpu())


def test_ctca_entropy_ranks_fused_distributions_not_top_probability(monkeypatch):
    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    distributions = torch.tensor(
        [[0.60, 0.20, 0.20, 0.0, 0.0], [0.55, 0.45, 0.0, 0.0, 0.0]]
    )

    def first_decision(remasking):
        sampler = CTCATSESampler(
            FixedMaskedModel(len(tokenizer.get_vocab())),
            FixedMaskedModel(len(tokenizer.get_vocab())),
            tokenizer,
            tokenizer,
            "cpu",
            "cpu",
            master_id="a",
            auxiliary_id="b",
            cache_dir=None,
            num_anchors=3,
            min_anchors=3,
        )

        def aligned_probabilities(canvas, *_args, **_kwargs):
            active_positions = (canvas == tokenizer.mask_token_id).nonzero(
                as_tuple=False
            )
            probabilities = distributions[active_positions[:, 1] - 1]
            return probabilities, probabilities, active_positions

        monkeypatch.setattr(sampler, "_forward_aligned_probabilities", aligned_probabilities)
        decisions = []
        sampler.sample(
            [torch.tensor([0])],
            auxiliary_inputs=[torch.tensor([0])],
            max_new_tokens=2,
            steps=2,
            block_size=2,
            remasking=remasking,
            selection_mode="tse",
            weighting_mode="static",
            alpha=0.5,
            trace_callback=lambda event, payload: (
                decisions.append(payload) if event == "decision" else None
            ),
        )
        return decisions[0]

    confidence = first_decision("low_confidence")
    entropy_ranked = first_decision("entropy")

    assert confidence["predicted_tokens"].tolist() == [0, 0]
    assert entropy_ranked["predicted_tokens"].tolist() == [0, 0]
    assert confidence["selected_positions"].tolist() == [[0, 1]]
    assert entropy_ranked["selected_positions"].tolist() == [[0, 2]]
    assert entropy_ranked["selection_score_name"] == "negative_fused_entropy"
    assert entropy_ranked["commit_reason"] == "lowest_fused_entropy"


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


def test_ctca_sampler_trace_exposes_forward_fusion_and_commit_decisions():
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
    events = []

    sampler.sample(
        [torch.tensor([0])],
        auxiliary_inputs=[torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        master_begin_suppress_tokens=list(
            range(len(master_tokenizer.get_vocab()))
        ),
        trace_callback=lambda event, payload: events.append((event, payload)),
    )

    payloads = {event: payload for event, payload in events}
    assert {"forward", "fusion", "decision", "summary"}.issubset(payloads)
    forward = payloads["forward"]
    assert forward["master_logit_adapter"] == "identity"
    assert forward["auxiliary_logit_adapter"] == "identity"
    assert torch.equal(
        forward["master_logits_raw"], forward["master_logits_sampling"]
    )
    assert torch.equal(
        forward["auxiliary_logits_raw"], forward["auxiliary_logits_sampling"]
    )
    assert forward["spatial_auxiliary_probabilities"].shape[0] == forward[
        "active_positions"
    ].shape[0]
    fusion = payloads["fusion"]
    assert fusion["jensen_shannon_divergence"].shape == fusion["scores"].shape
    assert fusion["agreement_scores"].shape == fusion["scores"].shape
    assert fusion["master_weights"].shape == fusion["scores"].shape
    assert fusion["candidate_is_assigned"].all()
    assert fusion["master_begin_suppressed"].all()
    assert torch.isneginf(fusion["scores"]).all()
    decision = payloads["decision"]
    assert decision["selected_mask"].sum() == decision[
        "scheduled_transfer_counts"
    ].sum()
    assert torch.all(decision["selection_ranks"] >= 0)
    summary = payloads["summary"]
    assert summary["sample_elapsed_seconds"] is not None
    assert summary["sample_elapsed_seconds"] >= 0
    assert sampler.last_sample_elapsed_seconds == summary["sample_elapsed_seconds"]


def test_ctca_jsonl_trace_with_dummy_model_outputs(tmp_path, capsys):
    """Trace a masked sentence through two fixed-output models and projection."""
    from scripts.tse.ctca.generate_dummy_trace import generate

    path = tmp_path / "dummy_trace.jsonl"
    generate(path)
    capsys.readouterr()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["event"] == "question"
    assert records[0]["masked_instance"] == "The cat sat on the [MASK][MASK]"
    assert records[-1]["event"] == "question_result"
    assert records[-1]["response"] == "mat."
    forwards = [r for r in records if r["event"] == "forward_position"]
    auxiliaries = [r for r in records if r["event"] == "auxiliary_masked_position"]
    fusions = [r for r in records if r["event"] == "fusion_candidate"]
    decisions = [r for r in records if r["event"] == "commit_decision"]
    assert forwards and auxiliaries and fusions and decisions
    assert len(forwards) == len(fusions) == len(decisions)
    distributions = (
        [r["master"]["distribution"] for r in forwards]
        + [r["distribution"] for r in auxiliaries]
        + [r["spatial_warp"]["distribution"] for r in forwards]
        + [r["projected_auxiliary_distribution"] for r in fusions]
        + [r["fused_distribution"] for r in fusions]
    )
    for distribution in distributions:
        assert len(distribution["top_k"]) == 3
        assert all("token_id" not in entry for entry in distribution["top_k"])
        assert all(isinstance(entry["token"], str) for entry in distribution["top_k"])
        assert abs(distribution["top_k_mass"] + distribution["remaining_mass"] - 1) < 1e-5
    assert all("master_top_k" not in record for record in fusions)
    assert all("token_id" not in record["predicted_token"] for record in fusions)
    for record in fusions:
        assert record["fusion_support"]
        support_by_token = {item[0]: item for item in record["fusion_support"]}
        for entry in record["projected_auxiliary_distribution"]["top_k"]:
            assert abs(
                support_by_token[entry["token"]][2] - entry["probability"]
            ) < 1e-7
        predicted = next(
            item for item in record["fusion_support"]
            if item[0] == record["predicted_token"]["token"]
        )
        master_weight = record["weights"]["master"]
        preview_probability = (
            master_weight * predicted[1] + (1 - master_weight) * predicted[2]
        )
        assert abs(preview_probability - record["fused_probability"]) < 1e-5
    first_master = next(
        record for record in forwards
        if record["step"] == [0, 0] and record["master"]["generation_position"] == 0
    )
    first_auxiliary = next(
        record for record in auxiliaries
        if record["step"] == [0, 0] and record["generation_position"] == 0
    )
    assert first_master["master"]["distribution"]["top_k"][0]["token"] == "mat"
    assert first_auxiliary["distribution"]["top_k"][0]["token"] == "rug"
    assert sum(record["selected"] for record in decisions) == 2


def test_ctca_sampler_applies_dream_right_shift_in_either_role():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    sampler = CTCATSESampler(
        PositionAwareDreamModel(len(master_tokenizer.get_vocab()), hidden_size=2),
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
    raw = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    adapted = sampler._adapt_logits_for_sampling("a", raw)

    assert sampler._logit_adapter_name("a") == "dream_right_shift"
    assert sampler._logit_adapter_name("b") == "identity"
    assert torch.equal(adapted[:, 0], raw[:, 0])
    assert torch.equal(adapted[:, 1:], raw[:, :-1])
    assert sampler._adapt_logits_for_sampling("b", raw) is raw


def test_ctca_sampler_passes_native_dream_position_ids_for_padded_batches():
    master_tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    auxiliary_tokenizer = CharacterTokenizer(
        ["a", "b", "c", "d"], "<mask>", "<pad>"
    )
    dream = PositionAwareDreamModel(len(master_tokenizer.get_vocab()), hidden_size=2)
    sampler = CTCATSESampler(
        dream,
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
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 3, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    sampler._forward_model("a", input_ids, attention_mask)

    assert torch.equal(
        dream.last_position_ids,
        torch.tensor([[0, 1, 2, 3], [0, 1, 1, 1]]),
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


def test_homogeneous_sampler_applies_master_begin_suppression(monkeypatch):
    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    model_a = FixedMaskedModel(len(tokenizer.get_vocab()))
    model_b = FixedMaskedModel(len(tokenizer.get_vocab()))
    sampler = TSESampler(model_a, model_b, tokenizer, "cpu", "cpu")
    observed = []
    original = sampler_module.suppress_master_commit_scores

    def record_scores(scores, predicted_tokens, token_ids):
        updated = original(scores, predicted_tokens, token_ids)
        observed.append((predicted_tokens.detach().clone(), updated.detach().clone()))
        return updated

    monkeypatch.setattr(
        sampler_module,
        "suppress_master_commit_scores",
        record_scores,
    )

    generated = sampler.sample(
        [torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        master_begin_suppress_tokens=[0],
    )

    output = generated[0, 1:]
    assert observed
    assert torch.isneginf(observed[0][1][observed[0][0] == 0]).all()
    assert output.tolist() == [0, 0]
    assert not torch.any(output == tokenizer.mask_token_id)


def test_homogeneous_sampler_never_predicts_always_suppressed_tokens():
    tokenizer = CharacterTokenizer(["a", "b", "c"], "[MASK]", "[PAD]")
    model_a = FixedMaskedModel(len(tokenizer.get_vocab()), preferred_token_id=0)
    model_b = FixedMaskedModel(len(tokenizer.get_vocab()), preferred_token_id=0)
    sampler = TSESampler(model_a, model_b, tokenizer, "cpu", "cpu")

    generated = sampler.sample(
        [torch.tensor([0])],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        selection_mode="tse",
        master_suppress_tokens=[0],
    )

    output = generated[0, 1:]
    assert not torch.any(output == 0)
    assert not torch.any(output == tokenizer.mask_token_id)
