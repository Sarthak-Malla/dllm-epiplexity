"""Run with `python -m pytest /home/juan.guevara/naruto/dllm-epiplexity/scripts/tests/test_tie.py`."""

from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.tie import TIEMember, TIESampler, TIESamplerConfig
from dllm.pipelines.tie.relay import RelayError, Vocabulary, transfer_blocks


class ToyTokenizer:
    def __init__(self, *, merged=False):
        self.tokens = {
            text: i for i, text in enumerate(
                ("[mask]", "[end]", "P", "|", "a", "b", "1", "2", "ab", "X")
            )
        }
        if not merged:
            del self.tokens["ab"]
        self.text = {i: text for text, i in self.tokens.items()}
        self.mask_token_id = 0
        self.eos_token_id = self.pad_token_id = 1
        self.all_special_ids = [0, 1]
        self.name_or_path = "toy"

    def get_vocab(self):
        return self.tokens

    def encode(self, text, **kwargs):
        if text == "ab" and "ab" in self.tokens:
            return [self.tokens["ab"]]
        return [self.tokens[char] for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(self.text[token] for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        return [self.tokens["P"]]


class ToyDenoiser(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, reasoning_token, *, stable=True, shift=False):
        super().__init__()
        self.reasoning_token = reasoning_token
        self.stable = stable
        self.shift = shift
        self.canvases = []

    def forward(self, tokens, attention_mask):
        assert attention_mask.bool().all()
        self.canvases.append(tokens[0].tolist())
        marker = self.canvases[-1].index(3)
        logits = torch.zeros((*tokens.shape, 11))
        logits[..., self.reasoning_token] = 10
        logits[..., 10] = 99  # An unassigned output row must never be committed.
        start = marker if self.shift else marker + 1
        logits[:, start:, 6 if self.stable or len(self.canvases) % 2 else 7] = 20
        return SimpleNamespace(logits=logits)


def config(relay_interval):
    return TIESamplerConfig(
        max_new_tokens=6, answer_tokens=2, block_size=2, steps=6,
        relay_interval=relay_interval, answer_marker="|",
    )


def members():
    return [
        TIEMember("llada", ToyDenoiser(4, stable=False), ToyTokenizer()),
        TIEMember("dream", ToyDenoiser(8, shift=True), ToyTokenizer(merged=True),
                  right_shift_logits=True),
    ]


def test_native_candidates_match_independent_runs_with_no_relay():
    output = TIESampler(members(), config(0)).sample([[2], [2]])
    independent = tuple(
        TIESampler([member], config(2)).sample([[2]]).selected for member in members()
    )
    assert output.completions == independent
    assert output.relays == ()
    assert output.selected.member == "dream"
    assert output.selected.answer_ids == (6, 6)
    assert independent[0].token_changes > independent[1].token_changes == 0


def test_periodic_js_uses_distribution_mass_and_divergence():
    from dllm.pipelines.tie.periodic_js_ensemble import OwnerWrite, common_support_js, fusion_decision

    nearly_equal = torch.tensor([0.51, 0.49])
    low_js, _, _ = common_support_js(nearly_equal, {0: 0.49, 1: 0.51})
    assert low_js < 0.001
    same_top = torch.tensor([0.60, 0.40, 0.0])
    high_js, _, _ = common_support_js(same_top, {0: 0.60, 1: 0.0, 2: 0.40})
    assert high_js == pytest.approx(0.4)
    write = OwnerWrite(0, 0, torch.tensor([0.60, 0.40]), ([], []))
    assert fusion_decision(write, {1: 0.4}, gated=False)[1]["decision"] == "low_coverage"
    peer = {0: 0.05, 1: 0.95}
    candidate, ungated = fusion_decision(write, peer, gated=False)
    assert candidate == 1 and ungated["decision"] == "fused"
    assert fusion_decision(write, peer, gated=True)[1]["decision"] == "high_divergence"


def test_periodic_js_preserves_owner_schedule_without_support(monkeypatch):
    from dllm.pipelines.tie.periodic_js_ensemble import PeriodicJSEnsemble

    native = TIESampler(members(), config(0)).sample([[2], [2]]).completions[0]
    sampler = TIESampler(members(), config(0))
    algorithm = PeriodicJSEnsemble(sampler)
    rounds = []

    def no_common_support(template, blocks, block, proposals, writes, counts, stats, observations, gated):
        rounds.append((block, [write.slot for write in writes], counts))
        return {}

    monkeypatch.setattr(algorithm, "project_round", no_common_support)
    state, template = sampler.initialize([[2], [2]])
    output, details = algorithm.generate(state, template, gated=True)
    assert output["reasoning_ids"] == native.reasoning_ids
    assert output["answer_ids"] == native.answer_ids
    assert output["forward_calls"] == native.forward_calls
    assert details["stats"]["rounds"] == 2
    assert [block for block, _, _ in rounds] == [0, 1]
    assert all(len(set(slots)) == len(slots) == sum(counts) == 2 for _, slots, counts in rounds)


@pytest.mark.parametrize("owner_index", [0, 1], ids=["llada-owner", "dream-owner"])
def test_selective_js_routes_on_second_step_and_skips_peer_work(monkeypatch, owner_index):
    from dllm.pipelines.tie.periodic_js_ensemble import ROUTING_CONFIDENCE_THRESHOLD, PeriodicJSEnsemble

    pair = members()
    owner, peer = pair[owner_index], pair[1 - owner_index]
    original_forward = owner.model.forward

    def soften_second_round_last_write(tokens, attention_mask):
        result = original_forward(tokens, attention_mask)
        if len(owner.model.canvases) == 4:
            marker = owner.model.canvases[-1].index(3)
            result.logits[:, :marker, owner.model.reasoning_token] = 1.0
        return result

    monkeypatch.setattr(owner.model, "forward", soften_second_round_last_write)
    sampler = TIESampler(pair, config(0))
    states = sampler.initialize([[2], [2]])
    output, details = PeriodicJSEnsemble(sampler, owner_index=owner_index).generate(
        states[owner_index], states[1 - owner_index], gated=True,
        confidence_threshold=ROUTING_CONFIDENCE_THRESHOLD,
    )
    first, second = details["routing"]
    assert first["min_owner_confidence"] >= ROUTING_CONFIDENCE_THRESHOLD
    assert not first["routed"] and first["peer_forwards"] == first["token_changes"] == 0
    assert second["min_owner_confidence"] < ROUTING_CONFIDENCE_THRESHOLD
    assert second["routed"] and second["peer_forwards"] == len(peer.model.canvases)
    conditional = details["stats"].get("conditional_forwards", 0)
    assert (conditional > 0) == (owner.name == "dream")
    assert second["peer_forwards"] == 3 + conditional
    assert output["forward_calls"] == len(sampler.schedule)
    assert details["stats"]["scheduled_positions"] == 4


def test_selective_js_dream_owner_keeps_native_canvas_when_all_rounds_skip():
    from dllm.pipelines.tie.periodic_js_ensemble import PeriodicJSEnsemble

    sampler = TIESampler(members(), config(0))
    result = PeriodicJSEnsemble(sampler, owner_index=1).sample(
        [[2], [2, 2]], confidence_threshold=0.0,
    )
    branches = result["branches"]
    assert branches["selective_js"] == branches["dream"]
    assert branches["periodic_js"]["member"] == "dream"
    assert result["fusion_diagnostics"]["selective_js"]["stats"].get("peer_forwards", 0) == 0


def test_live_relay_uses_donor_prefix_with_a_different_native_length():
    pair = members()
    output = TIESampler(pair, config(2)).sample([[2], [2]], capture_relay_canvases=True)
    first = output.relays[0]
    assert first.leader == "dream"
    assert first.applied
    assert first.before["llada"] == [[4, 4], [0, 0], [0, 0]]
    assert first.after["llada"] == [[4, 5, 4, 5], [0, 0], [0, 0]]
    assert pair[0].model.canvases[2] == [2, 4, 5, 4, 5, 0, 0, 3, 0, 0]
    assert all(completion.forward_calls == 6 for completion in output.completions)
    assert all(0 not in c.reasoning_ids + c.answer_ids for c in output.completions)
    assert [event.step for event in output.relays] == [2, 4]


def test_relay_preserves_separate_gaps_and_termination():
    source = Vocabulary.from_tokenizer(ToyTokenizer(merged=True))
    target = Vocabulary.from_tokenizer(ToyTokenizer())
    assert transfer_blocks([[8, 0, 8, 0, 0, 1]], source, target) == [
        [4, 5, 0, 4, 5, 0, 0, 1]
    ]
    assert transfer_blocks([[4, 5, 0, 1]], target, source) == [[8, 0, 1]]


def test_lossy_relay_leaves_both_states_intact():
    pair = members()
    pair[1].model.reasoning_token = 9
    pair[0].tokenizer.encode = lambda text, **kwargs: [4] if text != "|" else [3]
    output = TIESampler(pair, config(2)).sample([[2], [2]], capture_relay_canvases=True)
    first = output.relays[0]
    assert not first.applied
    assert "round-trip" in first.reason
    assert first.after == first.before
    assert pair[0].model.canvases[2] == [2, 4, 4, 0, 0, 3, 0, 0]


def test_relay_rejects_split_unicode_fragments():
    tokenizer = ToyTokenizer()
    tokenizer.text[4] = "\ufffd"
    with pytest.raises(RelayError, match="incomplete Unicode"):
        transfer_blocks(
            [[4, 0]], Vocabulary.from_tokenizer(tokenizer),
            Vocabulary.from_tokenizer(ToyTokenizer()),
        )


@pytest.mark.parametrize("mode, response", [("relay", "abababab| 11"), ("llada", "aaaa| 12")])
def test_gsm8k_eval_entrypoint_loads_and_traces_members(monkeypatch, tmp_path, capsys, mode, response):
    from dllm.pipelines.tie import eval as tie_eval

    monkeypatch.setattr(
        tie_eval, "get_tokenizer",
        lambda args: ToyTokenizer(merged="Dream" in args.model_name_or_path),
    )
    monkeypatch.setattr(
        tie_eval, "get_model",
        lambda args, **kwargs: ToyDenoiser(8, shift=True) if "Dream" in args.model_name_or_path
        else ToyDenoiser(4, stable=False),
    )
    trace = tmp_path / "relays.jsonl"
    harness = tie_eval.TIEEvalHarness(
        mode=mode, device="cpu", max_new_tokens=6, answer_tokens=2, block_size=2, steps=6,
        relay_interval=2, trace_path=trace, progress_log_interval=1,
    )
    # The tiny tokenizer uses a one-token answer marker.
    harness.sampler = TIESampler(list(harness.sampler.members), config(2))
    result = harness.generate_until([SimpleNamespace(args=("Q", {}), doc_id=1)])
    assert result == [response]
    assert ('"leader": "dream"' in trace.read_text()) == (mode == "relay")
    logs = capsys.readouterr()
    assert "Loading llada model on cpu" in logs.out
    assert "1/1 questions complete (doc 1" in logs.out
    assert f"{mode} GSM8K" in logs.err


def test_commit_gaps_follow_positions_and_can_promote_an_unselected_slot():
    from dllm.pipelines.tie.commit_actions import gap_candidates

    block = [4, 0, 0, 4, 0, 5, 0, 4]
    ranked = [1, 2, 6, 4]
    assert gap_candidates(block, ranked, [4] * 4, 0, {0, 1}, 2, False) == [(1, 2)]
    assert gap_candidates(block, ranked, [4] * 4, 0, {0, 1}, 2, True) == [(6,), (4,)]


def test_contextual_alignment_rejects_boundary_merges():
    from dllm.pipelines.tie.commit_actions import contextual_ids

    vocab = Vocabulary.from_tokenizer(ToyTokenizer(merged=True))
    assert contextual_ids(vocab, "a", "1", "b") == [6]
    with pytest.raises(RelayError, match="boundary"):
        contextual_ids(vocab, "a", "b", "")


def test_paired_branch_changes_only_the_chosen_write(monkeypatch):
    from dllm.pipelines.tie.commit_actions import CommitExperiment

    sampler = TIESampler(members(), config(0))
    experiment = CommitExperiment(sampler, "diagnostic")
    monkeypatch.setattr(
        "dllm.pipelines.tie.commit_actions.gap_candidates",
        lambda block, ranked, *args: [(ranked[0],)],
    )
    monkeypatch.setattr(
        experiment, "prepare",
        lambda *args, **kwargs: {"ids_b": [5], "ids_a": [4]},
    )
    sampler.schedule = [(2, 0)] * 15 + sampler.schedule
    # Observe forwards without consuming a reasoning slot before the checkpoint.
    result = experiment.sample([[2], [2]], 0)
    assert result["attempts"][0]["writes"]["replace"] == [[0], [5]]
    keep, edit = (result["branches"][name] for name in ("keep", "replace"))
    assert keep["reasoning_ids"] == (4, 4, 4, 4)
    assert edit["reasoning_ids"] == (5, 4, 4, 4)
    assert keep["forward_calls"] == edit["forward_calls"]


def test_shared_audit_uses_both_owners_and_matches_commit_counts(monkeypatch):
    from dllm.pipelines.tie import shared_audit

    class OrderedDenoiser(ToyDenoiser):
        def forward(self, tokens, attention_mask):
            output = super().forward(tokens, attention_mask)
            marker = self.canvases[-1].index(3)
            for index in range(marker):
                output.logits[0, index, self.reasoning_token] += (
                    index if self.shift else marker - index
                )
            return output

    pair = [
        TIEMember("llada", OrderedDenoiser(4), ToyTokenizer()),
        TIEMember("dream", OrderedDenoiser(5, shift=True), ToyTokenizer(), right_shift_logits=True),
    ]
    monkeypatch.setattr(shared_audit, "CHECKPOINTS", (1, 3))
    sampler = TIESampler(pair, config(0))
    record = shared_audit.SharedCanvasAudit(sampler).sample([[2], [2]])
    for member, token in (("llada", 4), ("dream", 5)):
        split = record["owners"][member]
        assert split["baseline"]["reasoning_ids"] == (token,) * 4
        assert len(split["checkpoints"]) == 2
        for event in split["checkpoints"]:
            assert event["order_disagrees"]
            for label in ("peer_positions", "owner_positions"):
                branch = event["alternatives"][label]
                assert len(branch["slots"]) == len(event["default_slots"])
                assert branch["ids"] == [token] * len(branch["slots"])
                assert branch["completion"]["forward_calls"] == split["baseline"]["forward_calls"]


def test_shared_audit_rejects_forced_position_checkpoints(monkeypatch):
    from dllm.pipelines.tie import shared_audit

    sampler = TIESampler(members(), TIESamplerConfig())
    shared_audit.SharedCanvasAudit(sampler)
    monkeypatch.setattr(shared_audit, "CHECKPOINTS", (16, 48, 80))
    with pytest.raises(ValueError, match="checkpoint 16 requires a choice"):
        shared_audit.SharedCanvasAudit(sampler)


def test_action_fusion_swaps_one_slot_and_scores_answer_in_both_models(monkeypatch):
    from dllm.pipelines.tie import action_fusion

    class SpacedTokenizer(ToyTokenizer):
        def __init__(self):
            super().__init__()
            self.tokens[" "] = self.tokens.pop("X")
            self.text[self.tokens[" "]] = " "

    pair = [
        TIEMember("llada", ToyDenoiser(4), SpacedTokenizer()),
        TIEMember("dream", ToyDenoiser(5, shift=True), SpacedTokenizer(),
                  right_shift_logits=True),
    ]
    monkeypatch.setattr(action_fusion, "ACTION_CHECKPOINTS", (1, 3))
    sampler = TIESampler(pair, config(0))
    record = action_fusion.ActionFusionDiagnostic(sampler).sample([[2], [2]])
    baseline = record["baseline"]
    assert baseline["reasoning_ids"] == (4,) * 4
    assert baseline["scores"]["llada"] < 0
    assert baseline["scores"]["dream"] < 0
    assert len(record["checkpoints"]) == 2
    for event in record["checkpoints"]:
        assert event["baseline_slots"] == [0]
        assert event["alternate_slots"] == [1]
        assert event["baseline_ids"] == event["alternate_ids"] == [4]
        assert event["alternate"]["response"] == baseline["response"]
    assert record["scoring_cost"]["llada"][0] == record["scoring_cost"]["dream"][0] == 1
    assert pair[0].model.canvases[-1] == pair[1].model.canvases[-1] == [2, 4, 4, 4, 4, 3, 0, 0, 0]


def test_extracted_answer_span_accepts_merged_punctuation_only():
    from dllm.pipelines.tie.action_fusion import native_answer_span

    class AnswerTokenizer(ToyTokenizer):
        def __init__(self):
            super().__init__()
            self.tokens.update({".": 10, "1.": 11, "a1": 12})
            self.text.update({10: ".", 11: "1.", 12: "a1"})

        def encode(self, text, **kwargs):
            if not text:
                return []
            for merged in ("1.", "a1"):
                if text.startswith(merged):
                    return [self.tokens[merged], *self.encode(text[len(merged):])]
            return [self.tokens[text[0]], *self.encode(text[1:])]

    vocabulary = Vocabulary.from_tokenizer(AnswerTokenizer())
    assert native_answer_span(vocabulary, "|", "1", ".") == ([3], [11], [])
    with pytest.raises(RelayError, match="would mask visible reasoning"):
        native_answer_span(vocabulary, "a", "1", "")


def test_wide_action_fusion_uses_three_distinct_alternatives_from_one_state(monkeypatch):
    from dllm.pipelines.tie import action_fusion

    monkeypatch.setattr(action_fusion, "ACTION_CHECKPOINTS", (1,))
    monkeypatch.setattr(action_fusion, "answer_reconstruction", lambda *args: -1.0)
    pair = [
        TIEMember("llada", ToyDenoiser(4), ToyTokenizer()),
        TIEMember("dream", ToyDenoiser(5, shift=True), ToyTokenizer(),
                  right_shift_logits=True),
    ]
    sampler = TIESampler(pair, TIESamplerConfig(
        max_new_tokens=6, answer_tokens=2, block_size=4, steps=4, answer_marker="|",
    ))
    record = action_fusion.ActionFusionDiagnostic(sampler, wide=True).sample([[2], [2]])
    events = record["checkpoints"]
    assert [event["action"] for event in events] == ["defer_last", "defer_first", "defer_two"]
    assert [event["baseline_slots"] for event in events] == [[0, 1]] * 3
    assert [event["alternate_slots"] for event in events] == [[0, 2], [1, 2], [2, 3]]
    assert [event["alternate_ids"] for event in events] == [[4, 4]] * 3
    assert [event["rollout_forwards"] for event in events] == [3, 3, 3]
    assert record["baseline"]["reasoning_ids"] == (4,) * 4


def test_short_action_fusion_hides_future_reasoning_and_matches_saved_actions(monkeypatch):
    import json

    from dllm.pipelines.tie import action_fusion, short_action_fusion

    monkeypatch.setattr(action_fusion, "ACTION_CHECKPOINTS", (1,))
    monkeypatch.setattr(short_action_fusion, "ACTION_CHECKPOINTS", (1,))
    monkeypatch.setattr(action_fusion, "answer_reconstruction", lambda *args: -1.0)
    scored = {}

    def score(members, prompts, marker, responses, *, answer_spans):
        scored.update(answer_spans)
        return ({response: {m.name: -1.0 for m in members} for response in responses},
                {m.name: [0, 0] for m in members})

    monkeypatch.setattr(short_action_fusion, "score_responses", score)
    pair = [
        TIEMember("llada", ToyDenoiser(4), ToyTokenizer()),
        TIEMember("dream", ToyDenoiser(5, shift=True), ToyTokenizer(),
                  right_shift_logits=True),
    ]
    sampler = TIESampler(pair, TIESamplerConfig(
        max_new_tokens=10, answer_tokens=2, block_size=4, steps=6, answer_marker="|",
    ))
    prompts = [[2], [2]]
    reference = json.loads(json.dumps(
        action_fusion.ActionFusionDiagnostic(sampler, wide=True).sample(prompts)
    ))
    reference["baseline"]["correct"] = True
    for replay in reference["checkpoints"]:
        replay["alternate"]["correct"] = True
    diagnostic = short_action_fusion.ShortActionFusionDiagnostic(sampler)
    result = diagnostic.sample(prompts, reference, lambda text: text)
    event, = result["checkpoints"]
    assert event["baseline_slots"] == [0, 1]
    assert event["alternate_slots"] == [2, 3]
    assert event["short"]["keep"]["response"] == "aaaa| 11"
    assert event["short"]["swap"]["forwards"] == 3
    assert event["full"]["keep"]["response"] == "aaaaaaaa| 11"
    assert scored["aaaa| 11"] == (6, 8)
    fresh = diagnostic.sample(prompts, None, lambda text: text)
    fresh_event, = fresh["checkpoints"]
    assert fresh_event["short"] == event["short"]
    assert fresh_event["full"]["swap"]["response"] == event["full"]["swap"]["response"]
    assert fresh_event["full"]["swap"]["forwards"] == 5
    assert fresh["baseline"] == result["baseline"]
    reference["checkpoints"][-1]["alternate_ids"] = [5, 5]
    with pytest.raises(ValueError, match="Full-rollout action differs"):
        diagnostic.sample(prompts, reference, lambda text: text)


def test_peer_block_proposal_preserves_prefix_and_matches_owner_width(monkeypatch):
    from dllm.pipelines.tie import peer_block_proposals

    monkeypatch.setattr(peer_block_proposals, "BLOCK_STARTS", (3,))
    monkeypatch.setattr(peer_block_proposals, "BLOCK_STEPS", 2)
    monkeypatch.setattr(peer_block_proposals, "PEER_WIDTH", 4)
    pair = [
        TIEMember("llada", ToyDenoiser(4), ToyTokenizer()),
        TIEMember("dream", ToyDenoiser(8, shift=True), ToyTokenizer(merged=True),
                  right_shift_logits=True),
    ]
    sampler = TIESampler(pair, TIESamplerConfig(
        max_new_tokens=10, answer_tokens=2, block_size=4, steps=6, answer_marker="|",
    ))
    record = peer_block_proposals.PeerBlockProposals(sampler).sample([[2], [2]], 0)
    event, = record["checkpoints"]
    assert event["dream_width"] == 2 and event["llada_width"] == 4
    assert event["peer"]["reasoning_ids"] == (4, 4, 4, 4, 4, 5, 4, 5)
    assert record["baseline"]["reasoning_ids"] == (4,) * 8
    assert event["proposal_forwards"] == {"dream": 2, "llada": 2}
    assert event["continuation_forwards_per_branch"] == 2


def test_exact_projection_follows_text_after_a_split_token():
    from dllm.pipelines.tie.exact_text_fusion import ExactTextFusion, mix_projected, peer_canvas

    owner = TIEMember("llada", ToyDenoiser(8), ToyTokenizer(merged=True))
    peer = TIEMember("dream", ToyDenoiser(4), ToyTokenizer())
    fusion = ExactTextFusion(TIESampler([owner, peer], config(0)))
    native, offsets = peer_canvas(
        owner.vocabulary, peer.vocabulary, [[0, 0], [0, 0], [0, 0]], 0,
        {0: 8, 1: 9}, [2], [3],
    )
    assert native == [2, 0, 0, 0, 0, 0, 3, 0, 0]
    assert offsets == {0: (1, 2), 1: (3,)}
    projected = fusion.project([((4, 5), 0.7), ((4, 9), 0.3)], ([], []), ([], []), 11)
    assert projected == {8: 0.7}
    base = torch.zeros(11)
    base[8], base[9] = 0.45, 0.55
    q = mix_projected(base, list(projected), list(projected.values()))
    assert q.sum() == pytest.approx(1.0)
    assert q[8] == pytest.approx(0.45 * 0.65 + 0.35)
    assert q.argmax() == 8


def test_exact_fusion_changes_tokens_and_positions_in_separate_arms():
    from dllm.pipelines.tie.exact_text_fusion import ExactTextFusion

    class SlotDenoiser(ToyDenoiser):
        def forward(self, tokens, attention_mask):
            output = super().forward(tokens, attention_mask)
            logits = output.logits
            answer = self.canvases[-1].index(3)
            logits[:, 1:answer] = -50
            if self.reasoning_token == 4:
                logits[:, 1:answer:2, 4] = 0.16
                logits[:, 1:answer:2, 9] = 0
                logits[:, 2:answer:2, 5] = 0.04
                logits[:, 2:answer:2, 4] = 0
            else:
                logits[:, 1:answer:2, 9] = 10
                logits[:, 2:answer:2, 5] = 10
            return output

    pair = [
        TIEMember("llada", SlotDenoiser(4), ToyTokenizer()),
        TIEMember("dream", SlotDenoiser(9), ToyTokenizer()),
    ]
    sampler = TIESampler(pair, config(0))
    record = ExactTextFusion(sampler).sample([[2], [2]])
    stats = record["fusion_diagnostics"]
    assert stats["tokens"]["token_changes"] == 2
    assert stats["tokens"]["position_changes"] == 0
    assert stats["positions"]["token_changes"] == 0
    assert stats["positions"]["position_changes"] == 2
    assert stats["joint"]["position_changes"] == stats["joint"]["token_changes"] == 2
    assert stats["joint"]["peer_forwards"] == 4
    assert record["branches"]["keep"]["reasoning_ids"] == (4, 5, 4, 5)
    assert record["branches"]["joint"]["reasoning_ids"] == (9, 5, 9, 5)
    assert {branch["forward_calls"] for branch in record["branches"].values()} == {6}


def test_commit_support_ranks_the_token_dream_will_write():
    from dllm.pipelines.tie.commit_support import CommitSupport

    class PositionDenoiser(ToyDenoiser):
        def forward(self, tokens, attention_mask):
            output = super().forward(tokens, attention_mask)
            logits = output.logits
            marker = self.canvases[-1].index(3)
            start = 0 if self.shift else 1
            end = marker - 1 if self.shift else marker
            logits[:, start:end] = -50
            if self.shift:
                logits[:, start:end:2, 4] = 0.16
                logits[:, start:end:2, 9] = 0
                logits[:, start + 1:end:2, 5] = 0.04
                logits[:, start + 1:end:2, 4] = 0
            else:
                logits[:, start:end:2, 9] = 10
                logits[:, start + 1:end:2, 5] = 0.85
                logits[:, start + 1:end:2, 4] = 0
            return output

    llada = TIEMember("llada", PositionDenoiser(9), ToyTokenizer())
    dream = TIEMember(
        "dream", PositionDenoiser(4, shift=True), ToyTokenizer(), right_shift_logits=True,
    )
    sampler = TIESampler([llada, dream], config(0))
    record = CommitSupport(sampler).sample([[2], [2]], 154)
    stats = record["fusion_diagnostics"]
    assert stats["positions"]["position_changes"] == 0
    assert stats["support"]["position_changes"] == 2
    assert stats["support"]["token_changes"] == stats["positions"]["token_changes"] == 0
    assert stats["support"]["peer_forwards"] == stats["positions"]["peer_forwards"] == 4
    assert stats["dream_multi"]["extra_forwards"] == 6
    assert record["branches"]["support"]["reasoning_ids"] == (4, 5, 4, 5)
    assert record["branches"]["support"]["member"] == "dream"
    assert record["branches"]["llada"]["reasoning_ids"] == (9, 5, 9, 5)
    replication = CommitSupport(sampler, replication=True).sample([[2], [2]], 154)
    for key in ("branches", "fusion_diagnostics"):
        assert replication[key] == {
            arm: result for arm, result in record[key].items() if arm != "support"
        }


def test_shared_audit_replays_different_text_inside_fixed_gap():
    from types import SimpleNamespace
    from dllm.pipelines.tie.commit_actions import CommitExperiment
    from dllm.pipelines.tie.shared_audit import SharedCanvasAudit

    owner = TIEMember("llada", ToyDenoiser(4), ToyTokenizer())
    peer = TIEMember("dream", ToyDenoiser(4), ToyTokenizer())
    sampler = TIESampler([owner, peer], TIESamplerConfig(
        max_new_tokens=6, answer_tokens=2, block_size=4, steps=6, answer_marker="|",
    ))
    state = sampler.initialize([[2], [2]])[0]
    state.blocks[0] = torch.tensor([4, 0, 0, 4])
    helper = CommitExperiment(SimpleNamespace(members=(owner, peer), config=sampler.config), "diagnostic")
    helper.cost = {"llada": [0, 0], "dream": [0, 0]}
    helper.dream_prompt = [2]
    canvas = torch.cat((state.prompt, state.blocks[0], state.marker, state.blocks[1]))
    with torch.inference_mode():
        event = SharedCanvasAudit(sampler).probe(
            helper, len(sampler.schedule), state, 0, 1, canvas,
            torch.tensor([1, 2]), torch.tensor([5, 5]), torch.tensor([0.9, 0.8]), 2,
        )
    assert event["text_checks"][0]["a"] == "bb"
    assert event["text_checks"][0]["b"] == "aa"
    action = event["alternatives"]["peer_text"]
    assert action["slots"] == event["default_slots"]
    assert action["completion"]["reasoning_ids"] == (4, 4, 4, 4)
    assert state.blocks[0].tolist() == [4, 0, 0, 4]
