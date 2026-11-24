"""Run with python -m pytest /home/sarthak.malla/dllm-epiplexity/scripts/tests/test_tse_eval.py."""

import pickle
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from lm_eval.api.instance import Instance

from dllm.pipelines.tse import eval as tse_eval
from dllm.pipelines.tse.models import TSEConfig


class FakeTokenizer:
    eos_token_id = 10
    eot_token_id = 11
    eos_token = "</s>"
    eot_token = "<|eot_id|>"

    def __init__(self, role):
        self.role = role
        self.inputs = []

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        if self.role == "master":
            content = content.strip()
        return f"{self.role}: {content}"

    def __call__(self, text, *, add_special_tokens, **kwargs):
        self.inputs.append((text, add_special_tokens))
        return {"input_ids": torch.tensor([[7]])}

    def batch_decode(self, rows, **kwargs):
        pieces = {
            1: "first",
            2: "second",
            3: " STOP tail",
            4: " tail",
            5: "<|eo",
            6: "t_id|>",
        }
        return ["".join(pieces[token] for token in row) for row in rows]


def make_harness():
    harness = tse_eval.TSEEvalHarness.__new__(tse_eval.TSEEvalHarness)
    harness.config = TSEConfig("test/master", "test/auxiliary", max_new_tokens=4)
    harness.batch_size = 2
    harness.selection_mode = "tse"
    harness.baseline_model = "b"
    harness.tokenizer = FakeTokenizer("master")
    harness.auxiliary_tokenizer = FakeTokenizer("auxiliary")
    harness._progress_wandb_run = None
    return harness


def test_chat_request_keeps_auxiliary_prompt_when_master_text_collides():
    renderer = make_harness()
    contexts = [
        renderer.apply_chat_template([{"role": "user", "content": content}])
        for content in ("question", "question ")
    ]
    assert contexts[0] == contexts[1]
    requests = [
        Instance("generate_until", {}, (context, {}), index)
        for index, context in enumerate(contexts)
    ]
    harness = make_harness()
    forwards = []

    def sample(prompts, **kwargs):
        forwards.append(kwargs["auxiliary_inputs"])
        answer_token = len(forwards)
        return torch.tensor([[7, answer_token, 10, 4, 4]] * len(prompts))

    harness.sampler = SimpleNamespace(sample=sample)
    requests = pickle.loads(pickle.dumps(requests))
    assert harness.generate_until(requests) == ["first", "first"]
    assert harness.generate_until(requests) == ["second", "second"]
    assert harness.tokenizer.inputs == [("master: question", False)] * 4
    assert harness.auxiliary_tokenizer.inputs == [
        ("auxiliary: question", False),
        ("auxiliary: question ", False),
    ] * 2


def test_plain_text_is_tokenized_by_both_models():
    harness = make_harness()
    harness._encode_prompts(("plain text",))

    assert harness.tokenizer.inputs == [("plain text", True)]
    assert harness.auxiliary_tokenizer.inputs == [("plain text", True)]


@pytest.mark.parametrize(
    ("generated", "eot_token_id", "until"),
    [
        ([1, 10, 4, 4], 11, None),
        ([1, 11, 4, 4], 11, []),
        ([1, 5, 6, 4], None, ["Q:"]),
        ([1, 5, 6, 4], None, ["id|>"]),
        ([1, 3, 10, 4], 11, " STOP"),
    ],
)
def test_decode_trims_only_generation_at_native_and_task_stops(
    generated, eot_token_id, until
):
    harness = make_harness()
    harness.tokenizer.eot_token_id = eot_token_id
    request_kwargs = {"until": until}
    original_kwargs = deepcopy(request_kwargs)

    # The prompt itself contains EOS and must stay outside stop detection.
    answer = harness._decode_outputs(
        torch.tensor([[10, 7, *generated]]),
        [torch.tensor([10, 7])],
        (request_kwargs,),
    )

    assert answer == ["first"]
    assert request_kwargs == original_kwargs


def test_cli_ignores_lm_eval_request_and_response_caches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(tse_eval, "cli_evaluate", calls.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py", "--model", "tse_llada", "--tasks", "gsm8k_cot",
            "--cache_requests", "true", "--use_cache", str(tmp_path / "cache"),
        ],
    )

    tse_eval.main()

    assert len(calls) == 1
    assert calls[0].use_cache is None
    assert calls[0].cache_requests is None
    assert calls[0].tasks == "gsm8k_cot"


def test_harness_rejects_programmatic_response_cache():
    with pytest.raises(ValueError, match="requires fresh decoding"):
        make_harness().set_cache_hook(SimpleNamespace(dbdict={}))
