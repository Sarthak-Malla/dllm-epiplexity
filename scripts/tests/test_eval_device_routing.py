"""Verify independent worker placement without loading weights or using GPUs.

Run on a compute node after sourcing ~/.zshrc and activating dllm:
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_eval_device_routing.py
"""

from types import SimpleNamespace

import pytest

import dllm.core.eval.base as eval_base
import dllm.utils.models as model_utils


def _mock_loader(monkeypatch, *, zero3=False):
    calls = []
    marker = object()

    def load(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return marker

    monkeypatch.setattr(model_utils.transformers.AutoModelForMaskedLM, "from_pretrained", load)
    monkeypatch.setattr(model_utils.transformers.modeling_utils,
                        "is_deepspeed_zero3_enabled", lambda: zero3)
    monkeypatch.setattr(model_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_utils, "load_peft", lambda model, args: model)
    arguments = SimpleNamespace(model_name_or_path="/tmp/unused-checkpoint", dtype="bfloat16",
                                load_in_4bit=False, attn_implementation=None)
    return arguments, calls, marker


@pytest.mark.parametrize("placement", [{"": "cuda:1"}, {"": "cpu"}, None])
def test_explicit_device_map_precedes_local_rank_without_constructing_partial_state(monkeypatch, placement):
    arguments, calls, marker = _mock_loader(monkeypatch)

    def forbidden_state():
        raise AssertionError("An explicit placement must not consult distributed local rank.")

    monkeypatch.setattr(model_utils.accelerate, "PartialState", forbidden_state)
    assert model_utils.get_model(arguments, device_map=placement) is marker
    assert calls == [{
        "path": "/tmp/unused-checkpoint", "dtype": "bfloat16", "device_map": placement,
        "quantization_config": None, "attn_implementation": None, "config": None,
    }]


def test_default_distributed_placement_keeps_local_rank(monkeypatch):
    arguments, calls, _ = _mock_loader(monkeypatch)
    monkeypatch.setattr(model_utils.accelerate, "PartialState",
                        lambda: SimpleNamespace(local_process_index=1))
    model_utils.get_model(arguments)
    assert calls[0]["device_map"] == {"": 1}


def test_zero3_still_owns_device_placement(monkeypatch):
    arguments, calls, _ = _mock_loader(monkeypatch, zero3=True)
    model_utils.get_model(arguments, device_map={"": "cuda:1"})
    assert calls[0]["device_map"] is None


@pytest.mark.parametrize("error_type", [model_utils.torch.OutOfMemoryError, MemoryError])
def test_allocation_failure_is_not_retried_with_another_model_loader(monkeypatch, error_type):
    arguments, _, _ = _mock_loader(monkeypatch)
    original_error = error_type("The assigned device has insufficient free memory.")

    def fail_load(*args, **kwargs):
        raise original_error

    def forbidden_fallback(*args, **kwargs):
        pytest.fail("Allocation failure must not trigger a second checkpoint load.")

    monkeypatch.setattr(model_utils.transformers.AutoModelForMaskedLM, "from_pretrained", fail_load)
    monkeypatch.setattr(model_utils.transformers.AutoModel, "from_pretrained", forbidden_fallback)
    with pytest.raises(error_type) as caught:
        model_utils.get_model(arguments, device_map={"": "cuda:0"})
    assert caught.value is original_error


def test_unsupported_masked_model_still_uses_auto_model_fallback(monkeypatch):
    arguments, calls, marker = _mock_loader(monkeypatch)

    def unsupported(*args, **kwargs):
        raise ValueError("This configuration is not supported by AutoModelForMaskedLM.")

    def fallback(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return marker

    monkeypatch.setattr(model_utils.transformers.AutoModelForMaskedLM, "from_pretrained", unsupported)
    monkeypatch.setattr(model_utils.transformers.AutoModel, "from_pretrained", fallback)
    assert model_utils.get_model(arguments, device_map={"": "cuda:1"}) is marker
    assert len(calls) == 1
    assert calls[0]["path"] == arguments.model_name_or_path
    assert calls[0]["device_map"] == {"": "cuda:1"}
    assert calls[0]["dtype"] == "bfloat16"


@pytest.mark.parametrize("processes", [1, 2])
def test_harness_routes_independent_loads_before_final_placement(monkeypatch, processes):
    loads, placements, prepared = [], [], []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            placements.append(device)
            return self

    model = Model()

    def load(*args, **kwargs):
        loads.append(kwargs)
        return model

    def prepare(value):
        prepared.append(value)
        return value

    accelerator = SimpleNamespace(num_processes=processes, device="cuda:1", prepare=prepare)
    monkeypatch.setattr(eval_base.accelerate, "Accelerator", lambda: accelerator)
    monkeypatch.setattr(eval_base.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(eval_base.dllm.utils, "get_model", load)
    monkeypatch.setattr(eval_base.dllm.utils, "get_tokenizer", lambda args: object())
    harness = eval_base.BaseEvalHarness(pretrained="/tmp/unused-checkpoint", device="cuda:1")
    if processes == 1:
        assert loads[0]["device_map"] == {"": "cuda:1"}
        assert placements == ["cuda:1"]
        assert not prepared
        assert harness.accelerator is None
    else:
        assert "device_map" not in loads[0]
        assert not placements
        assert prepared == [model]
        assert harness.accelerator is accelerator
