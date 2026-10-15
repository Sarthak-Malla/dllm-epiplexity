"""Run with: pytest scripts/tests/test_dream_suppression.py -v"""

from dllm.pipelines.dream.eval import _parse_token_ids
from dllm.pipelines.dream.sampler import DreamSamplerConfig


def test_dream_eval_parses_suppression_lists():
    assert _parse_token_ids("[151643;151645]") == (151643, 151645)
    assert _parse_token_ids("[]") == ()


def test_dream_sampler_config_carries_both_suppression_modes():
    config = DreamSamplerConfig(
        suppress_tokens=(151643,),
        begin_suppress_tokens=(151643, 151645),
    )

    assert config.suppress_tokens == (151643,)
    assert config.begin_suppress_tokens == (151643, 151645)
