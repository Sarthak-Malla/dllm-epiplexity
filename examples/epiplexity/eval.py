from dataclasses import dataclass
import os

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.epiplexity_oracle import OracleEpiplexitySampler, OracleEpiplexitySamplerConfig
from dllm.core.samplers.epiplexity_risk import RiskEpiplexitySampler, RiskEpiplexitySamplerConfig
from dllm.core.samplers.epiplexity_guided import GuidedEpiplexitySampler, GuidedEpiplexitySamplerConfig


@dataclass
class LLaDAEvalSamplerConfig(MDLMSamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024

@dataclass
class LLaDAEvalConfig(MDLMEvalConfig):
    max_length: int = 4096

@register_model("llada_epiplexity")
class LLaDAEpiplexityEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        sampler_type="greedy",
        **kwargs,
    ):
        eval_config = LLaDAEvalConfig()
        
        # Decide which sampler to use based on sampler_type string
        if sampler_type == "oracle":
            sampler_cls = OracleEpiplexitySampler
            # Pull out specific kwargs for config
            oracle_candidate_strategy = kwargs.pop("oracle_candidate_strategy", "mixed")
            sampler_config = OracleEpiplexitySamplerConfig(oracle_candidate_strategy=oracle_candidate_strategy)
            
        elif sampler_type == "risk":
            sampler_cls = RiskEpiplexitySampler
            risk_candidate_strategy = kwargs.pop("risk_candidate_strategy", "mixed")
            sampler_config = RiskEpiplexitySamplerConfig(risk_candidate_strategy=risk_candidate_strategy)

        elif sampler_type == "guided":
            sampler_cls = GuidedEpiplexitySampler
            guide_temperature = float(kwargs.pop("guide_temperature", 1.0))
            sampler_config = GuidedEpiplexitySamplerConfig(guide_temperature=guide_temperature, remasking="guided")
            
            # TODO: We need to load the guide_model here, but for now we expect it passed or we load a dummy/default checkpoint
            # For this evaluation script placeholder, we'll let it run without guide_model (which falls back to confidence) 
            # if we haven't trained one yet.
        else:
            sampler_cls = MDLMSampler
            sampler_config = LLaDAEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )

if __name__ == "__main__":
    cli_evaluate()
