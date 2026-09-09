"""Dependency-guided MDLM sampling without counterfactual lookahead.

Run the focused CPU tests with:
    source /apps/local/conda_init.sh
    conda activate /home/sarthak.malla/.conda/envs/dllm
    export PYTHONPATH=/home/sarthak.malla/dllm-selection-ensemble
    pytest /home/sarthak.malla/dllm-selection-ensemble/scripts/tests/test_dependency_guided_decoder.py -v
"""

from dataclasses import dataclass

from dllm.core.samplers.entropy_drop import (
    EntropyDropSampler,
    EntropyDropSamplerConfig,
)


@dataclass
class DependencyNonLookaheadSamplerConfig(EntropyDropSamplerConfig):
    """Defaults for dependency candidates scored from the current base pass."""

    proposal_strategy: str = "dependency"
    dependency_cardinality_strategy: str = "entropy_budget"
    dependency_max_action_size: int = 4
    dependency_entropy_budget: float = 2.0
    dependency_size_scoring: str = "per_token"
    dependency_candidate_selector: str = "max_confidence"
    diagnostic_metadata: bool = False


class DependencyNonLookaheadSampler(EntropyDropSampler):
    """Reuse the dependency decoding loop while disabling lookahead by default."""
