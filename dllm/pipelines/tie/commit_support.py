"""Run commit support with python -m dllm.pipelines.tie.eval --model tie."""

import torch

from .exact_text_fusion import ExactTextFusion


MODE = "commit_support"
REPLICATION_MODE = "fused_position_replication"


class CommitSupport:
    """Compare position scores while keeping Dream's native token choices."""

    def __init__(self, sampler, *, replication=False):
        self.sampler = sampler
        self.fusion = ExactTextFusion(sampler, owner_index=1)
        self.position_modes = ("positions",) if replication else ("positions", "support")

    @torch.inference_mode()
    def sample(self, prompts, doc_id):
        llada, dream = self.sampler.initialize(prompts)
        branches, diagnostics = {}, {}
        branches["llada"], _ = self.fusion.generate(llada, None, "keep")
        branches["dream"], _ = self.fusion.generate(dream, None, "keep")

        for mode in self.position_modes:
            template, state = self.sampler.initialize(prompts)
            branches[mode], diagnostics[mode] = self.fusion.generate(state, template, mode)

        _, draw = self.sampler.initialize(prompts)
        generator = torch.Generator(device=draw.member.device).manual_seed(1234 + 1009 * doc_id + 1)
        resampled, _ = self.fusion.generate(draw, None, "keep", generator=generator)
        selected = min((dream, draw), key=lambda state: state.history.rank(final=True))
        branches["dream_multi"] = dict(branches["dream"] if selected is dream else resampled)
        diagnostics["dream_multi"] = {
            "used_resample": selected is draw,
            "extra_forwards": draw.forward_calls,
            "extra_processed_tokens": draw.processed_tokens,
        }
        return {"branches": branches, "fusion_diagnostics": diagnostics}
