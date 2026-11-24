"""Synchronized two-model sampling for TSE.

Run with: python -m pytest scripts/tests/test_ctca_sampler.py
"""

import math

import torch

from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler

from .ctca.projection import projection_token_mask
from .fusion import fuse_probabilities, logits_to_probabilities
from .scoring import baseline_candidates, distribution_consensus, fused_confidence
from .selection import commit_tokens, generation_mask, select_positions
from .utils import forward_token_logits, initialize_canvas, timer
from .weighting import pairwise_weights


class TSESampler:
    """Run single-model or paired forwards on a shared diffusion canvas.

    Baseline mode commits tokens using one selected model. TSE mode uses
    probability fusion and fused-confidence-based selection.
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module | None,
        tokenizer,
        model_a_device: str,
        model_b_device: str | None = None,
    ):
        self.model_a = model_a.to(model_a_device).eval()
        self.tokenizer = tokenizer
        self.model_a_device = torch.device(model_a_device)
        self.model_b_device = torch.device(model_b_device or model_a_device)
        self.model_b = (
            model_b.to(self.model_b_device).eval() if model_b is not None else None
        )
        self.scheduler = LinearAlphaScheduler()
        self.last_paired_logits = []
        self.last_consensus_scores: list[torch.Tensor] = []

    @torch.no_grad()
    @timer()
    def forward_pair(
        self,
        canvas: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward the same canvas through both models."""
        if self.model_b is None:
            raise ValueError("forward_pair requires two models")
        canvas_a = canvas.to(self.model_a_device)
        mask_a = attention_mask.to(self.model_a_device)
        canvas_b = canvas.to(self.model_b_device)
        mask_b = attention_mask.to(self.model_b_device)

        logits_a = forward_token_logits(self.model_a, canvas_a, mask_a)
        logits_b = forward_token_logits(self.model_b, canvas_b, mask_b)

        if logits_a.shape != logits_b.shape:
            raise ValueError(
                "TSE models produced incompatible logits shapes: "
                f"{tuple(logits_a.shape)} != {tuple(logits_b.shape)}"
            )

        return logits_a, logits_b

    @torch.no_grad()
    @timer()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        *,
        max_new_tokens: int,
        steps: int,
        block_size: int,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        stochastic_transfer: bool = False,
        capture_logits: bool = False,
        baseline_model: str = "a",
        selection_mode: str = "baseline",
        alpha: float = 0.5,
        temperature_a: float = 1.0,
        temperature_b: float = 1.0,
        epsilon: float = 1e-9,
        fusion_device: str | None = None,
        weighting_mode: str = "static",
        weight_temperature: float = 1.0,
        normalize_entropy: bool = True,
    ) -> torch.Tensor:
        """Generate with baseline or TSE selection on a shared canvas."""
        if not inputs:
            raise ValueError("TSESampler.sample requires at least one input")
        if steps < 1 or block_size < 1 or max_new_tokens < 1:
            raise ValueError("steps, block_size, and max_new_tokens must be positive")
        if baseline_model not in {"a", "b"}:
            raise ValueError("baseline_model must be 'a' or 'b'")
        if selection_mode not in {"baseline", "tse"}:
            raise ValueError("selection_mode must be 'baseline' or 'tse'")
        single_model = self.model_b is None
        if single_model and (selection_mode != "baseline" or baseline_model != "a"):
            raise ValueError("Single-model sampling requires baseline selection with model A")
        if weighting_mode not in {"static", "online_entropy", "per_token_margin"}:
            raise ValueError(
                "weighting_mode must be 'static', 'online_entropy', "
                "or 'per_token_margin'"
            )

        mask_id = self.tokenizer.mask_token_id
        prompts = [
            torch.as_tensor(prompt, dtype=torch.long, device=self.model_a_device)
            for prompt in inputs
        ]
        pad_id = self.tokenizer.eos_token_id
        if single_model and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
        canvas, attention_mask, prompt_lens = initialize_canvas(
            prompts, max_new_tokens, mask_id, pad_id
        )
        target_device = torch.device(
            (fusion_device or self.model_a_device)
            if selection_mode == "tse"
            else self.model_a_device
        )
        all_slots = range(max_new_tokens)
        score_all_slots = capture_logits or (
            selection_mode == "tse" and weighting_mode == "online_entropy"
        )

        self.last_paired_logits = []
        self.last_consensus_scores = []
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        allowed_tokens = None

        for start in range(0, max_new_tokens, block_size):
            block = range(start, min(start + block_size, max_new_tokens))
            transfer_counts = get_num_transfer_tokens(
                generation_mask(canvas, prompt_lens, block, mask_id),
                steps_per_block,
                self.scheduler,
                stochastic=stochastic_transfer,
            )

            for counts in transfer_counts.unbind(dim=1):
                block_mask = generation_mask(canvas, prompt_lens, block, mask_id)
                active_mask = (
                    generation_mask(canvas, prompt_lens, all_slots, mask_id)
                    if score_all_slots
                    else block_mask
                )
                positions = active_mask.nonzero(as_tuple=False)
                if single_model:
                    active_a = forward_token_logits(
                        self.model_a, canvas, attention_mask
                    )[active_mask]
                    active_b = None
                else:
                    logits_a, logits_b = self.forward_pair(canvas, attention_mask)
                    active_a = logits_a[active_mask]
                    active_b = logits_b[active_mask.to(self.model_b_device)]
                if capture_logits:
                    self.last_paired_logits.append(
                        (
                            positions.cpu(),
                            active_a.cpu(),
                            active_b.cpu() if active_b is not None else None,
                        )
                    )

                if allowed_tokens is None:
                    if single_model:
                        allowed_tokens = projection_token_mask(
                            self.tokenizer, active_a.shape[-1]
                        ).to(target_device)
                    else:
                        allowed_tokens = torch.ones(
                            active_a.shape[-1], dtype=torch.bool, device=target_device
                        )
                        allowed_tokens[mask_id] = False
                if selection_mode == "tse":
                    probabilities_a = logits_to_probabilities(
                        active_a.to(target_device),
                        temperature_a,
                        allowed_tokens=allowed_tokens,
                    )
                    probabilities_b = logits_to_probabilities(
                        active_b.to(target_device),
                        temperature_b,
                        allowed_tokens=allowed_tokens,
                    )
                    weights_a = pairwise_weights(
                        probabilities_a,
                        probabilities_b,
                        mode=weighting_mode,
                        alpha=alpha,
                        weight_temperature=weight_temperature,
                        epsilon=epsilon,
                        normalize_entropy=normalize_entropy,
                    )
                    fused = fuse_probabilities(
                        probabilities_a, probabilities_b, weights_a
                    )
                    scores, predicted_tokens = fused_confidence(fused)
                    if capture_logits:
                        consensus = distribution_consensus(
                            scores, probabilities_a, probabilities_b, weights_a, epsilon
                        )
                        self.last_consensus_scores.append(consensus.cpu())
                else:
                    logits = active_a if baseline_model == "a" else active_b
                    probabilities = logits_to_probabilities(
                        logits.to(target_device),
                        temperature_a if single_model else 1.0,
                        allowed_tokens=allowed_tokens,
                    )
                    scores, predicted_tokens = baseline_candidates(
                        probabilities, temperature, remasking
                    )

                in_block = block_mask[active_mask]
                selected_positions, selected_tokens = select_positions(
                    scores.to(canvas.device)[in_block],
                    predicted_tokens.to(canvas.device)[in_block],
                    positions[in_block],
                    counts,
                )
                canvas = commit_tokens(canvas, selected_positions, selected_tokens)

        return canvas
