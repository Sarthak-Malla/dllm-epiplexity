"""Synchronized two-model sampling for TSE."""

import math

import torch
import torch.nn.functional as F

from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.pipelines.tse.divergence import agreement_factor, jensen_shannon_divergence
from dllm.pipelines.tse.fusion import fuse_probabilities, logits_to_probabilities
from dllm.pipelines.tse.scoring import consensus_scores, fused_confidence
from dllm.pipelines.tse.selection import commit_tokens, select_positions
from dllm.pipelines.tse.weighting import (
    online_entropy_weights,
    per_token_margin_weights,
    static_weights,
)


class TSESampler:
    """Run paired forwards on one shared diffusion canvas.

    Baseline mode commits tokens using one selected model. TSE mode uses
    probability fusion and consensus-based selection.
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        tokenizer,
        model_a_device: str,
        model_b_device: str,
    ):
        self.model_a = model_a.to(model_a_device).eval()
        self.model_b = model_b.to(model_b_device).eval()
        self.tokenizer = tokenizer
        self.model_a_device = torch.device(model_a_device)
        self.model_b_device = torch.device(model_b_device)
        self.scheduler = LinearAlphaScheduler()
        self.last_paired_logits = []

    @torch.no_grad()
    def forward_pair(
        self,
        canvas: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward the same canvas through both models."""
        canvas_a = canvas.to(self.model_a_device)
        mask_a = attention_mask.to(self.model_a_device)
        canvas_b = canvas.to(self.model_b_device)
        mask_b = attention_mask.to(self.model_b_device)

        logits_a = self.model_a(canvas_a, attention_mask=mask_a).logits
        logits_b = self.model_b(canvas_b, attention_mask=mask_b).logits

        if logits_a.shape != logits_b.shape:
            raise ValueError(
                "TSE models produced incompatible logits shapes: "
                f"{tuple(logits_a.shape)} != {tuple(logits_b.shape)}"
            )

        return logits_a, logits_b

    @torch.no_grad()
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
        """Generate with paired forwards and baseline or TSE selection."""
        if not inputs:
            raise ValueError("TSESampler.sample requires at least one input")
        if steps < 1 or block_size < 1 or max_new_tokens < 1:
            raise ValueError("steps, block_size, and max_new_tokens must be positive")
        if baseline_model not in {"a", "b"}:
            raise ValueError("baseline_model must be 'a' or 'b'")
        if selection_mode not in {"baseline", "tse"}:
            raise ValueError("selection_mode must be 'baseline' or 'tse'")
        if weighting_mode not in {"static", "online_entropy", "per_token_margin"}:
            raise ValueError(
                "weighting_mode must be 'static', 'online_entropy', "
                "or 'per_token_margin'"
            )

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id
        prompts = [
            torch.as_tensor(prompt, dtype=torch.long, device=self.model_a_device)
            if isinstance(prompt, list)
            else prompt.to(self.model_a_device)
            for prompt in inputs
        ]
        prompt_lens = [prompt.shape[0] for prompt in prompts]
        max_length = max(prompt_lens) + max_new_tokens
        batch_size = len(prompts)

        canvas = torch.full(
            (batch_size, max_length),
            eos_id,
            dtype=torch.long,
            device=self.model_a_device,
        )
        for index, prompt in enumerate(prompts):
            canvas[index, : prompt.shape[0]] = prompt
            canvas[index, prompt.shape[0] : prompt.shape[0] + max_new_tokens] = mask_id

        attention_mask = torch.zeros_like(canvas)
        for index, prompt_len in enumerate(prompt_lens):
            attention_mask[index, : prompt_len + max_new_tokens] = 1

        self.last_paired_logits = []
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        for block_index in range(num_blocks):
            block_mask = torch.zeros(
                (batch_size, block_size), dtype=torch.bool, device=self.model_a_device
            )
            for sample_index, prompt_len in enumerate(prompt_lens):
                start = prompt_len + block_index * block_size
                end = min(start + block_size, prompt_len + max_new_tokens)
                if start < end:
                    block_mask[sample_index, : end - start] = (
                        canvas[sample_index, start:end] == mask_id
                    )

            transfer_counts = get_num_transfer_tokens(
                block_mask,
                steps_per_block,
                self.scheduler,
                stochastic=stochastic_transfer,
            )

            for step_index in range(transfer_counts.shape[1]):
                mask_index = canvas == mask_id
                logits_a, logits_b = self.forward_pair(canvas, attention_mask)
                if capture_logits:
                    positions = mask_index.nonzero(as_tuple=False).cpu()
                    active_a = logits_a[mask_index.to(self.model_a_device)]
                    active_b = logits_b[mask_index.to(self.model_b_device)]
                    self.last_paired_logits.append(
                        (positions, active_a.detach().cpu(), active_b.detach().cpu())
                    )

                if selection_mode == "tse":
                    active_mask_a = mask_index.to(self.model_a_device)
                    active_mask_b = mask_index.to(self.model_b_device)
                    active_logits_a = logits_a[active_mask_a]
                    active_logits_b = logits_b[active_mask_b]
                    target_device = torch.device(
                        fusion_device or self.model_a_device
                    )
                    probabilities_a = logits_to_probabilities(
                        active_logits_a.to(target_device), temperature_a
                    )
                    probabilities_b = logits_to_probabilities(
                        active_logits_b.to(target_device), temperature_b
                    )
                    if weighting_mode == "static":
                        weighting = static_weights(
                            alpha, probabilities_a.shape[0], target_device
                        )
                    elif weighting_mode == "online_entropy":
                        weighting = online_entropy_weights(
                            probabilities_a,
                            probabilities_b,
                            weight_temperature,
                            epsilon,
                            normalize_entropy,
                        )
                    else:
                        weighting = per_token_margin_weights(
                            probabilities_a,
                            probabilities_b,
                            weight_temperature,
                        )
                    fused = fuse_probabilities(
                        probabilities_a, probabilities_b, weighting.weights_a
                    )
                    divergence = jensen_shannon_divergence(
                        probabilities_a,
                        probabilities_b,
                        weighting.weights_a,
                        epsilon,
                    )
                    agreement = agreement_factor(
                        divergence, weighting.weights_a, epsilon
                    )
                    confidence, predicted_tokens = fused_confidence(fused)
                    scores = consensus_scores(confidence, agreement)

                    x0 = canvas.clone()
                    confidence = torch.full_like(
                        canvas, -torch.inf, dtype=torch.float32
                    )
                    positions = mask_index.nonzero(as_tuple=False)
                    x0[positions[:, 0], positions[:, 1]] = predicted_tokens.to(
                        canvas.device
                    )
                    confidence[positions[:, 0], positions[:, 1]] = scores.to(
                        canvas.device
                    )
                else:
                    logits = (
                        logits_a
                        if baseline_model == "a"
                        else logits_b.to(self.model_a_device)
                    )
                    x0 = torch.argmax(
                        add_gumbel_noise(logits, temperature=temperature), dim=-1
                    )
                    if remasking == "low_confidence":
                        probabilities = F.softmax(logits, dim=-1)
                        confidence = torch.gather(
                            probabilities, -1, x0.unsqueeze(-1)
                        ).squeeze(-1)
                    elif remasking == "random":
                        confidence = torch.rand_like(x0, dtype=torch.float)
                    else:
                        raise ValueError(
                            f"Unsupported remasking strategy: {remasking}"
                        )

                for sample_index, prompt_len in enumerate(prompt_lens):
                    confidence[
                        sample_index, prompt_len + (block_index + 1) * block_size :
                    ] = -torch.inf

                x0 = torch.where(mask_index, x0, canvas)
                confidence = torch.where(mask_index, confidence, -torch.inf)
                active_positions = mask_index.nonzero(as_tuple=False)
                selected_positions, selected_tokens = select_positions(
                    confidence[mask_index],
                    x0[mask_index],
                    active_positions,
                    transfer_counts[:, step_index],
                )
                canvas = commit_tokens(canvas, selected_positions, selected_tokens)

        return canvas