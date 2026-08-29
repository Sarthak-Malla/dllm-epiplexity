"""
Decoding-risk epiplexity sampler.

Run via eval entrypoints, for example:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  bash /home/sarthak.malla/dllm-epiplexity/examples/epiplexity/eval_risk.slurm.sh
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class RiskReductionSamplerConfig(MDLMSamplerConfig):
    # The heuristic for generating candidate masks.
    risk_candidate_strategy: str = "mixed"


class RiskReductionSampler(MDLMSampler):
    """
    Epiplexity sampler using held-out decoding-risk reduction as the verifier.

    For each candidate reveal set U, the sampler reveals the current model
    predictions at U, runs one lookahead pass, and selects the candidate that
    most reduces 1 - max probability on the positions still masked after U.
    """

    def get_decoding_risk_per_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate decoding risk, 1 - max probability, for each position [B, T]."""
        max_probs = F.softmax(logits, dim=-1).amax(dim=-1)
        return 1.0 - max_probs

    def generate_candidate_sets(
        self,
        confidence: torch.Tensor,
        mask_idx: torch.Tensor,
        num_transfer: Union[int, List[int], torch.Tensor],
        strategy: str = "mixed",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate different sets of token indices to reveal (U).
        Returns a dictionary of boolean tensors matching mask_idx shape.
        """
        B, _ = mask_idx.shape
        num_transfer = self._normalize_num_transfer(num_transfer, B, mask_idx.device)

        if strategy == "mixed":
            # candidate_names = ["greedy", "spaced_0", "spaced_1"]
            candidate_names = ["spaced_0", "spaced_1"]
        elif strategy == "greedy":
            candidate_names = ["greedy"]
        elif strategy == "high_entropy":
            candidate_names = ["high_entropy"]
        elif strategy == "random":
            candidate_names = ["random"]
        else:
            raise ValueError(f"Unknown risk_candidate_strategy: {strategy}")

        candidates = {
            name: torch.zeros_like(mask_idx, dtype=torch.bool)
            for name in candidate_names
        }

        for b in range(B):
            valid_indices = torch.where(mask_idx[b])[0]
            num_valid = valid_indices.numel()
            k = int(num_transfer[b].item())
            k = max(0, min(k, num_valid))
            if k == 0:
                continue

            batch_conf = torch.where(mask_idx[b], confidence[b], -torch.inf)

            if strategy == "greedy":
                _, greedy_idx = torch.topk(batch_conf, k=k)
                candidates["greedy"][b, greedy_idx] = True

            if strategy == "high_entropy":
                inv_conf = torch.where(mask_idx[b], -confidence[b], -torch.inf)
                _, high_ent_idx = torch.topk(inv_conf, k=k)
                candidates["high_entropy"][b, high_ent_idx] = True

            if strategy == "mixed":
                for offset in range(2):
                    step_float = num_valid / k
                    start_offset = (offset * step_float) / 2
                    float_indices = (
                        torch.arange(k, device=valid_indices.device) * step_float
                        + start_offset
                    )
                    int_indices = float_indices.long().clamp(max=num_valid - 1)
                    selected_idx = valid_indices[int_indices]
                    candidates[f"spaced_{offset}"][b, selected_idx] = True

            if strategy == "random":
                rand_idx = valid_indices[
                    torch.randperm(num_valid, device=valid_indices.device)[:k]
                ]
                candidates["random"][b, rand_idx] = True

        return candidates

    def _normalize_num_transfer(
        self,
        num_transfer: Union[int, List[int], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize scalar or per-row transfer counts to a [B] tensor."""
        if isinstance(num_transfer, torch.Tensor):
            transfer = num_transfer.to(device=device, dtype=torch.long).flatten()
        elif isinstance(num_transfer, int):
            transfer = torch.full(
                (batch_size,), num_transfer, device=device, dtype=torch.long
            )
        else:
            transfer = torch.as_tensor(
                num_transfer, device=device, dtype=torch.long
            ).flatten()

        if transfer.numel() == 1 and batch_size > 1:
            transfer = transfer.expand(batch_size)
        if transfer.numel() != batch_size:
            raise ValueError(
                f"num_transfer must be scalar or length {batch_size}, got {transfer.numel()}"
            )
        return transfer

    def _select_best_candidate(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        mask_index: torch.Tensor,
        candidates: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        base_risk_map: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """Evaluate lookahead candidates and select the best mask per batch row."""
        B = x.shape[0]
        best_risk_reduction = torch.full((B,), -torch.inf, device=x.device)
        best_c_idx = torch.zeros_like(mask_index, dtype=torch.bool)
        best_candidate_names = [""] * B
        risk_reductions = {}

        for c_name, c_idx in candidates.items():
            c_lookahead = x.clone()
            c_lookahead[c_idx] = x0[c_idx]
            heldout_mask = mask_index & (~c_idx)

            la_logits = self.model(c_lookahead, attention_mask=attention_mask).logits
            la_risk = (
                self.get_decoding_risk_per_token(la_logits) * heldout_mask
            ).sum(dim=-1)
            base_risk = (base_risk_map * heldout_mask).sum(dim=-1)

            risk_reduction = base_risk - la_risk
            risk_reductions[c_name] = risk_reduction

            improved = risk_reduction > best_risk_reduction
            best_risk_reduction = torch.where(
                improved, risk_reduction, best_risk_reduction
            )
            best_c_idx = torch.where(improved.unsqueeze(-1), c_idx, best_c_idx)

            for batch_idx in range(B):
                if improved[batch_idx]:
                    best_candidate_names[batch_idx] = c_name

        return best_c_idx, risk_reductions, best_candidate_names

    @torch.no_grad()
    def sample(
        self,
        inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[RiskReductionSamplerConfig] = None,
        **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        if config is None:
            config = RiskReductionSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        risk_candidate_strategy = kwargs.get(
            "risk_candidate_strategy", config.risk_candidate_strategy
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p
                for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if cfg_keep_tokens and len(cfg_keep_tokens) > 0:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        selected_candidates = [[] for _ in range(B)]

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            block_span_mask = torch.zeros_like(x, dtype=torch.bool)

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    block_mask_index[j, : end - start] = x[j, start:end] == mask_id
                    block_span_mask[j, start:end] = True

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                num_transfer = num_transfer_tokens[:, i]
                if torch.all(num_transfer == 0):
                    continue

                mask_index = x == mask_id
                current_block_mask = mask_index & block_span_mask

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if suppress_tokens and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                confidence = torch.where(current_block_mask, x0_p, -float("inf"))

                candidates = self.generate_candidate_sets(
                    confidence,
                    current_block_mask,
                    num_transfer,
                    strategy=risk_candidate_strategy,
                )

                base_risk_map = self.get_decoding_risk_per_token(logits)
                best_c_idx, _, best_candidate_names = self._select_best_candidate(
                    x=x,
                    x0=x0,
                    mask_index=mask_index,
                    candidates=candidates,
                    attention_mask=attention_mask,
                    base_risk_map=base_risk_map,
                )

                for b_idx in range(B):
                    if best_candidate_names[b_idx]:
                        if best_candidate_names[b_idx] not in selected_candidates[b_idx]:
                            selected_candidates[b_idx].append(best_candidate_names[b_idx])

                x[best_c_idx] = x0[best_c_idx]

                if return_dict:
                    histories.append(x.clone())

        if return_dict:
            return BaseSamplerOutput(
                sequences=x,
                histories=histories,
                selected_candidates=selected_candidates,
            )
        return x
