"""
This is the MDLMSampler with Entropy drop to decide on the path selection.

Run via eval entrypoint, for example:
    source ~/.bashrc && conda activate dllm
    bash /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/eval_sampler.sh --sampler_type oracle 
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

@dataclass
class EntropyDropSamplerConfig(MDLMSamplerConfig):
    """
    Configuration for the EntropyDropSampler.
    """
    # The heuristic for generating candidate masks
    # "mixed" generates top-entropy and spaced anchors
    oracle_candidate_strategy: str = "mixed"


class EntropyDropSampler(MDLMSampler):
    """
    Epiplexity Oracle Sampler with Entropy Drop for path selection.

    Uses a 1-step lookahead to evaluate candidate sets of unmasked tokens based on
    their "Structure Gain" quantified by the entropy drop in the model's predictions.
    Structure Gain proxy = Expected baseline future entropy - Lookahead future entropy

    This uses Entropy Drop as the epiplexity proxy, which is not quite the proxy we want.
    """

    def get_entropy_per_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy per token given the logits.

        Args:
            logits (torch.Tensor): The logits from the model of shape (batch_size, seq_len, vocab_size).

        Returns:
            torch.Tensor: The entropy per token of shape (batch_size, seq_len) => [B, T].
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)
        return entropy
    

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
        B, T = mask_idx.shape
        num_transfer = self._normalize_num_transfer(num_transfer, B, mask_idx.device)

        if strategy == "mixed":
            candidate_names = ["spaced_0", "spaced_1", "random", "high_entropy"]
        elif strategy == "greedy":
            candidate_names = ["greedy"]
        elif strategy == "high_entropy":
            candidate_names = ["high_entropy"]
        elif strategy == "random":
            candidate_names = ["random"]
        else:
            raise ValueError(f"Unknown oracle_candidate_strategy: {strategy}")

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
            
            # if strategy == "mixed" or strategy == "greedy":
            if strategy == "greedy":
                # Candidate: Greedy (Highest confidence)
                _, greedy_idx = torch.topk(batch_conf, k=k)
                candidates["greedy"][b, greedy_idx] = True
            
            if strategy == "mixed" or strategy == "high_entropy":
            # if strategy == "high_entropy":
                # Candidate: High Entropy (Lowest confidence)
                inv_conf = torch.where(mask_idx[b], -confidence[b], -torch.inf)
                _, high_ent_idx = torch.topk(inv_conf, k=k)
                candidates["high_entropy"][b, high_ent_idx] = True

            if strategy == "mixed":
                # Candidates: Deterministic Spaced Anchors
                for offset in range(2):
                    # Create evenly spaced floating point indices across the span
                    # Shift the second pattern by half a step size
                    step_float = num_valid / k
                    start_offset = (offset * step_float) / 2
                    
                    # Compute integer indices
                    float_indices = torch.arange(k, device=valid_indices.device) * step_float + start_offset
                    int_indices = float_indices.long().clamp(max=num_valid - 1)
                    
                    selected_idx = valid_indices[int_indices]
                    candidates[f"spaced_{offset}"][b, selected_idx] = True

            if strategy == "mixed" or strategy == "random":
                # Baseline for structure gain definition
                rand_idx = valid_indices[torch.randperm(num_valid, device=valid_indices.device)[:k]]
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
        base_entropy_map: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """
        Select the best candidate set based on entropy drop.

        This is where the "epiplexity" heuristic is applied. The candidate set that results in the largest drop in entropy (i.e., the most confident predictions) is selected.
        
        Returns:
            - best_c_idx: boolean mask of best candidate positions
            - structure_gains: dict of entropy drops per candidate
            - best_candidate_names: list of best candidate name per batch element
        """
        """Evaluate lookahead candidates and select the best mask per batch row."""
        B = x.shape[0]
        best_structure_gain = torch.full((B,), -torch.inf, device=x.device)
        best_c_idx = torch.zeros_like(mask_index, dtype=torch.bool)
        best_candidate_names = [""] * B  # Track which candidate was best for each batch element
        structure_gains = {}

        for c_name, c_idx in candidates.items():
            c_lookahead = x.clone()
            c_lookahead[c_idx] = x0[c_idx]
            c_new_mask = mask_index & (~c_idx)

            # 1-step lookahead forward pass
            la_logits = self.model(c_lookahead, attention_mask=attention_mask).logits

            la_ent_c = (self.get_entropy_per_token(la_logits) * c_new_mask).sum(dim=-1)
            base_ent_c = (base_entropy_map * c_new_mask).sum(dim=-1)

            entropy_drop = base_ent_c - la_ent_c
            structure_gains[c_name] = entropy_drop

            improved = entropy_drop > best_structure_gain
            best_structure_gain = torch.where(improved, entropy_drop, best_structure_gain)
            best_c_idx = torch.where(improved.unsqueeze(-1), c_idx, best_c_idx)
            # Update best candidate names
            for batch_idx in range(B):
                if improved[batch_idx]:
                    best_candidate_names[batch_idx] = c_name

        return best_c_idx, structure_gains, best_candidate_names

    
    @torch.no_grad()
    def sample(
        self,
        inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[EntropyDropSamplerConfig] = None,
        **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        if config is None:
            config = EntropyDropSamplerConfig()
        
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        
        oracle_candidate_strategy = kwargs.get("oracle_candidate_strategy", config.oracle_candidate_strategy)

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [[bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs]
        
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, :prompt_lens[i]] = p
            x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = mask_id
        
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if cfg_keep_tokens and len(cfg_keep_tokens) > 0:
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        selected_candidates = [[] for _ in range(B)]  # Track candidates per example

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            block_span_mask = torch.zeros_like(x, dtype=torch.bool)

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    ) # which positions in this block are still masked
                    block_span_mask[j, start:end] = True

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                # Get the current number of tokens to transfer for each sample
                num_transfer = num_transfer_tokens[:, i]
                if torch.all(num_transfer == 0):
                    continue

                mask_index = x == mask_id # current global mask map
                current_block_mask = mask_index & block_span_mask # current mask map restricted to this block

                # Model forward pass to get the logits for the current state of x
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask.repeat(2, 1)).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf
                
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf
                
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                confidence = torch.where(current_block_mask, x0_p, -np.inf)

                # ----- Generate candidate sets based on the current confidence and mask -----
                candidates = self.generate_candidate_sets(
                    confidence=confidence,
                    mask_idx=current_block_mask,
                    num_transfer=num_transfer,
                    strategy=oracle_candidate_strategy,
                )

                # ----- Evaluate candidates and select the best one based on entropy drop -----
                base_entropy_map = self.get_entropy_per_token(logits)
                best_c_idx, _, best_candidate_names = self._select_best_candidate(
                    x=x,
                    x0=x0,
                    mask_index=mask_index,
                    candidates=candidates,
                    attention_mask=attention_mask,
                    base_entropy_map=base_entropy_map,
                )
                
                # Track selected candidates per example
                for b in range(B):
                    if best_candidate_names[b]:  # Only add if non-empty
                        if best_candidate_names[b] not in selected_candidates[b]:
                            selected_candidates[b].append(best_candidate_names[b])
                
                # Apply the best candidate
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