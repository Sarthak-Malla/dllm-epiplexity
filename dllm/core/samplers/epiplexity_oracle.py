import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

@dataclass
class OracleEpiplexitySamplerConfig(MDLMSamplerConfig):
    # Number of candidates to evaluate at each step
    num_oracle_candidates: int = 4
    # The heuristic for generating candidate masks
    # "mixed" generates top-entropy, top-confidence, and spaced anchors
    oracle_candidate_strategy: str = "mixed"

class OracleEpiplexitySampler(MDLMSampler):
    """
    Epiplexity Oracle Sampler.
    Uses a 1-step lookahead to evaluate candidate sets of unmasked tokens based on their "Structure Gain".
    Structure Gain proxy = Expected baseline future entropy - Lookahead future entropy.
    """
    
    def get_entropy_per_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate predictive entropy for each position [B, T]."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1)
        
    def generate_candidate_sets(
        self, 
        confidence: torch.Tensor, 
        mask_idx: torch.Tensor, 
        num_transfer: int, 
        strategy: str = "mixed"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate different sets of token indices to reveal (U).
        Returns a dictionary of boolean tensors matching mask_idx shape.
        """
        B, T = mask_idx.shape
        candidates = {}
        
        for b in range(B):
            batch_conf = confidence[b]
            
            if strategy == "mixed" or strategy == "greedy":
                # Candidate: Greedy (Highest confidence)
                _, greedy_idx = torch.topk(batch_conf, k=num_transfer)
                greedy_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
                greedy_mask[greedy_idx] = True
                candidates['greedy'] = greedy_mask.unsqueeze(0)
            
            if strategy == "mixed" or strategy == "high_entropy":
                # Candidate: High Entropy (Lowest confidence)
                inv_conf = torch.where(mask_idx[b], -batch_conf, -torch.inf) 
                _, high_ent_idx = torch.topk(inv_conf, k=num_transfer)
                high_ent_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
                high_ent_mask[high_ent_idx] = True
                candidates['high_entropy'] = high_ent_mask.unsqueeze(0)

            if strategy == "mixed":
                # Candidates: Deterministic Spaced Anchors
                valid_indices = torch.where(mask_idx[b])[0]
                step = max(1, len(valid_indices) // num_transfer)
                
                # generate up to 2 different spread patterns
                for offset in range(min(step, 2)):
                    spaced_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
                    selected_idx = valid_indices[offset::step][:num_transfer]
                    
                    if len(selected_idx) < num_transfer:
                        needed = num_transfer - len(selected_idx)
                        mask_not_selected = torch.ones_like(valid_indices, dtype=torch.bool)
                        mask_not_selected[offset::step][:num_transfer] = False
                        extra_idx = valid_indices[mask_not_selected][:needed]
                        selected_idx = torch.cat([selected_idx, extra_idx])
                        
                    spaced_mask[selected_idx] = True
                    candidates[f'spaced_{offset}'] = spaced_mask.unsqueeze(0)

            if strategy == "random":
                # Baseline for structure gain definition
                valid_indices = torch.where(mask_idx[b])[0]
                rand_idx = valid_indices[torch.randperm(len(valid_indices))[:num_transfer]]
                rand_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
                rand_mask[rand_idx] = True
                candidates['random'] = rand_mask.unsqueeze(0)

        # Extend dictionary to batched structure (assuming B=1 for simplicity, 
        # but correctly stacking for larger batches requires list of dicts transpose)
        # Assuming homogeneous candidate names across batch elements (B=1 commonly in our loop)
        return candidates

    @torch.no_grad()
    def sample(
        self,
        inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[OracleEpiplexitySamplerConfig] = None,
        **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        if config is None:
            config = OracleEpiplexitySamplerConfig()

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

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    block_mask_index[j, :end - start] = (x[j, start:end] == mask_id)

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                num_transfer = num_transfer_tokens[:, i].item() # Assuming B=1 or homogenous transfers
                if num_transfer == 0: continue
                
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask.repeat(2, 1)).logits
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
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                # --- ORACLE LOOKAHEAD LOGIC ---
                candidates = self.generate_candidate_sets(
                    confidence, mask_index, num_transfer, strategy=oracle_candidate_strategy
                )
                
                base_entropy_map = self.get_entropy_per_token(logits)
                best_structure_gain = -float('inf')
                best_c_idx = None
                best_la_ent = None
                
                random_candidate_idx = candidates.get('random', list(candidates.values())[0])

                for c_name, c_idx in candidates.items():
                    c_lookahead = x.clone()
                    c_lookahead[c_idx] = x0[c_idx]
                    c_new_mask = mask_index & (~c_idx)
                    
                    # 1-step lookahead forward pass
                    la_logits = self.model(c_lookahead, attention_mask=attention_mask).logits
                    
                    la_ent_c = (self.get_entropy_per_token(la_logits) * c_new_mask).sum(dim=-1).item()
                    base_ent_c = (base_entropy_map * c_new_mask).sum(dim=-1).item()
                    
                    entropy_drop = base_ent_c - la_ent_c
                    # For a true comparison, we proxy structure gain as the absolute entropy drop 
                    # from the set, normalized by how much random would drop. 
                    # (Here we simplify just tracking absolute drop since all candidates have |U| tokens).
                    structure_gain = entropy_drop
                        
                    if structure_gain > best_structure_gain:
                        best_structure_gain = structure_gain
                        best_c_idx = c_idx
                        best_la_ent = la_ent_c
                
                # Apply the best candidate
                x[best_c_idx] = x0[best_c_idx]

                if return_dict:
                    histories.append(x.clone())

        if return_dict:
            return BaseSamplerOutput(
                sequences=x,
                histories=histories,
            )
        return x