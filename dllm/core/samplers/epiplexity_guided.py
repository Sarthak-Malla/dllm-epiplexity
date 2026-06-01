import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Union

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

@dataclass
class GuidedEpiplexitySamplerConfig(MDLMSamplerConfig):
    # If standard 'remasking' string is set to "guided", we follow the guide model scores
    # Temperature to scale the guide scores when softmaxing or evaluating
    guide_temperature: float = 1.0

class GuidedEpiplexitySampler(MDLMSampler):
    """
    Epiplexity Guided Sampler.
    Uses a trained guide network (`EpiplexityGuide`) to score candidate unmask positions efficiently 
    instead of using expensive lookahead. The guide provides structural gain scores.
    """
    def __init__(self, model, tokenizer, scheduler=None, guide_model=None):
        super().__init__(model, tokenizer, scheduler)
        self.guide_model = guide_model

    @torch.no_grad()
    def sample(
        self,
        inputs: List[Union[torch.Tensor, List[int]]],
        config: Optional[GuidedEpiplexitySamplerConfig] = None,
        **kwargs,
    ) -> Union[BaseSamplerOutput, torch.Tensor]:
        if config is None:
            config = GuidedEpiplexitySamplerConfig()

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
        remasking = kwargs.get("remasking", config.remasking)
        guide_temperature = kwargs.get("guide_temperature", config.guide_temperature)

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
                num_transfer = num_transfer_tokens[:, i].item()
                if num_transfer == 0: continue
                
                mask_index = x == mask_id

                # Forward pass to get both logits and hidden states (we need hidden states for the guide)
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    out = self.model(x_, attention_mask=attention_mask.repeat(2, 1), output_hidden_states=True)
                    logits = out.logits
                    hidden_states = out.hidden_states[-1][:B] # take base prompt's hidden states
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    out = self.model(x, attention_mask=attention_mask, output_hidden_states=True)
                    logits = out.logits
                    hidden_states = out.hidden_states[-1]

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

                if remasking == "guided" and self.guide_model is not None:
                    # -- Fast Epiplexity Guided Unmasking --
                    # Use the Guide model to score positions
                    guide_scores = self.guide_model(hidden_states)
                    
                    # Ensure non-mask regions have -inf score
                    guide_scores = torch.where(mask_index, guide_scores / guide_temperature, torch.tensor(-float('inf'), device=logits.device))
                    
                    # Top-k selection based solely on the guide's score
                    _, transfer_index = torch.topk(guide_scores, k=num_transfer, dim=-1)
                else:
                    # Default MDLM confidence fallback
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    confidence = torch.where(mask_index, x0_p, -float('inf'))
                    _, transfer_index = torch.topk(confidence, k=num_transfer, dim=-1)

                # Commit chosen predictions for this diffusion step
                for j in range(B):
                    idx = transfer_index[j]
                    x[j, idx] = x0[j, idx]

                if return_dict:
                    histories.append(x.clone())

        if return_dict:
            return BaseSamplerOutput(
                sequences=x,
                histories=histories,
            )
        return x