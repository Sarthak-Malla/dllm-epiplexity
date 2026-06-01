import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

from dllm.utils import get_model, get_tokenizer
from dllm.core.samplers.epiplexity_oracle import OracleEpiplexitySampler, OracleEpiplexitySamplerConfig
from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

class DataCollectingOracleSampler(OracleEpiplexitySampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_data = []

    @torch.no_grad()
    def sample(self, inputs, config=None, **kwargs):
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
                num_transfer = num_transfer_tokens[:, i].item()
                if num_transfer == 0: continue
                
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    out = self.model(x_, attention_mask=attention_mask.repeat(2, 1), output_hidden_states=True)
                    logits = out.logits
                    hidden_states = out.hidden_states[-1]
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    hidden_states = hidden_states[:B]
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
                
                step_structure_gains = {}

                for c_name, c_idx in candidates.items():
                    c_lookahead = x.clone()
                    c_lookahead[c_idx] = x0[c_idx]
                    c_new_mask = mask_index & (~c_idx)
                    
                    # 1-step lookahead forward pass
                    la_logits = self.model(c_lookahead, attention_mask=attention_mask).logits
                    
                    la_ent_c = (self.get_entropy_per_token(la_logits) * c_new_mask).sum(dim=-1).item()
                    base_ent_c = (base_entropy_map * c_new_mask).sum(dim=-1).item()
                    
                    structure_gain = base_ent_c - la_ent_c
                    step_structure_gains[c_name] = structure_gain
                        
                    if structure_gain > best_structure_gain:
                        best_structure_gain = structure_gain
                        best_c_idx = c_idx
                
                # Record details of this diffusion step
                self.collected_data.append({
                    "hidden_states": hidden_states.cpu(),
                    "mask_positions": mask_index.cpu(),
                    "structure_gain_scores": step_structure_gains
                })
                
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

def main():
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model_name_or_path=model_name)
    print("Loading model...")
    model_config = {
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": "cuda",
        "torch_dtype": "bfloat16"
    }
    model = get_model(**model_config)
    model.eval()

    # Configure our collecting sampler
    sampler = DataCollectingOracleSampler(model=model, tokenizer=tokenizer)
    sampler_config = OracleEpiplexitySamplerConfig(
        steps=12,
        max_new_tokens=96,
        oracle_candidate_strategy="mixed"
    )

    # Load GSM8K test set
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # Take a small subset for demonstration
    subset_size = 5
    ds_subset = ds.select(range(subset_size))

    out_file = "epiplexity_data.pt"
    print(f"Generating responses and collecting data for {subset_size} samples...")

    all_data = []

    for i in tqdm(range(subset_size)):
        prompt = ds_subset[i]["question"]
        msgs = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids.to(model.device)
        
        # We process single batch but sampler can handle batches. We extract prompt list
        inputs = [input_ids[0]]

        sampler.collected_data = [] # reset per sequence or keep accumulating?
        # we can keep accumulating or attach prompt ID. Let's append to all_data.

        _ = sampler.sample(
            inputs, 
            config=sampler_config
        )

        all_data.append({
            "prompt_idx": i,
            "question": prompt,
            "step_data": sampler.collected_data
        })

    torch.save(all_data, out_file)
    print(f"Saved collected data to {out_file}")

if __name__ == "__main__":
    main()