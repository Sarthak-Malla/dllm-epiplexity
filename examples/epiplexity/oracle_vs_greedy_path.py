import os
import torch
import torch.nn.functional as F
from dllm.utils import get_model, get_tokenizer
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
import re

TEST_PROMPTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "42"),
    ("A farmer has 120 sheep. If he sells 40 of them and then buys 25 more, how many sheep does he have left?", "105")
]

def verify_ans(text, expected):
    """Simple heuristic to check if the expected number is in the text."""
    # Find all numbers in the text
    numbers = re.findall(r'\b\d+\b', text)
    # Check if the expected answer is the last number produced, or just exists in the output
    if not numbers: return 0.0
    return 1.0 if expected in numbers[-3:] else 0.0

def get_entropy_per_token(logits):
    """
    Calculate predictive entropy for each position.
    logits: [B, T, V]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1) # [B, T]

def generate_candidate_sets(confidence, mask_idx, mask_token_id, num_transfer):
    """
    Generate different sets of token indices to reveal (U).
    Returns a dictionary of boolean tensors (transfer indices).
    """
    B, T = mask_idx.shape
    candidates = {}
    
    for b in range(B):
        batch_conf = confidence[b]
        
        # Choice 1: Greedy (Highest confidence)
        _, greedy_idx = torch.topk(batch_conf, k=num_transfer)
        greedy_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
        greedy_mask[greedy_idx] = True
        candidates['greedy'] = greedy_mask.unsqueeze(0)
        
        # Choice 2: High Entropy (Lowest confidence)
        inv_conf = torch.where(mask_idx[b], -batch_conf, -torch.inf) 
        _, high_ent_idx = torch.topk(inv_conf, k=num_transfer)
        high_ent_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
        high_ent_mask[high_ent_idx] = True
        candidates['high_entropy'] = high_ent_mask.unsqueeze(0)

        # Choices 3-6: Deterministic Spaced Anchors
        valid_indices = torch.where(mask_idx[b])[0]
        step = max(1, len(valid_indices) // num_transfer)
        
        for offset in range(min(step, 4)): # Generate up to 4 different spread patterns
            spaced_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
            selected_idx = valid_indices[offset::step][:num_transfer]
            
            # Pad if offset pushes us slightly under the budget
            if len(selected_idx) < num_transfer:
                needed = num_transfer - len(selected_idx)
                mask_not_selected = torch.ones_like(valid_indices, dtype=torch.bool)
                mask_not_selected[offset::step][:num_transfer] = False
                extra_idx = valid_indices[mask_not_selected][:needed]
                selected_idx = torch.cat([selected_idx, extra_idx])
                
            spaced_mask[selected_idx] = True
            candidates[f'spaced_{offset}'] = spaced_mask.unsqueeze(0)

    # Simplified to single batch size for now
    return candidates

def test_full_path_oracle():
    print("Loading LLaDA Model...")
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = get_tokenizer(model_name_or_path=model_name)
    model_config = {
        "model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": "cuda",
        "torch_dtype": "bfloat16"
    }
    model = get_model(**model_config)
    model.eval()

    greedy_score_total = 0
    oracle_score_total = 0

    for idx, (prompt, expected_answer) in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"PROMPT {idx+1}/{len(TEST_PROMPTS)}:")
        print(f"{prompt}")
        print(f"{'='*60}")

        msgs = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids.to(model.device)
        
        B = input_ids.shape[0]
        max_new_tokens = 96
        steps = 12
        mask_ids = torch.full((B, max_new_tokens), tokenizer.mask_token_id, dtype=torch.long, device=model.device)
        initial_canvas = torch.cat([input_ids, mask_ids], dim=1)
        attention_mask = torch.ones_like(initial_canvas, dtype=torch.bool)
        
        def decode_path(canvas, strategy="greedy"):
            curr_canvas = canvas.clone()
            for step in range(steps):
                is_mask_region = (curr_canvas == tokenizer.mask_token_id)
                masks_left = is_mask_region.sum().item()
                if masks_left == 0:
                    break
                    
                num_transfer = max(1, masks_left // (steps - step))
                
                with torch.no_grad():
                    out = model(curr_canvas, attention_mask=attention_mask)
                    probs = F.softmax(out.logits, dim=-1)
                    x0 = torch.argmax(probs, dim=-1)
                    
                    x0_p = torch.squeeze(torch.gather(probs, -1, x0.unsqueeze(-1)), -1)
                    confidence = torch.where(is_mask_region, x0_p, -float('inf'))
                    
                    candidates = generate_candidate_sets(confidence, is_mask_region, tokenizer.mask_token_id, num_transfer)
                    
                    transfer_index = None
                    if strategy in candidates:
                        transfer_index = candidates[strategy]
                    elif strategy == "oracle":
                        base_entropy_map = get_entropy_per_token(out.logits)
                        best_score = -999.0

                        for c_name, c_idx in candidates.items():
                            c_lookahead = curr_canvas.clone()
                            # We ONLY commit the model's own (flawed) prediction x0!
                            c_lookahead[c_idx] = x0[c_idx]
                            c_new_mask = is_mask_region & (~c_idx)
                            
                            with torch.no_grad():
                                la_out_c = model(c_lookahead, attention_mask=attention_mask)
                                la_ent_c = (get_entropy_per_token(la_out_c.logits) * c_new_mask).sum(dim=-1).item()
                                base_ent_c = (base_entropy_map * c_new_mask).sum(dim=-1).item()
                                
                                # Absolute entropy drop on the exact remaining tokens
                                c_drop = base_ent_c - la_ent_c
                                
                            if c_drop > best_score:
                                best_score = c_drop
                                transfer_index = c_idx
                                
                    else:
                        transfer_index = candidates['greedy']
                        
                    curr_canvas[transfer_index] = x0[transfer_index]
                    
            return curr_canvas

        # Run strategies for this prompt
        greedy_final_canvas = decode_path(initial_canvas, strategy="greedy")
        greedy_ans = tokenizer.decode(greedy_final_canvas[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        oracle_final_canvas = decode_path(initial_canvas, strategy="oracle")
        oracle_ans = tokenizer.decode(oracle_final_canvas[0, input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n[GREEDY] Output:\n{greedy_ans}")
        print(f"\n[ORACLE] Output:\n{oracle_ans}")

        score_g = verify_ans(greedy_ans, expected_answer)
        score_o = verify_ans(oracle_ans, expected_answer)
        
        greedy_score_total += score_g
        oracle_score_total += score_o
        
        print(f"\n-> Expected: {expected_answer}")
        print(f"-> Greedy Correct: {bool(score_g)}")
        print(f"-> Oracle Correct: {bool(score_o)}")
        
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: ({len(TEST_PROMPTS)} Prompts)")
    print(f"Greedy Average Score: {greedy_score_total / len(TEST_PROMPTS)}")
    print(f"Oracle Average Score: {oracle_score_total / len(TEST_PROMPTS)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_full_path_oracle()
