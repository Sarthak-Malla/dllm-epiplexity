import os
import torch
import torch.nn.functional as F
from dllm.utils import get_model, get_tokenizer
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.rl.grpo.rewards.math import correctness_reward_func

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
        
        # Choice 2: Random
        # Create a random distribution over valid masked spots
        rand_conf = torch.where(mask_idx[b], torch.rand_like(batch_conf), -torch.inf)
        _, rand_idx = torch.topk(rand_conf, k=num_transfer)
        rand_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
        rand_mask[rand_idx] = True
        candidates['random'] = rand_mask.unsqueeze(0)
        
        # Choice 3: High Entropy (Lowest confidence)
        inv_conf = torch.where(mask_idx[b], -batch_conf, -torch.inf) 
        _, high_ent_idx = torch.topk(inv_conf, k=num_transfer)
        high_ent_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
        high_ent_mask[high_ent_idx] = True
        candidates['high_entropy'] = high_ent_mask.unsqueeze(0)

        # Choice 4: Spaced Anchors (Structural Skeleton)
        spaced_mask = torch.zeros_like(mask_idx[b], dtype=torch.bool)
        valid_indices = torch.where(mask_idx[b])[0]
        step = max(1, len(valid_indices) // num_transfer)
        selected_idx = valid_indices[::step][:num_transfer]
        spaced_mask[selected_idx] = True
        candidates['spaced'] = spaced_mask.unsqueeze(0)

    # Simplified to single batch size for now
    return candidates

def test_epiplexity_hypothesis():
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

    # Small GSM8k test prompt
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    expected_answer = "72" # 48 + 48/2

    msgs = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    input_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids.to(model.device)
    
    # 1. Setup Canvas (mimicking MDLMSampler)
    max_new_tokens = 64
    B = input_ids.shape[0]
    
    # Append masks
    mask_ids = torch.full((B, max_new_tokens), tokenizer.mask_token_id, dtype=torch.long, device=model.device)
    canvas = torch.cat([input_ids, mask_ids], dim=1)
    attention_mask = torch.ones_like(canvas, dtype=torch.bool)
    
    # Determine which tokens are purely masks (the target area)
    is_mask_region = (canvas == tokenizer.mask_token_id)
    
    print(f"Canvas created. Shape: {canvas.shape}, Total masks: {is_mask_region.sum().item()}")

    # 2. Base Forward Pass (State 0)
    with torch.no_grad():
        out = model(canvas, attention_mask=attention_mask)
        base_logits = out.logits
        base_entropy_map = get_entropy_per_token(base_logits)
        
        # Average entropy just for display
        base_entropy_avg = (base_entropy_map * is_mask_region).sum(dim=-1) / is_mask_region.sum(dim=-1).clamp(min=1)
        
        # Predict tokens & calculate confidence
        probs = F.softmax(base_logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.squeeze(torch.gather(probs, -1, x0.unsqueeze(-1)), -1)
        confidence = torch.where(is_mask_region, x0_p, -float('inf'))

    print(f"Initial Mask Average Entropy: {base_entropy_avg.item():.4f}")

    # 3. Generate candidate reveals for step 1 (reveal ~10% of tokens)
    num_transfer = max(1, int(0.1 * is_mask_region.sum().item()))
    candidates = generate_candidate_sets(confidence, is_mask_region, tokenizer.mask_token_id, num_transfer)

    results = []

    # 4. Lookahead eval for each candidate
    print(f"\nEvaluating candidates (Budget = {num_transfer} tokens):")
    for name, transfer_index in candidates.items():
        
        # Decode what tokens we are actually revealing!
        revealed_tokens = x0[transfer_index]
        decoded_reveals = tokenizer.decode(revealed_tokens)
        print(f"[{name.upper()}] Selected Tokens: {repr(decoded_reveals)}")

        # Create lookahead canvas
        lookahead_canvas = canvas.clone()
        # Apply the reveal: Replace masked spots where transfer_index is true with x0 predictions
        lookahead_canvas[transfer_index] = x0[transfer_index]
        
        # New mask region after reveal
        new_mask_region = is_mask_region & (~transfer_index)
        
        # We want to measure the drop on new_mask_region ONLY (Sum of entropy)
        base_ent_remaining = (base_entropy_map * new_mask_region).sum(dim=-1).item()
        
        # One-step lookahead
        with torch.no_grad():
            la_out = model(lookahead_canvas, attention_mask=attention_mask)
            la_entropy_map = get_entropy_per_token(la_out.logits)
            la_ent_remaining = (la_entropy_map * new_mask_region).sum(dim=-1).item()
            
            # What does the model think the rest of the sequence is right now?
            la_preds = torch.argmax(la_out.logits, dim=-1)
            predicted_full_state = lookahead_canvas.clone()
            predicted_full_state[new_mask_region] = la_preds[new_mask_region]
            prompt_len = input_ids.shape[1]
            decoded_full_prediction = tokenizer.decode(predicted_full_state[0, prompt_len:])
            
        print(f"[{name.upper()}] 1-Step Lookahead Prediction: {repr(decoded_full_prediction)}")
            
        # Entropy drop over EXACTLY the remaining tokens (sum difference)
        entropy_drop = base_ent_remaining - la_ent_remaining
        
        results.append({
            "name": name,
            "entropy_drop": entropy_drop,
            "canvas": lookahead_canvas,
            "new_mask_region": new_mask_region,
            "transfer_index": transfer_index
        })
        print(f" - {name.capitalize()}: Entropy Drop = {entropy_drop:.4f}")

    # Calculate Epiplexity Proxy (Structure Gain)
    # Baseline correction: using random choice as baseline
    baseline_drop = next(r['entropy_drop'] for r in results if r['name'] == 'random')
    
    best_candidate = None
    best_score = -999

    for r in results:
        struct_gain = r['entropy_drop'] - baseline_drop
        r['structure_gain'] = struct_gain
        print(f"   * {r['name'].capitalize()} Structure Gain: {struct_gain:.4f}")
        
        if struct_gain > best_score and r['name'] != 'random':
            best_score = struct_gain
            best_candidate = r

    if best_candidate['name'] == 'greedy':
        print("\nWait, Confidence Greedy had the highest Structure Gain. Hypothesis might be weak here.")
    else:
        print(f"\nHypothesis valid! {best_candidate['name'].capitalize()} showed more structure gain than Greedy.")

if __name__ == "__main__":
    test_epiplexity_hypothesis()
