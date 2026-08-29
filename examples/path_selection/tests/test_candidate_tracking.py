#!/usr/bin/env python3
"""
Quick test to verify candidate tracking works in the entropy_drop sampler.

Run from repo root:
    python test_candidate_tracking.py
"""

import torch
from dllm.core.samplers.entropy_drop import EntropyDropSampler, EntropyDropSamplerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_candidate_tracking():
    """Test that selected_candidates are returned and populated."""
    
    # Load a small model (or use a mock)
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    # Add mask token if not present (required for MDLM)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added mask token. New vocab size: {len(tokenizer)}")
        print(f"Mask token ID: {tokenizer.mask_token_id}")
    
    # Create sampler
    config = EntropyDropSamplerConfig(
        oracle_candidate_strategy="mixed",
        max_new_tokens=32,
        steps=8,
        block_size=8,
        return_dict=True,
    )
    
    sampler = EntropyDropSampler(model=model, tokenizer=tokenizer)
    
    # Prepare input
    prompt = "What is 2 + 2?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    
    print(f"Testing with prompt: {prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    # Sample with return_dict=True
    output = sampler.sample(
        inputs=[input_ids],
        config=config,
        return_dict=True,
    )
    
    # Check output type and fields
    from dllm.core.samplers.base import BaseSamplerOutput
    assert isinstance(output, BaseSamplerOutput), f"Expected BaseSamplerOutput, got {type(output)}"
    print("✓ Output is BaseSamplerOutput")
    
    # Check selected_candidates
    assert hasattr(output, 'selected_candidates'), "Missing selected_candidates attribute"
    print("✓ Output has selected_candidates attribute")
    
    assert output.selected_candidates is not None, "selected_candidates is None"
    print("✓ selected_candidates is not None")
    
    print(f"\nSelected candidates per example:")
    for i, candidates in enumerate(output.selected_candidates):
        print(f"  Example {i}: {candidates}")
    
    # Verify we got some candidates
    assert len(output.selected_candidates) > 0, "No selected_candidates"
    assert len(output.selected_candidates[0]) > 0, "No candidates selected for first example"
    print("\n✓ Candidates were tracked!")
    
    # Decode output
    decoded = tokenizer.decode(output.sequences[0])
    print(f"\nGenerated text: {decoded[:100]}...")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_candidate_tracking()
