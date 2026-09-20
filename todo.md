# Document procedures

### Ensemble Decoding
- Prepare different samplers to be used for ensemble decoding. For now, we will use the easy samplers:
    - Top-k sampling (MDLMSampler has this as "low_confidence")
    - Minimum Entropy (modify MDLMSampler to have this as "min_entropy")
    - Maximum Top2 Probability (modify MDLMSampler to have this as "max_top2_prob")