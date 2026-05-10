import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig

def setup_peft_duel_adapters(model, peft_config: LoraConfig):
    """
    Wraps the base model in PEFT and initializes two separate LoRA adapters 
    for the Epiplexity duel.
    """
    model = get_peft_model(model, peft_config, adapter_name="adapter_A")
    model.add_adapter("adapter_B", peft_config)
    
    # Ensure both adapters are trainable
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            
    return model

def compute_mdlm_loss(
    model,
    inputs: dict,
    tokenizer,
    scheduler,
    time_distribution: str = "uniform",
    time_epsilon: float = 1e-3,
    loss_weight_type: str = "scheduler",
    loss_norm_type: str = "token",
    return_outputs: bool = False,
):
    """
    Custom standalone MDLM loss computation without the Hugging Face Trainer.
    Extracts logic from MDLMTrainer's compute_loss.
    """
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs.get("attention_mask", None)
    
    b, l = input_ids.shape
    maskable_mask = labels != -100  # [b, l]

    # Sample diffusion timesteps t in [epsilon, 1) based on distribution
    if time_distribution == "uniform":
        t = time_epsilon + (1 - time_epsilon) * torch.rand(b, device=input_ids.device)
    elif time_distribution == "logit_normal":
        # Sample heavily around the middle difficulties
        t = torch.randn(b, device=input_ids.device) * 1.5  # Standard deviation mapping loosely to 0-1
        t = torch.sigmoid(t)
        t = time_epsilon + (1 - time_epsilon) * t
    else:
        raise ValueError(f"Unknown time_distribution: {time_distribution}")
    
    # Masking probability p_mask = 1 - α(t)
    p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, l)
    
    # Stochastic masking
    masked_mask = (torch.rand((b, l), device=input_ids.device) < p_mask) & maskable_mask
    noised_input_ids = torch.where(
        masked_mask, tokenizer.mask_token_id, input_ids
    )

    # Forward pass
    outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Compute loss weights
    if loss_weight_type == "scheduler":
        loss_weights = scheduler.weight(t).unsqueeze(1).repeat(1, l)
    elif loss_weight_type == "uniform":
        loss_weights = torch.ones_like(input_ids)
    else:
        raise NotImplementedError("Only scheduler or uniform loss weights supported.")

    # Weighted cross-entropy
    token_nll = F.cross_entropy(
        logits.transpose(1, 2),  # [b, V, l]
        input_ids,  # [b, l]
        reduction="none",  # [b, l]
    )
    token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

    # Normalize loss
    if loss_norm_type == "token":
        token_nll /= maskable_mask.sum().clamp_min(1)
    elif loss_norm_type == "sequence":
        token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
    elif loss_norm_type == "batch":
        token_nll /= b
    else:
        raise ValueError("Invalid loss_norm_type.")
        
    loss = token_nll.sum()

    if return_outputs:
        return loss, outputs
    return loss
