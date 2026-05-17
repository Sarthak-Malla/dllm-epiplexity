import os
import copy
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator

import dllm
from dllm.core.schedulers import LinearAlphaScheduler
from peft import LoraConfig
from epiplexity.utils import setup_peft_duel_adapters, compute_mdlm_loss

logger = dllm.utils.get_default_logger(__name__)

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    lora: bool = True

@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:9000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = True

@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/LLaDA-8B-Base-Epiplexity-Accum"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    num_train_epochs: float = 1.0


def sync_adapters(model, winner_adapter: str, loser_adapter: str):
    """
    Overwrites loser adapter weights with winner adapter weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if winner_adapter in name:
                loser_name = name.replace(winner_adapter, loser_adapter)
                loser_param = dict(model.named_parameters())[loser_name]
                loser_param.data.copy_(param.data)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    accelerator = Accelerator()
    
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = setup_peft_duel_adapters(model, peft_config)

    with accelerator.main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )

        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(map_fn, num_proc=data_args.num_proc)

        dataset = dllm.utils.post_process_dataset(dataset, data_args)

        keep_cols = {"input_ids", "labels", "attention_mask"}
        for split in dataset:
            remove_cols = [c for c in dataset[split].column_names if c not in keep_cols]
            dataset[split] = dataset[split].remove_columns(remove_cols)

    data_collator = dllm.utils.NoAttentionMaskWrapper(
        transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True, label_pad_token_id=tokenizer.pad_token_id,)
    )
    dataloader = DataLoader(
        dataset["train"], 
        batch_size=training_args.per_device_train_batch_size, 
        collate_fn=data_collator,
        shuffle=True
    )

    adapter_A_params = [p for n, p in model.named_parameters() if "adapter_A" in n]
    adapter_B_params = [p for n, p in model.named_parameters() if "adapter_B" in n]
    
    optimizer_A = torch.optim.AdamW(adapter_A_params, lr=training_args.learning_rate)
    optimizer_B = torch.optim.AdamW(adapter_B_params, lr=training_args.learning_rate)

    model, optimizer_A, optimizer_B, dataloader = accelerator.prepare(
        model, optimizer_A, optimizer_B, dataloader
    )

    scheduler_A = LinearAlphaScheduler()
    scheduler_B = LinearAlphaScheduler() 

    model.train()
    
    accum_delta_A = 0.0
    accum_delta_B = 0.0

    for step, batch in enumerate(dataloader):
        
        # --- PATH A ---
        model.set_adapter("adapter_A")
        # Step A: Forward (loss_before)
        loss_before_A = compute_mdlm_loss(model, batch, tokenizer, scheduler_A, time_distribution="uniform")
        accelerator.backward(loss_before_A)
        optimizer_A.step()
        optimizer_A.zero_grad()
        
        # Step A: Re-evaluate (loss_after)
        with torch.no_grad():
            loss_after_A = compute_mdlm_loss(model, batch, tokenizer, scheduler_A, time_distribution="uniform")
        
        delta_A = loss_before_A.item() - loss_after_A.item()
        accum_delta_A += delta_A

        # --- PATH B ---
        model.set_adapter("adapter_B")
        # Step B: Forward (loss_before)
        loss_before_B = compute_mdlm_loss(model, batch, tokenizer, scheduler_B, time_distribution="logit_normal")
        accelerator.backward(loss_before_B)
        optimizer_B.step()
        optimizer_B.zero_grad()
        
        # Step B: Re-evaluate (loss_after)
        with torch.no_grad():
            loss_after_B = compute_mdlm_loss(model, batch, tokenizer, scheduler_B, time_distribution="logit_normal")
        
        delta_B = loss_before_B.item() - loss_after_B.item()
        accum_delta_B += delta_B

        # --- WINNER SELECTION ---
        if (step + 1) % 50 == 0:
            if accum_delta_A > accum_delta_B:
                winner = "adapter_A"
                loser = "adapter_B"
                sync_adapters(model.module if hasattr(model, "module") else model, winner, loser)
            else:
                winner = "adapter_B"
                loser = "adapter_A"
                sync_adapters(model.module if hasattr(model, "module") else model, winner, loser)

            logger.info(f"Step {step}: Winner was {winner} (Accum Delta A: {accum_delta_A:.4f}, Accum Delta B: {accum_delta_B:.4f})")
            
            # Reset accumulators for next N steps
            accum_delta_A = 0.0
            accum_delta_B = 0.0

    if accelerator.is_main_process:
        accelerator.save_state(training_args.output_dir)

if __name__ == "__main__":
    train()