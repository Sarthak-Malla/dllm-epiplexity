"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --load_in_4bit True --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir .logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    lora: bool = True


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/LLaDA-8B-Base/epiplexity-oracle-upper-bound"
    group_by_length: bool = True
    num_train_epochs: float = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    report_to: str = "wandb"
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10


import subprocess

# class HumanEvalCallback(transformers.TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % args.save_steps == 0 and state.global_step > 0:
#             if accelerate.PartialState().is_main_process:
#                 try:
#                     checkpoint_dir = os.path.abspath(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))
#                     merged_dir = checkpoint_dir + "-merged"
                    
#                     logger.info(f"Merging adapter {checkpoint_dir} into {merged_dir}...")
#                     subprocess.run([
#                         "python", "dllm/tools/merge_peft_adapter.py",
#                         "--adapter_model_name_or_path", checkpoint_dir,
#                         "--output_model_name_or_path", merged_dir,
#                         "--dtype", "bf16"
#                     ], check=True)
                    
#                     logger.info(f"Running HumanEval on {merged_dir}...")
                    
#                     cmd = [
#                         "accelerate", "launch", "--num_processes", "1",
#                         "dllm/pipelines/llada/eval.py",
#                         "--tasks", "humaneval",
#                         "--num_fewshot", "0",
#                         "--model", "llada",
#                         "--model_args", f"pretrained={merged_dir},max_new_tokens=1024,steps=1024,block_size=1024,cfg_scale=0.0",
#                         "--confirm_run_unsafe_code"
#                     ]
                    
#                     subprocess.run(cmd, check=True)
#                     logger.info(f"Cleanup: Removing {merged_dir}...")
#                     import shutil
#                     shutil.rmtree(merged_dir)
#                 except Exception as e:
#                     logger.error(f"Failed to run HumanEval: {e}")

def train():
    os.environ["WANDB_PROJECT"] = "dllm-epiplexity"
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"],
            text=True
        )
        logger.info(f"GPU Status before initialization:\n{smi_output}")
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")

    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
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
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(  # padded <eos_token> should be visible
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,  # finetune on padded <eos_token>
                ),
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
