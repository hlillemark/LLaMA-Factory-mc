# Example configuration using TRL for DPO
from datasets import load_dataset
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model_ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure for distributed training on 8 GPUs
trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_length=8192,  # 8K context
    gradient_accumulation_steps=4,  # Adjust based on memory
    gradient_checkpointing=True,
    bf16=True,  # Mixed precision
    deepspeed="ds_config.json",  # Configure DeepSpeed
)

trainer.train()