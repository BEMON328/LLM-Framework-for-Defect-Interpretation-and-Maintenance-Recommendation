# ============================================================================
# WINDOWS TRITON WORKAROUND - Must be FIRST
# ============================================================================
import sys
import types
import os

# Mock triton BEFORE torch import
triton_mock = types.ModuleType('triton')
triton_mock.__spec__ = types.ModuleType('spec')
triton_mock.__version__ = '2.0.0'
sys.modules['triton'] = triton_mock
sys.modules['triton.language'] = types.ModuleType('triton.language')
sys.modules['triton.compiler'] = types.ModuleType('triton.compiler')

# Now import torch and other libraries
import torch
import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B parameters
HF_TOKEN = "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"

# Path to tokenized data
DATA_FILE = Path(
    r"C:\Users\Gx\Desktop\FineTune\1_DataPrep\formatted_training_data_phi3\phi3_training_data_tokenized.json")

print("=" * 80)
print("PHI-3 FINE-TUNING - WINDOWS COMPATIBLE VERSION")
print("=" * 80)

print("\nLoading pre-tokenized data...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    tokenized_data = json.load(f)

print(f"✅ Loaded {len(tokenized_data)} training examples")

# Load tokenizer first
print(f"\nLoading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    trust_remote_code=True,
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✅ Tokenizer loaded")
print(f"   Model: {BASE_MODEL}")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(tokenized_data)


# ============================================================================
# CUSTOM DATA COLLATOR
# ============================================================================

class DataCollatorForCompletionOnlyLM:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_length = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            max_length = (
                    (max_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
            )

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            padding_length = max_length - len(input_ids)

            padded_input_ids = input_ids + [self.pad_token_id] * padding_length
            padded_attention_mask = attention_mask + [0] * padding_length
            padded_labels = labels + [-100] * padding_length

            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(padded_attention_mask)
            batch["labels"].append(padded_labels)

        batch = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }

        return batch


data_collator = DataCollatorForCompletionOnlyLM(tokenizer, pad_to_multiple_of=8)

# ============================================================================
# MODEL LOADING
# ============================================================================
print(f"\nLoading model: {BASE_MODEL}")

# Try loading with different methods
loading_success = False
model = None

methods_to_try = [
    {
        "name": "FP16 with eager attention",
        "params": {
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    },
    {
        "name": "BF16 with eager attention",
        "params": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    },
    {
        "name": "Auto dtype with eager attention",
        "params": {
            "torch_dtype": "auto",
            "attn_implementation": "eager",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    }
]

for i, method in enumerate(methods_to_try):
    print(f"\n🔄 Attempt {i + 1}: {method['name']}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            **method['params']
        )
        loading_success = True
        print(f"✅ Success with {method['name']}")
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
        continue

if not loading_success or model is None:
    print(f"\n❌ All loading methods failed for {BASE_MODEL}")
    print("\n🚨 RECOMMENDED SOLUTION:")
    print("1. Use Phi-3 Mini instead (most compatible):")
    print("   BASE_MODEL = 'microsoft/Phi-3-mini-4k-instruct'")
    exit(1)

print(f"✅ Model loaded successfully!")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# LoRA configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n📊 Model parameters:")
model.print_trainable_parameters()

# ============================================================================
# TRAINING SETUP
# ============================================================================
OUTPUT_DIR = "./finetuned_phi3_windows"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=30,
    logging_dir="./logs",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
)

print(f"\n📈 Starting training with {len(dataset)} samples...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ============================================================================
# TRAIN
# ============================================================================
print("\n" + "=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)

try:
    trainer.train()
    print("\n✅ Training completed successfully!")

    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"📁 Model saved to: {OUTPUT_DIR}")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)