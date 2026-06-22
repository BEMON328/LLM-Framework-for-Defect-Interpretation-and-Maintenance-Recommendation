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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_MODEL = "microsoft/Phi-3-medium-4k-instruct"
# microsoft/Phi-3-small-8k-instruct
# "microsoft/Phi-3-medium-4k-instruct"
HF_TOKEN = "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"

# Path to tokenized data
DATA_FILE = Path(
    r"C:\Users\Gx\Desktop\FineTune\1_DataPrep\formatted_training_data_phi3\phi3_training_data_tokenized.json")

print("="*80)
print("PHI-3 FINE-TUNING WITH PROPER LOSS MASKING")
print("="*80)

print("\nLoading pre-tokenized data...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    tokenized_data = json.load(f)

print(f"✅ Loaded {len(tokenized_data)} training examples")

# Load tokenizer
print(f"\nLoading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✅ Tokenizer loaded")
print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(tokenized_data)
print(f"\nDataset columns: {dataset.column_names}")

# Verify a sample
sample = dataset[0]
print(f"\n📊 Sample verification:")
print(f"  Input IDs length: {len(sample['input_ids'])}")
print(f"  Labels length: {len(sample['labels'])}")
print(f"  Masked tokens: {sum(1 for l in sample['labels'] if l == -100)} ({sum(1 for l in sample['labels'] if l == -100)/len(sample['labels'])*100:.1f}%)")
print(f"  Unmasked tokens: {sum(1 for l in sample['labels'] if l != -100)} ({sum(1 for l in sample['labels'] if l != -100)/len(sample['labels'])*100:.1f}%)")


# ============================================================================
# CUSTOM DATA COLLATOR
# ============================================================================

class DataCollatorForCompletionOnlyLM:
    """
    Data collator for pre-tokenized data with proper masking
    """

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


data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)

print(f"✅ Data collator created")

# ============================================================================
# MODEL LOADING
# ============================================================================
print(f"\nLoading model: {BASE_MODEL}")
print("This may take a few minutes for 14B model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

print(f"✅ Model loaded")

# ✅ PHI-3 SPECIFIC: Check module names
print("\n🔍 Checking Phi-3 module structure...")
module_names = set()
for name, _ in model.named_modules():
    if 'proj' in name.lower() and 'lm_head' not in name.lower():
        # Extract the module type (e.g., 'qkv_proj', 'o_proj')
        parts = name.split('.')
        if len(parts) > 0:
            module_type = parts[-1]
            module_names.add(module_type)

print(f"Found projection modules: {sorted(module_names)}")

# ✅ PHI-3 SPECIFIC: Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
print("✅ Gradient checkpointing enabled")

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

# ✅ PHI-3 SPECIFIC: Optimized LoRA configuration
# Use conservative settings for 14B model
lora_config = LoraConfig(
    r=16,  # Reduced from 32 for memory
    lora_alpha=32,  # 2*r ratio
    target_modules=["qkv_proj", "o_proj"],  # Conservative: attention only
    # If you want more parameters: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n📊 Model parameters:")
model.print_trainable_parameters()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

# ✅ PHI-3 SPECIFIC: Check BF16 support
use_bf16 = torch.cuda.is_bf16_supported()
print(f"\n{'✅' if use_bf16 else '⚠️'} BF16 supported: {use_bf16}")
print(f"Using: {'BF16 (recommended for Phi-3)' if use_bf16 else 'FP16 (fallback)'}")

training_args = TrainingArguments(
    output_dir="./finetuned_phi3_with_masking",
    num_train_epochs=3,  # ✅ Reduced from 10 to avoid overfitting
    per_device_train_batch_size=1,  # ✅ Reduced from 4 for 14B model
    gradient_accumulation_steps=16,  # ✅ Increased to maintain effective batch size
    learning_rate=1e-4,  # ✅ More conservative for large model
    weight_decay=0.01,
    fp16=not use_bf16,  # ✅ Phi-3 specific
    bf16=use_bf16,      # ✅ Phi-3 specific
    logging_steps=5,
    save_steps=50,  # ✅ Adjusted for fewer total steps
    save_total_limit=3,
    warmup_steps=20,  # ✅ Adjusted proportionally
    warmup_ratio=0.1,
    logging_dir="./logs",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,  # ✅ Enable gradient checkpointing
    optim="adamw_torch",  # ✅ Stable optimizer
    lr_scheduler_type="cosine",  # ✅ Better than linear for fine-tuning
)

# Calculate training metrics
total_steps = (len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
print(f"\n📈 Training configuration:")
print(f"  Training samples: {len(dataset)}")
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Total training steps: ~{total_steps}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Learning rate: {training_args.learning_rate}")

# ============================================================================
# TRAINER
# ============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ============================================================================
# TRAIN
# ============================================================================
print("\n" + "="*80)
print("🚀 STARTING PHI-3 TRAINING WITH PROPER LOSS MASKING")
print("="*80)
print("\n💡 What to expect:")
print("  - Initial loss: ~2.0-3.0 (measuring only assistant responses)")
print("  - Target loss: <1.0 after 3 epochs")
print("  - Training time: ~2-4 hours (depending on GPU)")
print("\n" + "="*80 + "\n")

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "="*80)
print("💾 Saving final model...")
trainer.save_model("./finetuned_phi3_with_masking/final")
tokenizer.save_pretrained("./finetuned_phi3_with_masking/final")
print("✅ Training complete!")
print(f"📁 Model saved to: ./finetuned_phi3_with_masking/final")
print("="*80)