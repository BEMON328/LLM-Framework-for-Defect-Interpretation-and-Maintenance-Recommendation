# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model
# from pathlib import Path
#
# # Configuration
# HF_TOKEN = "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"
# TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data\llama3_training_data_simple.csv"
# OUTPUT_DIR = Path("./finetuned_model")
# # MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# # MODEL_NAME = "google/gemma-2-9b-it"
#
# print(f"Fine-tuning {MODEL_NAME}")
# print("=" * 60)
#
# # Load training data
# try:
#     df = pd.read_csv(TRAINING_DATA_PATH)
#     print(f"Loaded {len(df)} training samples from {TRAINING_DATA_PATH}")
# except FileNotFoundError:
#     print(f"Error: File {TRAINING_DATA_PATH} not found.")
#     exit(1)
# except pd.errors.EmptyDataError:
#     print("Error: The CSV file is empty.")
#     exit(1)
#
# # Validate and clean data
# assert "text" in df.columns, "CSV must have 'text' column"
# df = df.dropna(subset=["text"])
# df["text"] = df["text"].str.strip()
# print(f"After cleaning: {len(df)} valid samples")
#
# # Create dataset
# dataset = Dataset.from_pandas(df)
#
# # Load tokenizer and model
# print(f"\nLoading tokenizer and model...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         token=HF_TOKEN,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     exit(1)
#
# # Fix padding token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
#     model.resize_token_embeddings(len(tokenizer))
#     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
#     print("Added <pad> as padding token")
#
# # LoRA Configuration
# print("\nConfiguring LoRA...")
# config = LoraConfig(
#     r=64,  # Rank - higher = more capacity but slower
#     lora_alpha=64,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj", "lm_head"
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
#
# # Tokenize dataset
# def tokenize_function(examples):
#     """Tokenize text for causal language modeling"""
#     texts = examples["text"]
#     tokenized = tokenizer(
#         texts,
#         truncation=True,
#         padding="max_length",
#         max_length=512,  # Optimal for dialogue format
#         return_tensors="pt"
#     )
#     input_ids = tokenized["input_ids"]
#     return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}
#
# print("\nTokenizing dataset...")
# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#
# # Split into train and eval
# train_size = int(0.9 * len(tokenized_dataset))
# eval_size = len(tokenized_dataset) - train_size
# train_dataset = tokenized_dataset.select(range(train_size))
# eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
#
# print(f"Training samples: {train_size}")
# print(f"Evaluation samples: {eval_size}")
#
# # Training arguments
# training_args = TrainingArguments(
#     output_dir=str(OUTPUT_DIR),
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=5,  # Increased for better learning
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     eval_strategy="steps",
#     eval_steps=50,
#     fp16=True,
#     push_to_hub=False,
#     gradient_accumulation_steps=4,
#     load_best_model_at_end=True,
#     max_grad_norm=1.0,  # Gradient clipping for stability
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
# )
#
# # Initialize trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset if eval_size > 0 else None,
#     tokenizer=tokenizer,
# )
#
# # Train the model
# print("\nStarting training...")
# print("=" * 60)
# try:
#     trainer.train()
#     print("\nTraining completed successfully!")
# except Exception as e:
#     print(f"Error during training: {e}")
#     exit(1)
#
# # Save model and tokenizer
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# trainer.save_model(str(OUTPUT_DIR))
# tokenizer.save_pretrained(str(OUTPUT_DIR))
#
# print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")
# print("\nTraining Summary:")
# print(f"  - Model: {MODEL_NAME}")
# print(f"  - Training samples: {train_size}")
# print(f"  - Epochs: 5")
# print(f"  - LoRA rank: 64")
# print(f"  - Output: {OUTPUT_DIR}")
#
# print("\nNext steps:")
# print("1. Update MODEL_NAME for other architectures (Mistral, Qwen, Gemma)")
# print("2. Update TRAINING_DATA_PATH for traceable versions")
# print("3. Run inference evaluation to compare models")


# # MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# # MODEL_NAME = "google/gemma-2-9b-it"
#
# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model
# from pathlib import Path
#
# # Configuration
# HF_TOKEN = "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"
# TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data\mistral_training_data_simple.csv"
# OUTPUT_DIR = Path("./finetuned_model")
#
#
#
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#
# print(f"Fine-tuning {MODEL_NAME}")
# print("=" * 60)
#
# # Load training data
# try:
#     df = pd.read_csv(TRAINING_DATA_PATH)
#     print(f"Loaded {len(df)} training samples from {TRAINING_DATA_PATH}")
# except FileNotFoundError:
#     print(f"Error: File {TRAINING_DATA_PATH} not found.")
#     exit(1)
# except pd.errors.EmptyDataError:
#     print("Error: The CSV file is empty.")
#     exit(1)
#
# # Validate and clean data
# assert "text" in df.columns, "CSV must have 'text' column"
# df = df.dropna(subset=["text"])
# df["text"] = df["text"].str.strip()
# print(f"After cleaning: {len(df)} valid samples")
#
# # Create dataset
# dataset = Dataset.from_pandas(df)
#
# # Load tokenizer and model
# print(f"\nLoading tokenizer and model...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         token=HF_TOKEN,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True
#     )
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     exit(1)
#
# # Handle pad token - some models have it, some don't
# if tokenizer.pad_token is None:
#     print("No pad token found - using eos_token as pad_token")
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
# else:
#     print(f"Pad token already exists: {tokenizer.pad_token}")
#
# print(f"Tokenizer vocab size: {len(tokenizer)}")
# print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
#
# # DO NOT resize embeddings - keep original size
#
# # LoRA Configuration
# print("\nConfiguring LoRA...")
# config = LoraConfig(
#     r=128,  # Increased from 64 for stronger fine-tuning
#     lora_alpha=128,  # Match the rank
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"
#         # Removed "lm_head" to avoid vocab size issues
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, config)
# model.print_trainable_parameters()
#
#
# # Tokenize dataset
# def tokenize_function(examples):
#     """Tokenize text for causal language modeling"""
#     texts = examples["text"]
#     tokenized = tokenizer(
#         texts,
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#         return_tensors="pt"
#     )
#     input_ids = tokenized["input_ids"]
#     return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}
#
#
# print("\nTokenizing dataset...")
# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#
# # Split into train and eval
# train_size = int(0.9 * len(tokenized_dataset))
# eval_size = len(tokenized_dataset) - train_size
# train_dataset = tokenized_dataset.select(range(train_size))
# eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
#
# print(f"Training samples: {train_size}")
# print(f"Evaluation samples: {eval_size}")
#
#
# # Callback to save tokenizer with each checkpoint
# class SaveTokenizerCallback(TrainerCallback):
#     """Save tokenizer alongside model checkpoints"""
#
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#
#     def on_save(self, args, state, control, **kwargs):
#         """Called whenever a checkpoint is saved"""
#         checkpoint_folder = f"checkpoint-{state.global_step}"
#         output_dir = Path(args.output_dir) / checkpoint_folder
#
#         if output_dir.exists():
#             print(f"Saving tokenizer to {output_dir}")
#             self.tokenizer.save_pretrained(str(output_dir))
#
#
# # Training arguments
# training_args = TrainingArguments(
#     output_dir=str(OUTPUT_DIR),
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=5,
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     eval_strategy="steps",
#     eval_steps=50,
#     fp16=True,
#     push_to_hub=False,
#     gradient_accumulation_steps=4,
#     load_best_model_at_end=True,
#     max_grad_norm=1.0,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
# )
#
# # Initialize trainer with tokenizer callback
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset if eval_size > 0 else None,
#     tokenizer=tokenizer,
#     callbacks=[SaveTokenizerCallback(tokenizer)]
# )
#
# # Train the model
# print("\nStarting training...")
# print("=" * 60)
# try:
#     trainer.train()
#     print("\nTraining completed successfully!")
# except Exception as e:
#     print(f"Error during training: {e}")
#     exit(1)
#
# # Save final model and tokenizer
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# trainer.save_model(str(OUTPUT_DIR))
# tokenizer.save_pretrained(str(OUTPUT_DIR))
#
# print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")
# print("\nTraining Summary:")
# print(f"  - Model: {MODEL_NAME}")
# print(f"  - Training samples: {train_size}")
# print(f"  - Tokenizer vocab size: {len(tokenizer)}")
# print(f"  - Epochs: 5")
# print(f"  - LoRA rank: 64")
# print(f"  - Output: {OUTPUT_DIR}")


import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from pathlib import Path

# Configuration
HF_TOKEN = "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"
TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data_gpt_generalisation\llama3_training_data_simple.csv"
# TRAINING_DATA_PATH = r"..\GYData\question_sets\question_set_d_train.csv"

OUTPUT_DIR = Path("./finetuned_model_trainsetD_gpt")

# MODEL SELECTION - Change this to switch between models
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "google/gemma-7b-it"
# MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================
MODEL_CONFIGS = {
    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "learning_rate": 2e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 0.3,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "learning_rate": 2e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 0.3,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },

    # Gemma models - require much lower learning rates
    "google/gemma-7b-it": {
        "learning_rate": 5e-6,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "warmup_steps": 50,
        "max_grad_norm": 0.5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
    },
    "google/gemma-2-9b-it": {
        "learning_rate": 3e-6,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "warmup_steps": 50,
        "max_grad_norm": 0.5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
    },

    # Llama 3 models
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "learning_rate": 3e-5,
        "lora_r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "learning_rate": 3e-5,
        "lora_r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },
}

# Get configuration for the selected model
if MODEL_NAME not in MODEL_CONFIGS:
    print(f"Warning: No specific config for {MODEL_NAME}, using Mistral defaults")
    model_config = MODEL_CONFIGS["mistralai/Mistral-7B-Instruct-v0.3"]
else:
    model_config = MODEL_CONFIGS[MODEL_NAME]

print(f"Fine-tuning {MODEL_NAME}")
print("=" * 80)
print(f"Configuration for this model:")
for key, value in model_config.items():
    print(f"  {key}: {value}")
print("=" * 80)

# Load training data
try:
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Loaded {len(df)} training samples from {TRAINING_DATA_PATH}")
except FileNotFoundError:
    print(f"Error: File {TRAINING_DATA_PATH} not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    exit(1)

# Validate and clean data
assert "text" in df.columns, "CSV must have 'text' column"
df = df.dropna(subset=["text"])
df["text"] = df["text"].str.strip()
print(f"After cleaning: {len(df)} valid samples")

# Create dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
print(f"\nLoading tokenizer and model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Handle pad token - some models have it, some don't
if tokenizer.pad_token is None:
    print("No pad token found - using eos_token as pad_token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    print(f"Pad token already exists: {tokenizer.pad_token}")

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Enable gradient checkpointing if specified for this model (usually Gemma)
if model_config.get("gradient_checkpointing", False):
    print("Enabling gradient checkpointing for memory efficiency...")
    model.gradient_checkpointing_enable()

# LoRA Configuration - using model-specific parameters
print("\nConfiguring LoRA with model-specific parameters...")
config = LoraConfig(
    r=model_config["lora_r"],
    lora_alpha=model_config["lora_alpha"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=model_config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


# Tokenize dataset
def tokenize_function(examples):
    """Tokenize text for causal language modeling"""
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}


print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split into train and eval
train_size = int(0.9 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

print(f"Training samples: {train_size}")
print(f"Evaluation samples: {eval_size}")


# Callback to save tokenizer with each checkpoint
class SaveTokenizerCallback(TrainerCallback):
    """Save tokenizer alongside model checkpoints"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        """Called whenever a checkpoint is saved"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = Path(args.output_dir) / checkpoint_folder

        if output_dir.exists():
            print(f"Saving tokenizer to {output_dir}")
            self.tokenizer.save_pretrained(str(output_dir))


# Training arguments - using model-specific parameters
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=model_config["learning_rate"],
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    warmup_steps=model_config["warmup_steps"],
    weight_decay=model_config["weight_decay"],
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    fp16=True,
    push_to_hub=False,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    max_grad_norm=model_config["max_grad_norm"],
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    lr_scheduler_type=model_config["lr_scheduler_type"],
    logging_first_step=True,
    report_to="none",  # Disable wandb/tensorboard unless you want them
)

# Initialize trainer with tokenizer callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if eval_size > 0 else None,
    tokenizer=tokenizer,
    callbacks=[SaveTokenizerCallback(tokenizer)]
)

# Train the model
print("\nStarting training...")
print("=" * 80)
try:
    trainer.train()
    print("\nTraining completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save final model and tokenizer
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"  Model: {MODEL_NAME}")
print(f"  Training samples: {train_size}")
print(f"  Evaluation samples: {eval_size}")
print(f"  Tokenizer vocab size: {len(tokenizer)}")
print(f"  Epochs: 5")
print(f"  Learning rate: {model_config['learning_rate']}")
print(f"  LoRA rank: {model_config['lora_r']}")
print(f"  LoRA alpha: {model_config['lora_alpha']}")
print(f"  Max grad norm: {model_config['max_grad_norm']}")
print(f"  Output directory: {OUTPUT_DIR}")
print("=" * 80)