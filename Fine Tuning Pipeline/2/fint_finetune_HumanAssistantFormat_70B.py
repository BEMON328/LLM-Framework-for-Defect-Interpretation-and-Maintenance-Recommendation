# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, \
#     BitsAndBytesConfig
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model
# from pathlib import Path
#
# # Configuration
# # REMOVED: HF_TOKEN = "..." (Relies on environment variable or login)
# TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data_gpt_robustness\phi3_training_data_simple.csv"
# OUTPUT_DIR = Path("./finetuned_model_15B_phi_gpt")
#
# # MODEL SELECTION - Updated to 70B model
# MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"
#
# # ============================================================================
# # MODEL-SPECIFIC CONFIGURATIONS
# # ============================================================================
# MODEL_CONFIGS = {
#     # Llama 3 - 8B
#     "meta-llama/Meta-Llama-3-8B-Instruct": {
#         "learning_rate": 3e-5,
#         "lora_r": 128,
#         "lora_alpha": 256,
#         "lora_dropout": 0.05,
#         "warmup_steps": 100,
#         "max_grad_norm": 1.0,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": False,
#     },
#
#     # NEW: Llama 3 - 70B (Optimized for A6000)
#     "meta-llama/Meta-Llama-3-70B-Instruct": {
#         "learning_rate": 1e-5,  # Lower LR for a much larger model
#         "lora_r": 32,  # A smaller rank is often sufficient for 70B
#         "lora_alpha": 64,  # Keep the 1:2 ratio
#         "lora_dropout": 0.05,
#         "warmup_steps": 100,
#         "max_grad_norm": 1.0,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": True,  # CRITICAL for saving memory
#     },
#
#     # Other models kept for reference...
#     "mistralai/Mistral-7B-Instruct-v0.3": {
#         "learning_rate": 2e-5,
#         "lora_r": 64,
#         "lora_alpha": 128,
#         "lora_dropout": 0.05,
#         "warmup_steps": 100,
#         "max_grad_norm": 0.3,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": False,
#     },
#     "google/gemma-2-9b-it": {
#         "learning_rate": 3e-6,
#         "lora_r": 32,
#         "lora_alpha": 64,
#         "lora_dropout": 0.1,
#         "warmup_steps": 50,
#         "max_grad_norm": 0.5,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": True,
#     },
# }
#
# # Get configuration for the selected model
# if MODEL_NAME not in MODEL_CONFIGS:
#     raise ValueError(f"Error: No configuration found for model {MODEL_NAME}")
# else:
#     model_config = MODEL_CONFIGS[MODEL_NAME]
#
# print(f"Fine-tuning {MODEL_NAME}")
# print("=" * 80)
# print(f"Configuration for this model:")
# for key, value in model_config.items():
#     print(f"  {key}: {value}")
# print("=" * 80)
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
#
# def load_model_and_tokenizer():
#     """Load Llama 3 70B with 4-bit QLoRA"""
#     print(f"\nLoading tokenizer and model (4-bit QLoRA)...")
#
#     # 1. Setup QLoRA 4-bit config
#     # This will load the 70B model in 4-bit precision
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         # bfloat16 is supported on A6000 (Ampere) and is better for training
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )
#
#     try:
#         # Tokenizer uses HF_TOKEN from env
#         tokenizer = AutoTokenizer.from_pretrained(
#             MODEL_NAME,
#             trust_remote_code=True
#         )
#
#         # Model uses HF_TOKEN from env
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             device_map="auto",  # Automatically map layers across GPUs
#             quantization_config=bnb_config,  # Apply the 4-bit config
#             trust_remote_code=True
#         )
#         print("Model loaded successfully in 4-bit (QLoRA)")
#     except Exception as e:
#         print(f"Error loading model or tokenizer: {e}")
#         exit(1)
#
#     # Handle pad token
#     if tokenizer.pad_token is None:
#         print("No pad token found - using eos_token as pad_token")
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     else:
#         print(f"Pad token already exists: {tokenizer.pad_token}")
#
#     print(f"Tokenizer vocab size: {len(tokenizer)}")
#     print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
#
#     return tokenizer, model
#
#
# # Load components
# tokenizer, model = load_model_and_tokenizer()
#
# # Enable gradient checkpointing if specified (it is for 70B)
# # This is a redundant check but good practice
# if model_config.get("gradient_checkpointing", False):
#     print("Enabling gradient checkpointing for memory efficiency...")
#     model.gradient_checkpointing_enable()
#
# # LoRA Configuration
# print("\nConfiguring LoRA...")
# config = LoraConfig(
#     r=model_config["lora_r"],
#     lora_alpha=model_config["lora_alpha"],
#     # Target modules for Llama-3
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"
#     ],
#     lora_dropout=model_config["lora_dropout"],
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
#         max_length=512,  # Max sequence length
#         return_tensors="pt"
#     )
#     input_ids = tokenized["input_ids"]
#     # For Causal LM, labels are the same as input_ids
#     return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}
#
#
# print("\nTokenizing dataset...")
# tokenized_dataset = dataset.map(
#     tokenize_function,
#     batched=True,
#     remove_columns=["text"],
#     num_proc=4  # Use multiple processes to speed up
# )
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
#     learning_rate=model_config["learning_rate"],
#
#     # --- MEMORY OPTIMIZATIONS FOR 70B ---
#     per_device_train_batch_size=1,  # CRITICAL: Use 1 for 70B
#     per_device_eval_batch_size=1,  # CRITICAL: Use 1 for 70B
#     gradient_accumulation_steps=8,  # Compensate for batch size 1. Effective batch size = 8
#     gradient_checkpointing=model_config.get("gradient_checkpointing", False),
#     bf16=True,  # Use bfloat16 on A6000 (Ampere)
#     # ---
#
#     num_train_epochs=5,
#     warmup_steps=model_config["warmup_steps"],
#     weight_decay=model_config["weight_decay"],
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     eval_strategy="steps",
#     eval_steps=50,
#     load_best_model_at_end=True,
#     max_grad_norm=model_config["max_grad_norm"],
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     lr_scheduler_type=model_config["lr_scheduler_type"],
#     logging_first_step=True,
#     report_to="none",
# )
#
# # Initialize trainer
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
# print("=" * 80)
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
# print("\n" + "=" * 80)
# print("TRAINING SUMMARY")
# print("=" * 80)
# print(f"  Model: {MODEL_NAME}")
# print(f"  Training samples: {train_size}")
# print(f"  Evaluation samples: {eval_size}")
# print(f"  Tokenizer vocab size: {len(tokenizer)}")
# print(f"  Epochs: 5")
# print(f"  Learning rate: {model_config['learning_rate']}")
# print(f"  LoRA rank: {model_config['lora_r']}")
# print(f"  LoRA alpha: {model_config['lora_alpha']}")
# print(f"  Batch Size (Per Device): 1")
# print(f"  Grad. Accum. Steps: 8")
# print(f"  Effective Batch Size: 8")
# print(f"  Output directory: {OUTPUT_DIR}")
# print("=" * 80)
#
# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, \
#     BitsAndBytesConfig
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model
# from pathlib import Path
#
# # Configuration
# TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data_gpt_robustness\phi3_training_data_simple.csv"
# OUTPUT_DIR = Path("./finetuned_phi3_medium_gptdata")
#
# # MODEL SELECTION - Phi-3 Medium 14B
# MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"
#
# # ============================================================================
# # MODEL-SPECIFIC CONFIGURATIONS
# # ============================================================================
# MODEL_CONFIGS = {
#     # Phi-3 Medium 14B (Optimized for A6000)
#     "microsoft/Phi-3-medium-4k-instruct": {
#         "learning_rate": 2e-5,
#         "lora_r": 64,
#         "lora_alpha": 128,
#         "lora_dropout": 0.05,
#         "warmup_steps": 100,
#         "max_grad_norm": 1.0,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": True,
#         "target_modules": [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
#     },
#
#     # Other models for reference...
#     "meta-llama/Meta-Llama-3-8B-Instruct": {
#         "learning_rate": 3e-5,
#         "lora_r": 128,
#         "lora_alpha": 256,
#         "lora_dropout": 0.05,
#         "warmup_steps": 100,
#         "max_grad_norm": 1.0,
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "gradient_checkpointing": False,
#         "target_modules": [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
#     },
# }
#
# # Get configuration for the selected model
# if MODEL_NAME not in MODEL_CONFIGS:
#     raise ValueError(f"Error: No configuration found for model {MODEL_NAME}")
# else:
#     model_config = MODEL_CONFIGS[MODEL_NAME]
#
# print(f"Fine-tuning {MODEL_NAME}")
# print("=" * 80)
# print(f"Configuration for this model:")
# for key, value in model_config.items():
#     print(f"  {key}: {value}")
# print("=" * 80)
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
#
# def load_model_and_tokenizer():
#     """Load Phi-3 Medium with 4-bit QLoRA"""
#     print(f"\nLoading Phi-3 Medium tokenizer and model (4-bit QLoRA)...")
#
#     # QLoRA 4-bit config optimized for Phi-3
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )
#
#     try:
#         # Load tokenizer - Phi-3 requires trust_remote_code=True
#         tokenizer = AutoTokenizer.from_pretrained(
#             MODEL_NAME,
#             trust_remote_code=True
#         )
#
#         # Load model with 4-bit quantization
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             device_map="auto",
#             quantization_config=bnb_config,
#             trust_remote_code=True,
#             torch_dtype=torch.bfloat16
#         )
#         print("Phi-3 Medium loaded successfully in 4-bit (QLoRA)")
#     except Exception as e:
#         print(f"Error loading model or tokenizer: {e}")
#         exit(1)
#
#     # Handle pad token for Phi-3
#     if tokenizer.pad_token is None:
#         print("No pad token found - using eos_token as pad_token")
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#
#     print(f"Tokenizer vocab size: {len(tokenizer)}")
#     print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
#
#     return tokenizer, model
#
#
# # Load components
# tokenizer, model = load_model_and_tokenizer()
#
# # Enable gradient checkpointing if specified
# if model_config.get("gradient_checkpointing", False):
#     print("Enabling gradient checkpointing for memory efficiency...")
#     model.gradient_checkpointing_enable()
#
# # LoRA Configuration for Phi-3
# print("\nConfiguring LoRA for Phi-3...")
# config = LoraConfig(
#     r=model_config["lora_r"],
#     lora_alpha=model_config["lora_alpha"],
#     target_modules=model_config["target_modules"],
#     lora_dropout=model_config["lora_dropout"],
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
#         max_length=2048,  # Increased for Phi-3's 4k context
#         return_tensors="pt"
#     )
#     input_ids = tokenized["input_ids"]
#     # For Causal LM, labels are the same as input_ids
#     return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}
#
#
# print("\nTokenizing dataset...")
# tokenized_dataset = dataset.map(
#     tokenize_function,
#     batched=True,
#     remove_columns=dataset.column_names,
#     num_proc=4
# )
#
# # Split into train and eval
# if len(tokenized_dataset) > 1:
#     train_size = int(0.9 * len(tokenized_dataset))
#     eval_size = len(tokenized_dataset) - train_size
#     train_dataset = tokenized_dataset.select(range(train_size))
#     eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
# else:
#     # If only one sample, use it for both train and eval
#     train_dataset = tokenized_dataset
#     eval_dataset = tokenized_dataset
#
# print(f"Training samples: {len(train_dataset)}")
# print(f"Evaluation samples: {len(eval_dataset)}")
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
# # Training arguments optimized for Phi-3 Medium on A6000
# training_args = TrainingArguments(
#     output_dir=str(OUTPUT_DIR),
#     learning_rate=model_config["learning_rate"],
#
#     # Memory optimizations for Phi-3 Medium 14B
#     per_device_train_batch_size=2,  # Can use 2 for 14B on A6000
#     per_device_eval_batch_size=2,
#     gradient_accumulation_steps=4,  # Effective batch size = 8
#     gradient_checkpointing=model_config.get("gradient_checkpointing", False),
#
#     # Precision settings
#     bf16=True,  # Use bfloat16 on A6000
#
#     # Training schedule
#     num_train_epochs=3,  # Reduced epochs for faster experimentation
#     warmup_steps=model_config["warmup_steps"],
#     weight_decay=model_config["weight_decay"],
#
#     # Logging and saving
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     eval_strategy="steps" if len(eval_dataset) > 0 else "no",
#     eval_steps=50,
#     load_best_model_at_end=True if len(eval_dataset) > 0 else False,
#
#     # Optimization
#     max_grad_norm=model_config["max_grad_norm"],
#     metric_for_best_model="eval_loss" if len(eval_dataset) > 0 else None,
#     greater_is_better=False,
#     lr_scheduler_type=model_config["lr_scheduler_type"],
#     logging_first_step=True,
#     report_to="none",  # Disable wandb/etc logging
#
#     # Additional optimizations
#     dataloader_pin_memory=False,
#     ddp_find_unused_parameters=False,
# )
#
# # Initialize trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
#     tokenizer=tokenizer,
#     callbacks=[SaveTokenizerCallback(tokenizer)]
# )
#
# # Train the model
# print("\nStarting training...")
# print("=" * 80)
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
# print("\n" + "=" * 80)
# print("TRAINING SUMMARY")
# print("=" * 80)
# print(f"  Model: {MODEL_NAME}")
# print(f"  Training samples: {len(train_dataset)}")
# print(f"  Evaluation samples: {len(eval_dataset)}")
# print(f"  Tokenizer vocab size: {len(tokenizer)}")
# print(f"  Epochs: 3")
# print(f"  Learning rate: {model_config['learning_rate']}")
# print(f"  LoRA rank: {model_config['lora_r']}")
# print(f"  LoRA alpha: {model_config['lora_alpha']}")
# print(f"  Batch Size (Per Device): 2")
# print(f"  Grad. Accum. Steps: 4")
# print(f"  Effective Batch Size: 8")
# print(f"  Max Sequence Length: 2048")
# print(f"  Output directory: {OUTPUT_DIR}")
# print("=" * 80)


import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, \
    BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from pathlib import Path
from typing import Dict, Any, Union
from functools import partial

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# Note: Ensure your Hugging Face token is set as an environment variable (HF_TOKEN)
# ✅ UPDATED: Using new training data with concise system message
TRAINING_DATA_PATH = Path(r"..\1_DataPrep\formatted_training_data\phi3_training_data_simple.csv")
OUTPUT_DIR = Path("./finetuned_phi3_medium_gptdata_SystemMessage")
MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"  # Phi-3 Medium (14B)

# Model Configs - Defined globally for reference
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Phi-3 Medium 14B (Configuration)
    "microsoft/Phi-3-medium-4k-instruct": {
        "learning_rate": 2e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,  # Enabled for memory saving
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },
    # Llama 3 8B config left here for reference
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "learning_rate": 3e-5,
        "lora_r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": False,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },
}


# ============================================================================
# 2. CORE FUNCTIONS
# ============================================================================

def load_data(path: Path) -> Dataset:
    """Load and prepare training data."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise

    assert "text" in df.columns, "CSV must have 'text' column"
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    return Dataset.from_pandas(df)


def load_model_and_tokenizer(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load model with QLoRA and bfloat16."""
    print(f"\nLoading {model_name} tokenizer and model (4-bit QLoRA)...")

    # QLoRA 4-bit config optimized for A6000 (Ampere GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Load tokenizer - Phi-3 requires trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        print(f"{model_name} loaded successfully in 4-bit (QLoRA)")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

    # Handle pad token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("No explicit pad token found - using eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    return tokenizer, model


# Tokenize function (Accepts tokenizer as argument to solve NameError)
def tokenize_function_safe(examples: Dict[str, Union[str, list]], tokenizer_local: AutoTokenizer) -> Dict[str, Any]:
    """Tokenize text for causal language modeling, receiving tokenizer as an argument."""
    texts = examples["text"]
    tokenized = tokenizer_local(
        texts,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]

    # FIX: Return attention mask along with input_ids and labels
    return {
        "input_ids": input_ids.tolist(),
        "labels": input_ids.tolist(),
        "attention_mask": tokenized["attention_mask"].tolist()
    }


# Callback Class (Defined globally for multiprocessing safety)
class SaveTokenizerCallback(TrainerCallback):
    """Save tokenizer alongside model checkpoints"""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        """Called whenever a checkpoint is saved"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = Path(args.output_dir) / checkpoint_folder

        if output_dir.exists():
            print(f"Saving tokenizer to {output_dir}")
            self.tokenizer.save_pretrained(str(output_dir))


# ============================================================================
# 3. MAIN EXECUTION BLOCK (FIXED FOR MULTIPROCESSING/NAMING SCOPE)
# ============================================================================

if __name__ == '__main__':
    # 1. Configuration Lookup (FIX: Define model_config here)
    if MODEL_NAME not in MODEL_CONFIGS:
        print(f"Warning: No specific config for {MODEL_NAME}, using Mistral defaults")
        model_config = MODEL_CONFIGS["meta-llama/Meta-Llama-3-8B-Instruct"]
    else:
        model_config = MODEL_CONFIGS[MODEL_NAME]

    print(f"Fine-tuning {MODEL_NAME} with CONCISE system message")
    print("=" * 80)
    print(f"Configuration for this model:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # 2. Load Model and Tokenizer
    try:
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    except Exception:
        exit(1)

    # 3. Load Data
    try:
        dataset = load_data(TRAINING_DATA_PATH)
        print(f"✅ Loaded training data from: {TRAINING_DATA_PATH}")
        print(f"✅ Total samples: {len(dataset)}")
    except Exception:
        exit(1)

    # 4. Apply LoRA Configuration
    print("\nConfiguring LoRA...")
    config = LoraConfig(
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_alpha"],
        target_modules=model_config["target_modules"],
        lora_dropout=model_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # ✅ CRITICAL FIX: Enable gradients for inputs (required for gradient checkpointing with quantized models)
    print("\nEnabling input gradients for gradient checkpointing compatibility...")
    model.enable_input_require_grads()

    # 5. Tokenize dataset (CRITICAL FIX: Use partial to safely pass the tokenizer)
    print("\nTokenizing dataset (using 4 processes)...")

    tokenized_dataset = dataset.map(
        # Use partial to bind the local 'tokenizer' object to the 'tokenize_function_safe'
        partial(tokenize_function_safe, tokenizer_local=tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # 6. Split into train and eval
    if len(tokenized_dataset) > 1:
        train_size = int(0.9 * len(tokenized_dataset))
        if train_size == len(tokenized_dataset):
            train_dataset = tokenized_dataset
            eval_dataset = Dataset.from_dict({'input_ids': [], 'labels': []})
        else:
            train_dataset = tokenized_dataset.select(range(train_size))
            eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    else:
        train_dataset = tokenized_dataset
        eval_dataset = Dataset.from_dict({'input_ids': [], 'labels': []})

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # Determine evaluation strategy
    has_eval_data = len(eval_dataset) > 0 and len(eval_dataset) != len(train_dataset)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=model_config["learning_rate"],

        # Memory/Speed Optimizations
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        gradient_checkpointing=model_config["gradient_checkpointing"],
        bf16=True,  # Use bfloat16 on A6000

        # Training schedule
        num_train_epochs=3,  # ✅ Increased to 10 epochs for better convergence
        warmup_steps=model_config["warmup_steps"],
        weight_decay=model_config["weight_decay"],

        # Logging and saving
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps" if has_eval_data else "no",
        eval_steps=50,
        load_best_model_at_end=has_eval_data,
        metric_for_best_model="eval_loss" if has_eval_data else None,
        greater_is_better=False,
        lr_scheduler_type=model_config["lr_scheduler_type"],
        logging_first_step=True,
        report_to="none",
        dataloader_pin_memory=True,
    )

    # 8. Initialize and Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval_data else None,
        tokenizer=tokenizer,
        callbacks=[SaveTokenizerCallback(tokenizer)]
    )

    print("\nStarting training with CONCISE system message...")
    print("=" * 80)
    try:
        trainer.train()
        print("\nTraining completed successfully! 🎉")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)

    # 9. Save final model and tokenizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Final Summary
    print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    print(
        f"  Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  LoRA rank: {model_config['lora_r']}, Alpha: {model_config['lora_alpha']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  System message: CONCISE (one-sentence recommendations)")
    print("=" * 80)