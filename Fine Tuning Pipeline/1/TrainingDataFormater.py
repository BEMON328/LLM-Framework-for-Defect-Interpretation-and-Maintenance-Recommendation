# import pandas as pd
# import json
# import re
# from pathlib import Path
#
# # ============================================================================
# # CONFIGURATION - Update these paths as needed
# # ============================================================================
#
# INPUT_CSV_PATH = "separated_gpt4_response_v2.csv"
# OUTPUT_DIR = Path("formatted_training_data")
#
# OUTPUT_DIR.mkdir(exist_ok=True)
#
# print("🔧 MULTI-MODEL TRAINING DATA FORMATTER - PHI-3 WITH OPTIMIZED SYSTEM MESSAGE")
# print("=" * 60)
#
#
# class TrainingDataFormatter:
#     def __init__(self):
#         self.model_formats = {
#             'llama3': {
#                 'name': 'Llama-3 (Meta)',
#                 'format': 'human_assistant',
#                 'special_tokens': False
#             },
#             'mistral': {
#                 'name': 'Mistral-8B',
#                 'format': 'inst',
#                 'special_tokens': False
#             },
#             'qwen': {
#                 'name': 'Qwen-2.5',
#                 'format': 'human_assistant_qwen',
#                 'special_tokens': False
#             },
#             'gemma': {
#                 'name': 'Gemma-3',
#                 'format': 'user_model',
#                 'special_tokens': False
#             },
#             'phi3': {
#                 'name': 'Phi-3 Medium',
#                 'format': 'chatml_with_system',
#                 'special_tokens': False
#             }
#         }
#
#         self.cluster_data = None
#
#         # ✅ OPTIMIZED SYSTEM MESSAGE FOR PHI-3
#         self.phi3_system_message = "You are a bridge engineer. Provide brief, specific repair recommendations for steel girder bridges."
#
#     def load_raw_data(self, csv_path):
#         """Load and validate raw training data with cluster information"""
#         try:
#             df = pd.read_csv(csv_path)
#             print(f"Loaded {len(df)} training samples from {csv_path}")
#
#             # Validate required columns
#             required_cols = ['UserInput', 'Response']
#             for col in required_cols:
#                 if col not in df.columns:
#                     raise ValueError(f"Missing required column: {col}")
#
#             # Check if cluster information is available
#             if 'Cluster_ID' in df.columns:
#                 print(f"Found cluster information: {df['Cluster_ID'].nunique()} unique clusters")
#                 self.cluster_data = df
#             else:
#                 print("No cluster information found - will use generic references")
#
#             # Clean data
#             df = df.dropna(subset=required_cols)
#             df['UserInput'] = df['UserInput'].str.strip()
#             df['Response'] = df['Response'].str.strip()
#
#             # Remove empty entries
#             df = df[(df['UserInput'] != '') & (df['Response'] != '')]
#
#             print(f"After cleaning: {len(df)} valid samples")
#             return df
#
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return None
#
#     def extract_defect_entities(self, user_input):
#         """Extract both defect type and location from user input"""
#         input_lower = user_input.lower()
#
#         defect_types = {
#             'section loss': 'Section Loss',
#             'corrosion': 'Corrosion',
#             'crack': 'Crack',
#             'missing rivet': 'Missing Fastener',
#             'missing bolt': 'Missing Fastener',
#             'paint failure': 'Paint Failure',
#             'fatigue': 'Fatigue Damage'
#         }
#
#         locations = {
#             'main girder': 'Main Girder',
#             'cross girder': 'Cross Girder',
#             'flange': 'Flange',
#             'web plate': 'Web Plate',
#             'connection': 'Connection',
#             'bearing': 'Bearing Area',
#             'external face': 'External Face',
#             'internal face': 'Internal Face',
#             'lower crank': 'Lower Crank Angle',
#             'upper crank': 'Upper Crank Angle',
#             'girder': 'Girder',
#             'plate': 'Plate',
#             'beam': 'Beam'
#         }
#
#         found_defect = "General Defect"
#         found_location = "Unspecified Location"
#
#         for keyword, defect_type in defect_types.items():
#             if keyword in input_lower:
#                 found_defect = defect_type
#                 break
#
#         for keyword, location in locations.items():
#             if keyword in input_lower:
#                 found_location = location
#                 break
#
#         return f"{found_defect} at {found_location}"
#
#     def enhance_with_traceability(self, row):
#         """Add simple cluster ID reference to the response with location-aware classification"""
#         user_input = row['UserInput']
#         response = row['Response']
#
#         defect_and_location = self.extract_defect_entities(user_input)
#         main_answer = response.strip()
#         cluster_id = row.get('Cluster_ID')
#
#         if cluster_id:
#             traceable_response = f"{main_answer}. Defect: {defect_and_location} (Cluster {cluster_id})"
#         else:
#             traceable_response = f"{main_answer}. Defect: {defect_and_location}"
#
#         return traceable_response
#
#     def format_for_llama3(self, row, use_enhanced=True):
#         """Format for Llama-3"""
#         user_input = row['UserInput']
#         response = row['Response']
#
#         if use_enhanced:
#             enhanced_response = self.enhance_with_traceability(row)
#         else:
#             enhanced_response = response
#
#         return f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{enhanced_response}<|eot_id|>"
#
#     def format_for_mistral(self, row, use_enhanced=True):
#         """Format for Mistral"""
#         user_input = row['UserInput']
#         response = row['Response']
#
#         if use_enhanced:
#             enhanced_response = self.enhance_with_traceability(row)
#         else:
#             enhanced_response = response
#
#         return f"[INST] {user_input} [/INST] {enhanced_response}"
#
#     def format_for_qwen(self, row, use_enhanced=True):
#         """Format for Qwen-2.5"""
#         user_input = row['UserInput']
#         response = row['Response']
#
#         if use_enhanced:
#             enhanced_response = self.enhance_with_traceability(row)
#         else:
#             enhanced_response = response
#
#         return f"<|im_start|>system\nYou are a bridge engineering expert providing traceable repair recommendations.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{enhanced_response}<|im_end|>"
#
#     def format_for_gemma(self, row, use_enhanced=True):
#         """Format for Gemma"""
#         user_input = row['UserInput']
#         response = row['Response']
#
#         if use_enhanced:
#             enhanced_response = self.enhance_with_traceability(row)
#         else:
#             enhanced_response = response
#
#         return f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n{enhanced_response}<end_of_turn>"
#
#     def format_for_phi3(self, row, use_enhanced=True):
#         """
#         ✅ UPDATED: Phi-3 format WITH optimized system message
#         Uses ultra-concise system message for better performance
#         """
#         user_input = row['UserInput']
#         response = row['Response']
#
#         if use_enhanced:
#             enhanced_response = self.enhance_with_traceability(row)
#         else:
#             enhanced_response = response
#
#         # ✅ Phi-3 ChatML format WITH system message
#         return f"<|system|>\n{self.phi3_system_message}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n{enhanced_response}<|end|>"
#
#     def create_formatted_datasets(self, df, use_enhanced=True):
#         """Create formatted datasets for all model architectures"""
#         datasets = {}
#
#         print(f"\nCreating formatted datasets (Enhanced: {use_enhanced})...")
#
#         for model_key, model_info in self.model_formats.items():
#             print(f"Formatting for {model_info['name']}...")
#
#             formatted_data = []
#
#             for _, row in df.iterrows():
#                 if model_key == 'llama3':
#                     formatted = self.format_for_llama3(row, use_enhanced)
#                 elif model_key == 'mistral':
#                     formatted = self.format_for_mistral(row, use_enhanced)
#                 elif model_key == 'qwen':
#                     formatted = self.format_for_qwen(row, use_enhanced)
#                 elif model_key == 'gemma':
#                     formatted = self.format_for_gemma(row, use_enhanced)
#                 elif model_key == 'phi3':
#                     formatted = self.format_for_phi3(row, use_enhanced)
#
#                 formatted_data.append(formatted)
#
#             datasets[model_key] = {
#                 'name': model_info['name'],
#                 'data': formatted_data
#             }
#
#         return datasets
#
#     def save_datasets(self, datasets, use_enhanced=True):
#         """Save formatted datasets to files"""
#         suffix = "_traceable" if use_enhanced else "_simple"
#
#         print(f"\n💾 Saving formatted datasets...")
#
#         for model_key, dataset_info in datasets.items():
#             # Save as CSV format
#             csv_data = []
#             for text in dataset_info['data']:
#                 csv_data.append({'text': text})
#
#             csv_df = pd.DataFrame(csv_data)
#             csv_filename = f"{model_key}_training_data{suffix}.csv"
#             csv_path = OUTPUT_DIR / csv_filename
#             csv_df.to_csv(csv_path, index=False)
#
#             # Save as JSONL format
#             jsonl_filename = f"{model_key}_training_data{suffix}.jsonl"
#             jsonl_path = OUTPUT_DIR / jsonl_filename
#             with open(jsonl_path, 'w', encoding='utf-8') as f:
#                 for text in dataset_info['data']:
#                     json_obj = {'text': text}
#                     f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
#
#             print(f"  ✅ {dataset_info['name']}: {csv_filename} & {jsonl_filename}")
#             print(f"     Samples: {len(dataset_info['data'])}")
#
#     def create_sample_preview(self, datasets):
#         """Create a preview file showing sample formatted data"""
#         preview_path = OUTPUT_DIR / "sample_preview_phi3_with_system.txt"
#
#         with open(preview_path, 'w', encoding='utf-8') as f:
#             f.write("TRAINING DATA FORMAT PREVIEW - PHI-3 WITH OPTIMIZED SYSTEM MESSAGE\n")
#             f.write("=" * 80 + "\n")
#             f.write("✅ Phi-3 format WITH ultra-concise system message\n")
#             f.write(f"✅ System message: '{self.phi3_system_message}'\n")
#             f.write("✅ Goal: Clear role definition + brevity instruction\n")
#             f.write("=" * 80 + "\n\n")
#
#             for model_key, dataset_info in datasets.items():
#                 f.write(f"{dataset_info['name']} FORMAT:\n")
#                 f.write("-" * 60 + "\n")
#                 f.write(dataset_info['data'][0])
#                 f.write("\n\n" + "=" * 80 + "\n\n")
#
#         print(f"  📋 Sample preview: sample_preview_phi3_with_system.txt")
#
#     def run_formatting(self):
#         """Main execution function"""
#         # Load raw data
#         df = self.load_raw_data(INPUT_CSV_PATH)
#         if df is None:
#             return
#
#         # Create both versions
#         print("\n🚀 Creating traceable version...")
#         traceable_datasets = self.create_formatted_datasets(df, use_enhanced=True)
#         self.save_datasets(traceable_datasets, use_enhanced=True)
#
#         print("\n🚀 Creating simple version...")
#         simple_datasets = self.create_formatted_datasets(df, use_enhanced=False)
#         self.save_datasets(simple_datasets, use_enhanced=False)
#
#         # Create preview
#         self.create_sample_preview(simple_datasets)
#
#         print(f"\n✅ FORMATTING COMPLETE!")
#         print(f"📁 All files saved in: {OUTPUT_DIR}")
#         print(f"📊 Total samples per model: {len(df)}")
#
#         print(f"\n🎯 PHI-3 SYSTEM MESSAGE UPDATE:")
#         print("=" * 60)
#         print(f"✅ System message: '{self.phi3_system_message}'")
#         print("✅ Why optimized:")
#         print("   - Ultra-concise (15 words)")
#         print("   - Clear role definition")
#         print("   - Explicit brevity instruction")
#         print("   - No contradictory phrasing")
#         print("=" * 60)
#
#         print(f"\n📋 PHI-3 FORMAT STRUCTURE:")
#         print("\n<|system|>")
#         print(f"{self.phi3_system_message}<|end|>")
#         print("<|user|>")
#         print("[question]<|end|>")
#         print("<|assistant|>")
#         print("[response]<|end|>")
#
#         print("\n⚠️  NEXT STEPS:")
#         print("1. ✅ Data prepared with system message")
#         print("2. 📝 Update inference script to use SAME system message")
#         print("3. 🚀 Retrain Phi-3 with phi3_training_data_simple.csv")
#         print("4. 🧪 Test with updated inference script")
#         print("\n💡 Expected: Better performance than no-system-message version!")
#
#
# if __name__ == "__main__":
#     formatter = TrainingDataFormatter()
#     formatter.run_formatting()


import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, \
    BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from pathlib import Path

# Configuration
TRAINING_DATA_PATH = r"..\1_DataPrep\formatted_training_data_gpt_robustness\phi3_training_data_simple.csv"
OUTPUT_DIR = Path("./finetuned_phi3_medium_gptdata")

# MODEL SELECTION - Phi-3 Medium 14B
MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"

# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================
MODEL_CONFIGS = {
    # Phi-3 Medium 14B (Optimized for A6000)
    "microsoft/Phi-3-medium-4k-instruct": {
        "learning_rate": 2e-5,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },

    # Other models for reference...
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

# Get configuration for the selected model
if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(f"Error: No configuration found for model {MODEL_NAME}")
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


def load_model_and_tokenizer():
    """Load Phi-3 Medium with 4-bit QLoRA"""
    print(f"\nLoading Phi-3 Medium tokenizer and model (4-bit QLoRA)...")

    # QLoRA 4-bit config optimized for Phi-3
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Load tokenizer - Phi-3 requires trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        print("Phi-3 Medium loaded successfully in 4-bit (QLoRA)")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit(1)

    # Handle pad token for Phi-3
    if tokenizer.pad_token is None:
        print("No pad token found - using eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer, model


# Load components
tokenizer, model = load_model_and_tokenizer()

# Enable gradient checkpointing if specified
if model_config.get("gradient_checkpointing", False):
    print("Enabling gradient checkpointing for memory efficiency...")
    model.gradient_checkpointing_enable()

# LoRA Configuration for Phi-3
print("\nConfiguring LoRA for Phi-3...")
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


# Tokenize dataset
def tokenize_function(examples):
    """Tokenize text for causal language modeling"""
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=2048,  # Increased for Phi-3's 4k context
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    # For Causal LM, labels are the same as input_ids
    return {"input_ids": input_ids.tolist(), "labels": input_ids.tolist()}


print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4
)

# Split into train and eval
if len(tokenized_dataset) > 1:
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
else:
    # If only one sample, use it for both train and eval
    train_dataset = tokenized_dataset
    eval_dataset = tokenized_dataset

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")


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


# Training arguments optimized for Phi-3 Medium on A6000
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=model_config["learning_rate"],

    # Memory optimizations for Phi-3 Medium 14B
    per_device_train_batch_size=2,  # Can use 2 for 14B on A6000
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    gradient_checkpointing=model_config.get("gradient_checkpointing", False),

    # Precision settings
    bf16=True,  # Use bfloat16 on A6000

    # Training schedule
    num_train_epochs=3,  # Reduced epochs for faster experimentation
    warmup_steps=model_config["warmup_steps"],
    weight_decay=model_config["weight_decay"],

    # Logging and saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps" if len(eval_dataset) > 0 else "no",
    eval_steps=50,
    load_best_model_at_end=True if len(eval_dataset) > 0 else False,

    # Optimization
    max_grad_norm=model_config["max_grad_norm"],
    metric_for_best_model="eval_loss" if len(eval_dataset) > 0 else None,
    greater_is_better=False,
    lr_scheduler_type=model_config["lr_scheduler_type"],
    logging_first_step=True,
    report_to="none",  # Disable wandb/etc logging

    # Additional optimizations
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
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
print(f"  Training samples: {len(train_dataset)}")
print(f"  Evaluation samples: {len(eval_dataset)}")
print(f"  Tokenizer vocab size: {len(tokenizer)}")
print(f"  Epochs: 3")
print(f"  Learning rate: {model_config['learning_rate']}")
print(f"  LoRA rank: {model_config['lora_r']}")
print(f"  LoRA alpha: {model_config['lora_alpha']}")
print(f"  Batch Size (Per Device): 2")
print(f"  Grad. Accum. Steps: 4")
print(f"  Effective Batch Size: 8")
print(f"  Max Sequence Length: 2048")
print(f"  Output directory: {OUTPUT_DIR}")
print("=" * 80)