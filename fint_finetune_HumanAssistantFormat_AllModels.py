"""
finetune_all_models.py
======================
Fine-tunes all three Description-to-Action models sequentially in one run,
reading training data directly from the seeded train CSV produced by
datasetABCD.py (UserInput = defect, Response = repair).

Models:
  - meta-llama/Meta-Llama-3-8B-Instruct
  - microsoft/Phi-3-medium-4k-instruct   (13B)
  - microsoft/Phi-3-mini-4k-instruct     (3.8B)

Computational cost logged per model (fills Table 9):
  - Training time (seconds / formatted HH:MM:SS)
  - Peak GPU memory during training (GB, reserved)
  - Inference latency per query (mean over test set, seconds)
  - Peak GPU memory during inference (GB, reserved)

Outputs:
  - finetuned_<model>/ : saved adapters + tokenizer
  - finetune_cost.csv  : cost table rows (same schema as RAG cost table)

Usage:
  export HF_TOKEN=hf_...
  python finetune_all_models.py

  # To run a single model only:
  python finetune_all_models.py --models llama3

  # To skip training and only measure inference cost:
  python finetune_all_models.py --inference_only
"""

# ============================================================================
# WINDOWS TRITON WORKAROUND (keep first)
# ============================================================================
import sys, types
for mod in ['triton', 'triton.language', 'triton.compiler']:
    m = types.ModuleType(mod)
    m.__spec__ = types.ModuleType('spec')
    if mod == 'triton':
        m.__version__ = '2.0.0'
    sys.modules[mod] = m

import os
import gc
import json
import time
import argparse
import datetime
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Phi-3 is a bf16-native model and is numerically unstable in fp16 (overflow /
# NaN / degenerate output). Use bf16 whenever the GPU supports it, else fall
# back to fp16. This drives both the model load dtype and TrainingArguments.
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Training data: seed2 full train CSV from datasetABCD.py
# Paths are relative to this script so they work on any drive (J:, C:, etc.)
_DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "RAG" / "GYData" / "question_sets_seeded_gpt_300data" / "seed1"
TRAIN_CSV = (_DATA_DIR / "train_full_seed1.csv").resolve()
TEST_CSV = (_DATA_DIR / "question_set_a_test_seed1.csv").resolve()

COST_OUTPUT = Path("finetune_cost.csv")

# Evaluation model (same as RAG pipeline — must stay identical for fair comparison)
EMBED_MODEL_EVAL = "all-MiniLM-L6-v2"
THRESHOLD = 0.7

# Per-model configuration. Each model carries its own LoRA config, training
# args, and prepare_kbit flag so hyperparameters can be tuned independently.
MODELS = {
    "llama3": {
        "hf_id":     "meta-llama/Meta-Llama-3-8B-Instruct",
        "label":     "LLaMA3-8B-Instruct",
        "template":  "llama3",
        "output_dir": "./finetuned_llama3",
        "prepare_kbit": False,
        "lora": LoraConfig(
            r=32, lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
        "train": dict(
            num_train_epochs=3,            # 3 epochs for all models
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,            # was 2e-4 — more stable convergence
            weight_decay=0.01, fp16=not USE_BF16, bf16=USE_BF16, logging_steps=5,
            save_steps=100, save_total_limit=2, warmup_steps=50,
            report_to="none", remove_unused_columns=False,
            gradient_checkpointing=True, optim="adamw_torch",
            lr_scheduler_type="cosine", max_grad_norm=0.3,
        ),
    },
    "phi3_13b": {
        "hf_id":     "microsoft/Phi-3-medium-4k-instruct",
        "label":     "Phi3-13B-Instruct",
        "template":  "phi3",
        "output_dir": "./finetuned_phi3_13b",
        "prepare_kbit": True,             # required for 13B stability
        "lora": LoraConfig(
            r=16, lora_alpha=32,          # smaller rank for 13B stability
            # attention + MLP (fused Phi-3 medium names) for more capacity
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
        "train": dict(
            num_train_epochs=3,            # avoids overfitting on small data
            per_device_train_batch_size=1, # 13B needs smaller batch
            gradient_accumulation_steps=16, # maintains effective batch size
            learning_rate=1e-4,            # critical for 13B stability
            weight_decay=0.01, fp16=not USE_BF16, bf16=USE_BF16, logging_steps=5,
            save_steps=50, save_total_limit=3,
            warmup_steps=20, warmup_ratio=0.1,
            report_to="none", remove_unused_columns=False,
            gradient_checkpointing=True, optim="adamw_torch",
            lr_scheduler_type="cosine",
        ),
    },
    "phi3_3b": {
        "hf_id":     "microsoft/Phi-3-mini-4k-instruct",
        "label":     "Phi3-3.8B-Instruct",
        "template":  "phi3",
        "output_dir": "./finetuned_phi3_3b",
        "prepare_kbit": False,
        "lora": LoraConfig(
            r=32, lora_alpha=64,          # more capacity so mini actually converges
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
        "train": dict(
            num_train_epochs=5,            # extra epochs: last run stalled at loss~1.8
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,            # higher LR; mini under-trained at 1e-4
            weight_decay=0.01, fp16=not USE_BF16, bf16=USE_BF16, logging_steps=5,
            save_steps=100, save_total_limit=2, warmup_steps=10,  # short warmup on small set
            report_to="none", remove_unused_columns=False,
            gradient_checkpointing=True, optim="adamw_torch",
            lr_scheduler_type="cosine", max_grad_norm=0.3,
        ),
    },
}


# ============================================================================
# DATA FORMATTING
# ============================================================================
SYSTEM_PROMPT = (
    "You are a bridge engineering assistant. Given a defect description, "
    "recommend the single most appropriate repair method in one sentence."
)

def format_llama3(defect, repair=None):
    """Llama-3 chat format."""
    s = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Defect: {defect}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    if repair:
        s += f"{repair}<|eot_id|>"
    return s

def format_phi3(defect, repair=None):
    """Phi-3 chat format."""
    s = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\nDefect: {defect}<|end|>\n"
        f"<|assistant|>\n"
    )
    if repair:
        s += f"{repair}<|end|>"
    return s

FORMATTERS = {"llama3": format_llama3, "phi3": format_phi3}


def build_completion_labels(input_ids, response_start):
    """Mask prompt tokens; train only on the assistant completion."""
    labels = [-100] * len(input_ids)
    for j in range(response_start, len(input_ids)):
        labels[j] = input_ids[j]
    return labels


def find_response_start(input_ids, prompt_ids):
    """Align prompt/completion split after tokenization (handles truncation)."""
    # Fast path: separate encode of the prompt prefix matches full sequence.
    if input_ids[:len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)

    # Truncation can clip the tail; use longest shared prefix.
    n = min(len(input_ids), len(prompt_ids))
    for i in range(n, 0, -1):
        if input_ids[:i] == prompt_ids[:i]:
            return i

    return len(input_ids)


def build_dataset(train_csv, tokenizer, template):
    """Read train CSV, format and tokenize each defect-repair pair."""
    df = pd.read_csv(train_csv)
    formatter = FORMATTERS[template]
    records = []
    skipped = 0
    for _, row in df.iterrows():
        defect = str(row["UserInput"]).strip()
        repair = str(row["Response"]).strip()
        if not defect or not repair:
            continue

        prompt_text = formatter(defect, repair=None)
        full_text = formatter(defect, repair)
        enc = tokenizer(
            full_text, truncation=True, max_length=512,
            padding=False, return_tensors=None,
        )
        input_ids = enc["input_ids"]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_start = find_response_start(input_ids, prompt_ids)
        labels = build_completion_labels(input_ids, response_start)

        if all(l == -100 for l in labels):
            skipped += 1
            continue

        records.append({
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        })
    print(f"   Built {len(records)} training samples.", end="")
    if skipped:
        print(f" ({skipped} skipped — no trainable completion tokens)")
    else:
        print()
    return Dataset.from_list(records)


def build_or_load_tokenized(train_csv, tokenizer, template, cache_path):
    """Load pre-tokenized JSON if present, else tokenize the CSV and cache it.

    The cache is keyed per model (via cache_path under each model's output_dir)
    and per seed (via the train CSV stem), because input_ids/labels depend on
    the model's own tokenizer and chat template. Mirrors the pre-tokenized JSON
    workflow used by the standalone Phi-3 scripts.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading pre-tokenized data from {cache_path.name}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"   Loaded {len(records)} pre-tokenized samples.")
        return Dataset.from_list(records)

    ds = build_dataset(train_csv, tokenizer, template)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ds.to_list(), f)
    print(f"   Cached pre-tokenized data to {cache_path.name}")
    return ds


# ============================================================================
# DATA COLLATOR (unchanged from original)
# ============================================================================
class DataCollatorForCompletionOnlyLM:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = ((max_len + 7) // 8) * 8   # pad to multiple of 8
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ============================================================================
# COST HELPERS
# ============================================================================
def free_gpu_memory():
    """Release GPU memory between sequential model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_base_model(hf_id, prepare_kbit=False):
    """Load causal LM; keep on a single GPU when possible to avoid meta tensors."""
    free_gpu_memory()
    kwargs = dict(
        token=HF_TOKEN,
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        # Single-GPU load avoids accelerate meta/offload tensors that break Trainer.
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.gradient_checkpointing_enable()
    if prepare_kbit:
        # Stabilises gradient flow (casts norms to fp32, enables input grads).
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    model.enable_input_require_grads()
    return model


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / (1024 ** 3)
    return 0.0

def fmt_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# ============================================================================
# INFERENCE TIMING
# ============================================================================
def measure_inference_cost(model, tokenizer, test_csv, template, eval_model):
    """Generate repairs for every Set A test row; measure latency + peak memory.
    Returns (mean_latency_s, peak_mem_gb, correctness_pct)."""
    df = pd.read_csv(test_csv)
    formatter = FORMATTERS[template]
    device = next(model.parameters()).device

    reset_peak_memory()
    latencies = []
    sims = []
    model.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            defect = str(row["UserInput"]).strip()
            gt = str(row["Response"]).strip()
            prompt = formatter(defect, repair=None)
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512).to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,  # avoids DynamicCache.get_max_length API mismatch (Phi-3 remote code vs transformers 4.49)
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

            # decode only generated tokens
            gen = out[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = pred.split("\n")[0].strip()

            # score
            ea = eval_model.encode(gt, convert_to_tensor=True)
            eb = eval_model.encode(pred, convert_to_tensor=True)
            sims.append(util.cos_sim(ea, eb).item())

    mean_lat = sum(latencies) / len(latencies)
    mem = peak_memory_gb()
    correctness = sum(s >= THRESHOLD for s in sims) / len(sims) * 100
    return mean_lat, mem, correctness


# ============================================================================
# MAIN: TRAIN + MEASURE PER MODEL
# ============================================================================
def run_model(key, cfg, args, eval_model):
    print("\n" + "=" * 70)
    print(f"  MODEL: {cfg['label']}  ({cfg['hf_id']})")
    print("=" * 70)

    # ---- Tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["hf_id"], token=HF_TOKEN, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cost_row = {"model": cfg["label"], "hf_id": cfg["hf_id"]}

    # ---- TRAINING ----
    if not args.inference_only:
        print("Preparing training dataset (pre-tokenized JSON cache)...")
        cache_path = Path(cfg["output_dir"]) / f"tokenized_{TRAIN_CSV.stem}.json"
        train_dataset = build_or_load_tokenized(
            TRAIN_CSV, tokenizer, cfg["template"], cache_path
        )

        print(f"Loading base model ({'BF16' if USE_BF16 else 'FP16'})...")
        reset_peak_memory()
        model = load_base_model(cfg["hf_id"], prepare_kbit=cfg.get("prepare_kbit", False))
        model = get_peft_model(model, cfg["lora"])
        model.print_trainable_parameters()

        t_args = TrainingArguments(
            output_dir=cfg["output_dir"],
            logging_dir=cfg["output_dir"] + "/logs",
            **cfg["train"],
        )
        trainer = Trainer(
            model=model,
            args=t_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
        )

        print(f"\nTraining {cfg['label']}...")
        reset_peak_memory()
        t_train_start = time.perf_counter()
        trainer.train()
        train_time_s = time.perf_counter() - t_train_start
        train_mem_gb = peak_memory_gb()

        trainer.save_model()
        tokenizer.save_pretrained(cfg["output_dir"])

        cost_row["training_time_s"] = round(train_time_s, 1)
        cost_row["training_time_fmt"] = fmt_duration(train_time_s)
        cost_row["peak_mem_train_gb"] = round(train_mem_gb, 2)
        print(f"\n[OK] Training done: {fmt_duration(train_time_s)}, "
              f"peak mem {train_mem_gb:.2f} GB")

        # free training memory before inference measurement
        del trainer, model
        free_gpu_memory()

    else:
        cost_row["training_time_s"] = "skipped"
        cost_row["training_time_fmt"] = "skipped"
        cost_row["peak_mem_train_gb"] = "skipped"

    # ---- INFERENCE COST ----
    print(f"\nLoading fine-tuned model for inference timing...")
    base_model = load_base_model(cfg["hf_id"])
    ft_model = PeftModel.from_pretrained(base_model, cfg["output_dir"])
    ft_model.eval()
    if hasattr(ft_model, "config"):
        ft_model.config.use_cache = False

    print(f"Measuring inference latency over {TEST_CSV.name}...")
    mean_lat, infer_mem, correctness = measure_inference_cost(
        ft_model, tokenizer, TEST_CSV, cfg["template"], eval_model,
    )
    cost_row["inference_latency_mean_s"] = round(mean_lat, 2)
    cost_row["peak_mem_infer_gb"] = round(infer_mem, 2)
    cost_row["correctness_pct"] = round(correctness, 2)

    print(f"[OK] Inference: {mean_lat:.2f} s/query, "
          f"peak mem {infer_mem:.2f} GB, correctness {correctness:.1f}%")

    del ft_model, base_model
    free_gpu_memory()

    return cost_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    choices=list(MODELS.keys()) + ["all"],
                    default=["all"],
                    help="Which models to run (default: all)")
    ap.add_argument("--inference_only", action="store_true",
                    help="Skip training; only measure inference cost "
                         "(requires saved adapters in output_dir)")
    args = ap.parse_args()

    keys = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"\nFine-tuning run: {keys}")
    print(f"Train CSV:  {TRAIN_CSV}")
    print(f"Test CSV:   {TEST_CSV}")
    print(f"Inference only: {args.inference_only}\n")

    # Load the shared evaluation model once (CPU — only needed for inference scoring)
    print("Loading judge model for inference scoring...")
    eval_model = SentenceTransformer(EMBED_MODEL_EVAL)

    cost_rows = []
    for key in keys:
        row = run_model(key, MODELS[key], args, eval_model)
        cost_rows.append(row)
        # save after each model so a crash doesn't lose earlier results
        pd.DataFrame(cost_rows).to_csv(COST_OUTPUT, index=False)
        print(f"\nCost table updated: {COST_OUTPUT}")

    print("\n" + "=" * 70)
    print("FINAL COST SUMMARY")
    print("=" * 70)
    df = pd.DataFrame(cost_rows)
    print(df[["model", "training_time_fmt",
              "peak_mem_train_gb",
              "inference_latency_mean_s",
              "peak_mem_infer_gb",
              "correctness_pct"]].to_string(index=False))
    print(f"\nSaved to: {COST_OUTPUT}")


if __name__ == "__main__":
    main()