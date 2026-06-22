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
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # ✅ Changed from medium to mini
max_samples = None
HF_TOKEN = "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"

SYSTEM_MESSAGE = "You are a bridge inspection assistant. For each defect description provided, give a brief, specific repair recommendation."

CHECKPOINTS_DIR = Path(r"C:\Users\Gx\Desktop\FineTune\2_FineTuning\finetuned_phi3_windows")  # ✅ Updated path
OUTPUT_DIR = Path("./evaluation_results_phi3_mini_gptData_accuracy")  # ✅ Updated output path
OUTPUT_DIR.mkdir(exist_ok=True)
TEST_CSV = r"..\GYData\question_sets_separated_gpt4_response_counted_v2\separated_gpt4_response_v2_by1.csv"

# ✅ PROPERLY OPTIMIZED CONFIG
GENERATION_CONFIG = {
    "max_new_tokens": 80,
    "min_new_tokens": 10,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "num_beams": 1,
}

print("Phi-3 Mini PROPERLY FIXED Batch Evaluation")
print("=" * 80)


def format_prompt(prompt):
    """Format prompt WITH system message to match training format"""
    return f"<|system|>\n{SYSTEM_MESSAGE}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"


def extract_response(full_text):
    """
    ✅ PROPERLY FIXED: Extract only the FIRST complete sentence
    Stop at: first newline OR first period followed by space/newline
    """

    # Find assistant response
    if "<|assistant|>" in full_text:
        parts = full_text.split("<|assistant|>")
        if len(parts) > 1:
            response = parts[1]

            # Remove explicit end tokens first
            for token in ["<|end|>", "<|endoftext|>", "<|assistant|>", "<|user|>", "<|system|>", "</s>", "<s>"]:
                if token in response:
                    response = response.split(token)[0]

            response = response.strip()

            # ✅ CRITICAL: Take only FIRST line (before any newline)
            if '\n' in response:
                lines = response.split('\n')
                # Take first non-empty line
                for line in lines:
                    if line.strip():
                        response = line.strip()
                        break

            # ✅ If response contains multiple sentences, take only the first one
            if '. ' in response:  # Period followed by space
                sentences = response.split('. ')
                response = sentences[0].strip()
                if not response.endswith('.'):
                    response += '.'

            return response.strip()

    # Fallback
    response = full_text.strip()
    for token in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>", "</s>", "<s>"]:
        response = response.replace(token, "")

    # Take first line
    if '\n' in response:
        lines = response.split('\n')
        for line in lines:
            if line.strip():
                response = line.strip()
                break

    return response.strip()


def load_model_and_tokenizer(checkpoint_path, base_model):
    """Load fine-tuned model"""
    print(f"\nLoading checkpoint: {checkpoint_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)

    print(f"  ✅ Model loaded")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, generation_config):
    """
    ✅ PROPERLY FIXED with DynamicCache error prevention
    """
    formatted_prompt = format_prompt(prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Get EOS tokens
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    endoftext_token_id = tokenizer.eos_token_id
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

    eos_token_ids = []
    if end_token_id not in [None, tokenizer.unk_token_id]:
        eos_token_ids.append(end_token_id)
    if endoftext_token_id is not None:
        eos_token_ids.append(endoftext_token_id)
    if assistant_token_id not in [None, tokenizer.unk_token_id]:
        eos_token_ids.append(assistant_token_id)
    if not eos_token_ids:
        eos_token_ids = [tokenizer.eos_token_id]

    # ✅ CRITICAL FIX: Added use_cache=False to prevent DynamicCache error
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                min_new_tokens=generation_config["min_new_tokens"],
                do_sample=generation_config["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
                use_cache=False,  # ✅ CRITICAL: Prevents DynamicCache error
            )
    except Exception as e:
        print(f"  ⚠️ Generation error: {e}, retrying with fallback...")
        # Fallback: try without eos_token_id specification
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                min_new_tokens=generation_config["min_new_tokens"],
                do_sample=generation_config["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )

    # Decode full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract ONLY first sentence
    clean_response = extract_response(full_response)

    # Calculate tokens generated
    input_len = inputs['input_ids'].shape[1]
    output_len = outputs.shape[1]
    tokens_generated = output_len - input_len

    # Check if response has newlines
    has_newline = '\n' in full_response.split("<|assistant|>")[-1].split("<|end|>")[
        0] if "<|assistant|>" in full_response else False

    return clean_response, tokens_generated, has_newline


def calculate_similarity(text1, text2, similarity_model):
    """Calculate cosine similarity"""
    emb1 = similarity_model.encode(text1, convert_to_tensor=True)
    emb2 = similarity_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() * 100


def evaluate_checkpoint(checkpoint_path, test_df, similarity_model, base_model, generation_config, max_samples):
    """Evaluate a checkpoint"""
    try:
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, base_model)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return []

    results = []
    test_df_subset = test_df.head(max_samples) if max_samples else test_df

    print(f"Evaluating {len(test_df_subset)} samples...")

    total_tokens = 0
    multi_line_count = 0
    error_count = 0

    for idx, row in tqdm(test_df_subset.iterrows(), total=len(test_df_subset), desc="  Progress"):
        try:
            prompt = row['UserInput']
            ground_truth = row['Response']

            response, tokens_gen, has_newline = generate_response(model, tokenizer, prompt, generation_config)

            total_tokens += tokens_gen
            if has_newline:
                multi_line_count += 1

            # Handle empty responses
            if not response or response.strip() == "":
                response = "[EMPTY RESPONSE]"

            similarity = calculate_similarity(response, ground_truth, similarity_model)

            # Determine if response is "correct" (similarity >= 70%)
            is_correct = similarity >= 70.0

            results.append({
                'checkpoint': checkpoint_path.name,
                'sample_id': idx,
                'prompt': prompt,
                'ground_truth': ground_truth,
                'generated': response,
                'similarity': similarity,
                'is_correct': is_correct,
                'response_length': len(response),
                'tokens_generated': tokens_gen,
                'had_newline': has_newline
            })
        except Exception as e:
            error_count += 1
            print(f"\n  ❌ Error on sample {idx}: {e}")
            results.append({
                'checkpoint': checkpoint_path.name,
                'sample_id': idx,
                'prompt': prompt if 'prompt' in locals() else '[UNKNOWN]',
                'ground_truth': ground_truth if 'ground_truth' in locals() else '[UNKNOWN]',
                'generated': f"[ERROR: {str(e)}]",
                'similarity': 0.0,
                'is_correct': False,
                'response_length': 0,
                'tokens_generated': 0,
                'had_newline': False
            })
            continue

    avg_tokens = total_tokens / len(test_df_subset) if test_df_subset.shape[0] > 0 else 0
    print(f"\n  ℹ️  Average tokens generated: {avg_tokens:.1f}")
    print(
        f"  ℹ️  Responses with newlines: {multi_line_count}/{len(test_df_subset)} ({multi_line_count / len(test_df_subset) * 100:.1f}%)")
    if error_count > 0:
        print(f"  ⚠️  Errors encountered: {error_count}/{len(test_df_subset)}")

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    print(f"Base Model: {BASE_MODEL}")
    print(f"System Message: '{SYSTEM_MESSAGE}'")
    print(f"Generation: GREEDY (let model generate naturally)")
    print(f"Extraction: Take ONLY first line/sentence from output")
    print(f"Format: WITH SYSTEM MESSAGE + PROPER MASKING")
    print("=" * 80)

    # Load test data
    try:
        test_df = pd.read_csv(TEST_CSV)
        print(f"✅ Loaded {len(test_df)} test samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Load similarity model
    print("Loading similarity model...")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Similarity model loaded")

    # Find checkpoints
    checkpoints = [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    if not checkpoints and (CHECKPOINTS_DIR / "adapter_config.json").exists():
        checkpoints = [CHECKPOINTS_DIR]
    else:
        checkpoints = sorted(checkpoints,
                             key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)

    # Use final checkpoint
    if checkpoints:
        final_checkpoint = CHECKPOINTS_DIR / "final"
        if final_checkpoint.exists() and (final_checkpoint / "adapter_config.json").exists():
            checkpoints = [final_checkpoint]
        else:
            checkpoints = [checkpoints[-1]]

    if not checkpoints:
        print("❌ No checkpoints found!")
        print(f"Looking in: {CHECKPOINTS_DIR}")
        return

    print(f"✅ Evaluating final checkpoint: {checkpoints[0].name if checkpoints else 'None'}")
    print("=" * 80)

    # Evaluate
    all_results = []
    for checkpoint in checkpoints:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {checkpoint.name}")
        print(f"{'=' * 80}")
        results = evaluate_checkpoint(checkpoint, test_df, similarity_model, BASE_MODEL, GENERATION_CONFIG, max_samples)
        all_results.extend(results)

    if not all_results:
        print("\n❌ No results generated!")
        return

    # Save results
    df = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / "evaluation_results_phi3_mini.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n{'=' * 80}")
    print(f"✅ Results saved to: {output_file}")

    # Diagnostics
    multi_line_extracted = (df['generated'].str.contains('\n', na=False)).sum()
    had_newlines = df['had_newline'].sum()

    print(f"\n📊 Multi-line Generation Analysis:")
    print(f"  Model generated multi-line: {had_newlines}/{len(df)} ({had_newlines / len(df) * 100:.1f}%)")
    print(
        f"  After extraction (should be 0): {multi_line_extracted}/{len(df)} ({multi_line_extracted / len(df) * 100:.1f}%)")

    if multi_line_extracted > 0:
        print(f"\n  ⚠️  WARNING: {multi_line_extracted} responses still have newlines after extraction!")
    else:
        print(f"\n  ✅ SUCCESS: All extracted responses are single-line!")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - PHI-3 MINI")
    print("=" * 80)

    # Correctness Analysis (Similarity >= 70%)
    correct_count = df['is_correct'].sum()
    correct_percentage = (correct_count / len(df)) * 100

    print(f"\n🎯 CORRECTNESS ANALYSIS (Similarity >= 70%):")
    print(f"  Correct responses: {correct_count}/{len(df)} ({correct_percentage:.1f}%)")
    print(f"  Incorrect responses: {len(df) - correct_count}/{len(df)} ({100 - correct_percentage:.1f}%)")

    print(f"\n📊 Similarity Statistics:")
    print(f"Mean Similarity:     {df['similarity'].mean():.2f}%")
    print(f"Median Similarity:   {df['similarity'].median():.2f}%")
    print(f"Std Deviation:       {df['similarity'].std():.2f}%")
    print(f"Min Similarity:      {df['similarity'].min():.2f}%")
    print(f"Max Similarity:      {df['similarity'].max():.2f}%")

    print(f"\n📈 Similarity Distribution:")
    print(f"  Above 90%: {(df['similarity'] >= 90).sum()} ({(df['similarity'] >= 90).sum() / len(df) * 100:.1f}%)")
    print(f"  Above 80%: {(df['similarity'] >= 80).sum()} ({(df['similarity'] >= 80).sum() / len(df) * 100:.1f}%)")
    print(f"  Above 70%: {(df['similarity'] >= 70).sum()} ({(df['similarity'] >= 70).sum() / len(df) * 100:.1f}%)")
    print(
        f"  50-70%: {((df['similarity'] >= 50) & (df['similarity'] < 70)).sum()} ({((df['similarity'] >= 50) & (df['similarity'] < 70)).sum() / len(df) * 100:.1f}%)")
    print(f"  Below 50%: {(df['similarity'] < 50).sum()} ({(df['similarity'] < 50).sum() / len(df) * 100:.1f}%)")

    print(f"\nResponse Quality:")
    print(f"  Average length: {df['response_length'].mean():.1f} chars")
    print(f"  Median length: {df['response_length'].median():.1f} chars")
    print(f"  Average tokens: {df['tokens_generated'].mean():.1f}")
    print(f"  Too short (<10): {(df['response_length'] < 10).sum()}")
    print(f"  Too long (>200): {(df['response_length'] > 200).sum()}")

    print("=" * 80)

    # Print examples
    print(f"\n{'=' * 80}")
    print("SAMPLE OUTPUTS (First 5 examples):")
    print("=" * 80)
    for i in range(min(5, len(df))):
        print(f"\n--- Example {i + 1} ---")
        print(f"Prompt: {df.iloc[i]['prompt'][:80]}...")
        print(f"Ground Truth: {df.iloc[i]['ground_truth']}")
        print(f"Generated: {df.iloc[i]['generated']}")
        print(f"Similarity: {df.iloc[i]['similarity']:.2f}%")
        print(f"Correct (≥70%): {'✅ YES' if df.iloc[i]['is_correct'] else '❌ NO'}")
        print(f"Length: {df.iloc[i]['response_length']} chars")
        print(f"Had multi-line: {'Yes' if df.iloc[i]['had_newline'] else 'No'}")

    # Generate correctness summary file
    print(f"\n{'=' * 80}")
    print("GENERATING DETAILED CORRECTNESS REPORT")
    print("=" * 80)

    # Separate correct and incorrect samples
    correct_samples = df[df['is_correct'] == True].copy()
    incorrect_samples = df[df['is_correct'] == False].copy()

    # Create summary report
    summary_file = OUTPUT_DIR / "correctness_summary_phi3_mini.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHI-3 MINI FINE-TUNED MODEL - CORRECTNESS EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("CORRECTNESS DEFINITION: Similarity >= 70% with ground truth\n\n")

        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Correct responses: {correct_count} ({correct_percentage:.1f}%)\n")
        f.write(f"Incorrect responses: {len(df) - correct_count} ({100 - correct_percentage:.1f}%)\n\n")

        f.write(f"Mean similarity: {df['similarity'].mean():.2f}%\n")
        f.write(f"Median similarity: {df['similarity'].median():.2f}%\n")
        f.write(f"Std deviation: {df['similarity'].std():.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"CORRECT RESPONSES ({len(correct_samples)} samples)\n")
        f.write("=" * 80 + "\n\n")

        if len(correct_samples) > 0:
            f.write(
                f"Similarity range: {correct_samples['similarity'].min():.1f}% - {correct_samples['similarity'].max():.1f}%\n")
            f.write(f"Average similarity: {correct_samples['similarity'].mean():.1f}%\n")
            f.write(f"Average length: {correct_samples['response_length'].mean():.1f} chars\n\n")

            f.write("Sample correct responses:\n")
            f.write("-" * 80 + "\n")
            for idx, row in correct_samples.head(10).iterrows():
                f.write(f"\n[Sample {row['sample_id']}] Similarity: {row['similarity']:.1f}%\n")
                f.write(f"Prompt: {row['prompt']}\n")
                f.write(f"Ground Truth: {row['ground_truth']}\n")
                f.write(f"Generated: {row['generated']}\n")
                f.write("-" * 80 + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"INCORRECT RESPONSES ({len(incorrect_samples)} samples)\n")
        f.write("=" * 80 + "\n\n")

        if len(incorrect_samples) > 0:
            f.write(
                f"Similarity range: {incorrect_samples['similarity'].min():.1f}% - {incorrect_samples['similarity'].max():.1f}%\n")
            f.write(f"Average similarity: {incorrect_samples['similarity'].mean():.1f}%\n")
            f.write(f"Average length: {incorrect_samples['response_length'].mean():.1f} chars\n\n")

            # Group by similarity ranges
            very_low = incorrect_samples[incorrect_samples['similarity'] < 30]
            low = incorrect_samples[(incorrect_samples['similarity'] >= 30) & (incorrect_samples['similarity'] < 50)]
            borderline = incorrect_samples[
                (incorrect_samples['similarity'] >= 50) & (incorrect_samples['similarity'] < 70)]

            f.write(f"Breakdown:\n")
            f.write(f"  Very low (<30%): {len(very_low)} samples\n")
            f.write(f"  Low (30-50%): {len(low)} samples\n")
            f.write(f"  Borderline (50-70%): {len(borderline)} samples\n\n")

            f.write("Sample incorrect responses (sorted by similarity, worst first):\n")
            f.write("-" * 80 + "\n")
            for idx, row in incorrect_samples.nsmallest(10, 'similarity').iterrows():
                f.write(f"\n[Sample {row['sample_id']}] Similarity: {row['similarity']:.1f}%\n")
                f.write(f"Prompt: {row['prompt']}\n")
                f.write(f"Ground Truth: {row['ground_truth']}\n")
                f.write(f"Generated: {row['generated']}\n")
                f.write("-" * 80 + "\n")

    print(f"✅ Detailed correctness report saved to: {summary_file}")
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()