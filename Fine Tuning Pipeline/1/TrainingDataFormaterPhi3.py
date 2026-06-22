import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV_PATH = "separated_gpt4_response_v2_by1.csv"
OUTPUT_DIR = Path("formatted_training_data_phi3")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_MODEL = "microsoft/Phi-3-medium-4k-instruct"
SYSTEM_MESSAGE = "You are a bridge inspection assistant. For each defect description provided, give a brief, specific repair recommendation."

print("🔧 PHI-3 TRAINING DATA PREPARATION - WITH PROPER LOSS MASKING")
print("=" * 80)

# Load tokenizer
print(f"Loading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Tokenizer loaded")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")


def create_conversation_text(user_input, response):
    """Create the formatted conversation text"""
    return f"<|system|>\n{SYSTEM_MESSAGE}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n{response}<|end|>"


def tokenize_with_loss_masking(example):
    """
    ✅ CRITICAL FIX: Tokenize with proper loss masking

    Only compute loss on assistant's response tokens.
    Mask everything else with -100 (ignored by PyTorch loss).
    """
    # Get the full conversation text
    full_text = example['text']

    # Tokenize the full text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding=False,  # Don't pad yet
        return_tensors=None
    )

    input_ids = tokenized['input_ids']

    # Initialize labels with -100 (masked)
    labels = [-100] * len(input_ids)

    # Find where assistant's response starts
    # We need to find "<|assistant|>\n" in the token sequence
    assistant_start_text = "<|assistant|>\n"
    assistant_tokens = tokenizer.encode(assistant_start_text, add_special_tokens=False)

    # Search for the assistant token sequence
    assistant_start_idx = None
    for i in range(len(input_ids) - len(assistant_tokens) + 1):
        if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
            # Found it! The response starts AFTER these tokens
            assistant_start_idx = i + len(assistant_tokens)
            break

    if assistant_start_idx is None:
        print(f"⚠️  WARNING: Could not find <|assistant|> tag in example")
        return {"input_ids": input_ids, "attention_mask": tokenized['attention_mask'], "labels": labels}

    # Unmask tokens from assistant's response onwards
    # But stop at <|end|> token if present
    end_token_text = "<|end|>"
    end_tokens = tokenizer.encode(end_token_text, add_special_tokens=False)

    # Find where response ends (first <|end|> after assistant tag)
    response_end_idx = len(input_ids)  # Default to end of sequence
    for i in range(assistant_start_idx, len(input_ids) - len(end_tokens) + 1):
        if input_ids[i:i + len(end_tokens)] == end_tokens:
            response_end_idx = i + len(end_tokens)  # Include the end token
            break

    # Unmask the assistant's response (including <|end|>)
    for i in range(assistant_start_idx, response_end_idx):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized['attention_mask'],
        "labels": labels
    }


def verify_masking(tokenized_example, original_text):
    """Verify that masking is working correctly"""
    input_ids = tokenized_example['input_ids']
    labels = tokenized_example['labels']

    # Count masked vs unmasked
    total_tokens = len(input_ids)
    masked_tokens = sum(1 for l in labels if l == -100)
    unmasked_tokens = total_tokens - masked_tokens

    print(f"\n{'=' * 80}")
    print("MASKING VERIFICATION")
    print(f"{'=' * 80}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Masked tokens:   {masked_tokens} ({masked_tokens / total_tokens * 100:.1f}%)")
    print(f"Unmasked tokens: {unmasked_tokens} ({unmasked_tokens / total_tokens * 100:.1f}%)")

    # Show which parts are masked
    print(f"\n{'=' * 80}")
    print("TOKEN-BY-TOKEN BREAKDOWN (First 50 tokens)")
    print(f"{'=' * 80}")
    print(f"{'Token':<30} {'Input ID':<10} {'Label':<10} {'Status'}")
    print("-" * 80)

    for i in range(min(50, len(input_ids))):
        token_text = tokenizer.decode([input_ids[i]])
        status = "MASKED" if labels[i] == -100 else "TRAINED"
        print(f"{token_text[:28]:<30} {input_ids[i]:<10} {labels[i]:<10} {status}")

    # Decode what will actually be trained on
    trained_tokens = [input_ids[i] for i in range(len(input_ids)) if labels[i] != -100]
    trained_text = tokenizer.decode(trained_tokens)

    print(f"\n{'=' * 80}")
    print("WHAT THE MODEL WILL LEARN (unmasked text only)")
    print(f"{'=' * 80}")
    print(trained_text)
    print(f"{'=' * 80}")

    # ✅ UPDATED: Sanity checks
    if unmasked_tokens == 0:
        print("\n❌ ERROR: No tokens to train on! All tokens are masked.")
        return False

    if unmasked_tokens == total_tokens:
        print("\n❌ ERROR: No tokens are masked! Model will learn inputs too.")
        return False

    # ✅ UPDATED: Check if <|assistant|> tag was found and masked correctly
    # The assistant tag itself should be masked, but content after should be unmasked
    assistant_tag_found = False
    for i in range(len(input_ids) - 1):
        token = tokenizer.decode([input_ids[i]])
        if '<|assistant|>' in token:
            assistant_tag_found = True
            # Check that tokens AFTER assistant tag are unmasked
            if i + 1 < len(input_ids) and labels[i + 1] != -100:
                print("\n✅ Assistant tag found and tokens after it are being trained!")
                break

    if not assistant_tag_found:
        print("\n⚠️  WARNING: Could not verify <|assistant|> tag location")
        return False

    # ✅ UPDATED: More lenient keyword check - just check it's not empty or too short
    if len(trained_text.strip()) < 5:
        print("\n❌ ERROR: Trained text is too short (less than 5 characters)")
        return False

    # ✅ UPDATED: Check that trained text doesn't contain system/user prompts
    if any(tag in trained_text.lower() for tag in ['<|system|>', '<|user|>', 'you are a bridge engineer']):
        print("\n❌ ERROR: Trained text contains system/user prompts - masking is wrong!")
        return False

    # ✅ Check reasonable masking ratio (should mask most tokens, unmask assistant response)
    masking_ratio = masked_tokens / total_tokens
    if masking_ratio < 0.3 or masking_ratio > 0.95:
        print(f"\n⚠️  WARNING: Unusual masking ratio: {masking_ratio * 100:.1f}%")
        print("    Expected: 60-90% masked (system + user), 10-40% unmasked (assistant)")
        return False

    print("\n✅ Masking looks correct!")
    print(f"   - {masked_tokens} tokens masked (system + user messages)")
    print(f"   - {unmasked_tokens} tokens will be trained (assistant response)")
    print(f"   - Training text: '{trained_text.strip()}'")
    return True


def main():
    # Load data
    print(f"\nLoading data from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} samples")

    # Clean data
    df = df.dropna(subset=['UserInput', 'Response'])
    df['UserInput'] = df['UserInput'].str.strip()
    df['Response'] = df['Response'].str.strip()
    df = df[(df['UserInput'] != '') & (df['Response'] != '')]
    print(f"After cleaning: {len(df)} valid samples")

    # Create formatted conversations
    print("\nFormatting conversations...")
    formatted_data = []
    for _, row in df.iterrows():
        text = create_conversation_text(row['UserInput'], row['Response'])
        formatted_data.append({'text': text})

    # Tokenize with proper masking
    print("\nTokenizing with loss masking...")
    tokenized_data = []
    for example in formatted_data:
        tokenized = tokenize_with_loss_masking(example)
        tokenized_data.append(tokenized)

    # Verify masking on first example
    print("\nVerifying masking on first example...")
    is_correct = verify_masking(tokenized_data[0], formatted_data[0]['text'])

    if not is_correct:
        print("\n❌ MASKING VERIFICATION FAILED!")
        print("Please review the tokenization function before training.")
        return

    # Save tokenized data
    output_file = OUTPUT_DIR / "phi3_training_data_tokenized.json"
    print(f"\nSaving tokenized data to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tokenized_data, f, indent=2)

    print(f"✅ Saved {len(tokenized_data)} tokenized examples")

    # Also save the raw formatted text for reference
    text_output_file = OUTPUT_DIR / "phi3_training_data_text.json"
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved raw text to {text_output_file}")

    # Summary
    print(f"\n{'=' * 80}")
    print("PREPARATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"✅ System message: '{SYSTEM_MESSAGE}'")
    print(f"✅ Training samples: {len(tokenized_data)}")
    print(f"✅ Loss masking: ENABLED (only trains on assistant responses)")
    print(f"✅ Ready for training!")
    print(f"\n📁 Files created:")
    print(f"   - {output_file}")
    print(f"   - {text_output_file}")
    print(f"\n🚀 Next step: Update your training script to use the tokenized data")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()