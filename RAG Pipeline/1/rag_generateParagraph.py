import pandas as pd
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Configuration
HUGGING_FACE_TOKEN = "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"
os.environ["HUGGINGFACE_TOKEN"] = HUGGING_FACE_TOKEN

INPUT_CSV = r"C:\Users\Gx\Desktop\FineTune\GYData\question_sets_llama_generalisation\question_set_d_train_rag.csv"
OUTPUT_PDF = "report_separated_llama_response_v2_llama_generalisation.pdf"
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # Fixed model name
MAX_ROWS = None  # Set to None to process all rows


def load_model():
    """Load Llama 3 with authentication"""
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL,
        token=HUGGING_FACE_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HUGGING_FACE_TOKEN
    )
    return tokenizer, model


def generate_inspection_paragraph(defect, proposal, tokenizer, model):
    """Generate professional inspection paragraph"""
    prompt = f"""
    As a senior bridge engineer, describe this inspection finding in two concise sentences:
    1. First sentence describes the defect (include location if specified)
    2. Second sentence states the repair proposal

    Defect: {defect}
    Proposal: {proposal}

    Example Output:
    During a routine bridge inspection, it was identified that there are missing rivets on the main girder connection plates.
    The recommended solution is to install new high-strength bolts.

    Your Response:
    """

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,  # Reduced for more concise output
        temperature=0.3,  # Lower for more consistency
        top_p=0.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    paragraph = full_text.split("Your Response:")[1].strip()

    # Ensure proper sentence structure
    sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
    if len(sentences) >= 2:
        return f"{sentences[0]}. {sentences[1]}."
    elif sentences:
        return f"{sentences[0]}."  # Fallback if only one sentence
    else:
        return f"Observed {defect.lower()}. Recommended action: {proposal.lower()}."  # Final fallback


def create_pdf_report(paragraphs):
    """Generate professional PDF report"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Bridge Inspection Findings Report", ln=True, align='C')
    pdf.ln(10)

    # Findings
    pdf.set_font("Arial", size=12)
    for idx, para in enumerate(paragraphs, 1):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Finding {idx}:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, para)
        pdf.ln(5)

    pdf.output(OUTPUT_PDF)


def main():
    print("Initializing model...")
    tokenizer, model = load_model()

    print(f"Loading data from {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        if MAX_ROWS is not None:
            df = df.head(MAX_ROWS)
            print(f"Processing first {MAX_ROWS} records")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return

    print("Generating findings...")
    paragraphs = []
    for idx, row in df.iterrows():
        try:
            defect = str(row['defect']).strip()
            proposal = str(row['proposal']).strip()

            if not defect or not proposal:
                paragraphs.append("Error: Missing defect or proposal data")
                continue

            paragraph = generate_inspection_paragraph(defect, proposal, tokenizer, model)
            paragraphs.append(paragraph)
            print(f"\nFinding {idx + 1}:")
            print(paragraph)
        except Exception as e:
            print(f"Error processing row {idx + 1}: {str(e)}")
            paragraphs.append(f"Error generating description for: {row['defect']}")

    print("\nGenerating PDF...")
    create_pdf_report(paragraphs)
    print(f"\nReport saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()