
# import os
# import torch
# import pandas as pd
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.llms.huggingface import HuggingFaceLLM
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from transformers import BitsAndBytesConfig
# from tqdm import tqdm
#
# # Configuration
# PDF_PATH = "bridge_defect_report.pdf"
# INPUT_CSV_PATH = "dataset_separated.csv"
# OUTPUT_CSV_PATH = "queries_with_responses.csv"
# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MAX_ROWS = 5  # Set to None to process all rows
#
# os.environ['HF_TOKEN'] = "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"
#
#
# def initialize_components():
#     # Configure 4-bit quantization
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )
#
#     # Initialize LLAMA-3 model with optimized settings
#     llm = HuggingFaceLLM(
#         model_name=LLAMA_MODEL,
#         tokenizer_name=LLAMA_MODEL,
#         context_window=8192,
#         max_new_tokens=100,  # Shorter responses
#         device_map="auto",
#         model_kwargs={
#             "quantization_config": quantization_config,
#             "torch_dtype": torch.float16,
#             "trust_remote_code": True
#         },
#         generate_kwargs={
#             "temperature": 0.2,  # More deterministic
#             "do_sample": True,
#             "repetition_penalty": 1.2  # Reduce verbosity
#         }
#     )
#
#     # Initialize embedding model
#     embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#
#     # Configure chunking settings
#     Settings.node_parser = SentenceSplitter(
#         chunk_size=512,
#         chunk_overlap=100
#     )
#     Settings.embed_model = embed_model
#     Settings.llm = llm
#
#     return llm, embed_model
#
#
# def build_rag_pipeline():
#     # Load PDF documents with verification
#     documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
#     print(f"Loaded {len(documents)} document chunks")
#
#     # Debug: Verify document content
#     if len(documents) == 0:
#         raise ValueError("No documents loaded - check PDF file path")
#     print("\nSample document content:")
#     print(documents[0].text[:200] + "...")
#
#     # Create vector index
#     index = VectorStoreIndex.from_documents(documents)
#
#     # Configure query engine for concise answers
#     query_engine = index.as_query_engine(
#         similarity_top_k=2,  # Fewer but more relevant chunks
#         node_postprocessors=[
#             SimilarityPostprocessor(similarity_cutoff=0.6)  # Higher threshold
#         ],
#         response_mode="compact"  # More focused answers
#     )
#
#     return query_engine
#
#
# def generate_repair_solution(query, query_engine):
#     """Generate a concise repair solution using the RAG pipeline"""
#     # Structured prompt for repair solutions
#     structured_query = (
#         f"Provide only the repair method for this bridge defect in one sentence: {query}\n"
#         "Answer format: 'Repair method: [solution]'"
#     )
#
#     try:
#         response = query_engine.query(structured_query)
#         response_text = str(response).strip()
#
#         # Extract just the solution
#         if "Repair method:" in response_text:
#             return response_text.split("Repair method:")[1].strip()
#         return response_text.split(".")[0] + "."  # Return first complete sentence
#
#     except Exception as e:
#         print(f"Error generating solution: {str(e)}")
#         return "Unable to determine repair method"
#
#
# def process_csv(input_path, output_path, query_engine, max_rows=None):
#     df = pd.read_csv(input_path)
#     if max_rows:
#         df = df.head(max_rows)
#
#     df['generated_response'] = ""
#
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
#         query = row['UserInput']
#
#         # Generate solution
#         solution = generate_repair_solution(query, query_engine)
#         df.at[idx, 'generated_response'] = solution
#
#         # Print progress
#         print(f"\nQuery {idx + 1}: {query[:50]}...")
#         print(f"Solution: {solution}")
#
#     # Save results
#     df.to_csv(output_path, index=False)
#     print(f"\nSaved results to {output_path}")
#
#
# if __name__ == "__main__":
#     print("Initializing components...")
#     llm, embed_model = initialize_components()
#
#     print("Building RAG pipeline...")
#     query_engine = build_rag_pipeline()
#
#     print("\nProcessing queries...")
#     process_csv(
#         input_path=INPUT_CSV_PATH,
#         output_path=OUTPUT_CSV_PATH,
#         query_engine=query_engine,
#         max_rows=MAX_ROWS
#     )
#
# import os
# import torch
# import pandas as pd
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.llms.huggingface import HuggingFaceLLM
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from transformers import BitsAndBytesConfig
# from tqdm import tqdm
# import re
#
# # Configuration
# PDF_PATH = r"..\1_data\bridge_defect_report.pdf"
# INPUT_CSV_PATH = r"..\3_evaluation\bridge_inspection_QA_deepseek.csv"
# OUTPUT_CSV_PATH = "queries_with_responses_deepseek.csv"
# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MAX_ROWS = 3  # Set to None to process all rows
#
# os.environ['HF_TOKEN'] = "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"
#
#
# def initialize_components():
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )
#
#     llm = HuggingFaceLLM(
#         model_name=LLAMA_MODEL,
#         tokenizer_name=LLAMA_MODEL,
#         context_window=8192,
#         max_new_tokens=150,
#         device_map="auto",
#         model_kwargs={
#             "quantization_config": quantization_config,
#             "torch_dtype": torch.float16,
#             "trust_remote_code": True
#         },
#         generate_kwargs={
#             "temperature": 0.2,
#             "do_sample": True,
#             "repetition_penalty": 1.2
#         }
#     )
#
#     embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#
#     Settings.node_parser = SentenceSplitter(
#         chunk_size=512,
#         chunk_overlap=100
#     )
#     Settings.embed_model = embed_model
#     Settings.llm = llm
#
#     return llm, embed_model
#
#
# def build_rag_pipeline():
#     documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
#     print(f"Loaded {len(documents)} document chunks")
#
#     if len(documents) == 0:
#         raise ValueError("No documents loaded - check PDF file path")
#     print("\nSample document content:")
#     print(documents[0].text[:200] + "...")
#
#     index = VectorStoreIndex.from_documents(documents)
#
#     query_engine = index.as_query_engine(
#         similarity_top_k=2,
#         node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
#         response_mode="compact"
#     )
#
#     return query_engine
#
#
# def parse_response(response_text):
#     """Separate the concise repair solution from additional notes"""
#     # Split at the first line break followed by potential whitespace
#     parts = re.split(r'\n\s*\n', response_text.strip(), maxsplit=1)
#
#     if len(parts) == 1:
#         # If no clear separation, try to split at first sentence
#         sentences = re.split(r'(?<=[.!?])\s+', response_text.strip(), maxsplit=1)
#         if len(sentences) > 1:
#             return sentences[0], sentences[1]
#         return response_text, ""
#
#     # Clean up the parts
#     solution = parts[0].strip()
#     notes = parts[1].strip() if len(parts) > 1 else ""
#
#     return solution, notes
#
#
# def generate_repair_solution(query, query_engine):
#     """Generate a repair solution using the RAG pipeline"""
#     structured_query = (
#         f"Provide the repair method for this bridge defect in one concise paragraph, "
#         f"followed by any additional notes: {query}"
#     )
#
#     try:
#         response = query_engine.query(structured_query)
#         return str(response).strip()
#     except Exception as e:
#         print(f"Error generating solution: {str(e)}")
#         return "Unable to determine repair method", "Error occurred"
#
#
# def process_csv(input_path, output_path, query_engine, max_rows=None):
#     df = pd.read_csv(input_path)
#     if max_rows:
#         df = df.head(max_rows)
#
#     df['generated_response'] = ""
#     df['Additional_Notes'] = ""
#
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
#         query = row['UserInput']
#
#         response_text = generate_repair_solution(query, query_engine)
#         solution, notes = parse_response(response_text)
#
#         df.at[idx, 'generated_response'] = solution
#         df.at[idx, 'Additional_Notes'] = notes
#
#         print(f"\nQuery {idx + 1}: {query[:50]}...")
#         print(f"Solution: {solution}")
#         print(f"Notes: {notes}")
#
#     df.to_csv(output_path, index=False)
#     print(f"\nSaved results to {output_path}")
#
#
# if __name__ == "__main__":
#     print("Initializing components...")
#     llm, embed_model = initialize_components()
#
#     print("Building RAG pipeline...")
#     query_engine = build_rag_pipeline()
#
#     print("\nProcessing queries...")
#     process_csv(
#         input_path=INPUT_CSV_PATH,
#         output_path=OUTPUT_CSV_PATH,
#         query_engine=query_engine,
#         max_rows=MAX_ROWS
#     )

#
# import os
# import torch
# import pandas as pd
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.llms.huggingface import HuggingFaceLLM
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from transformers import BitsAndBytesConfig
# from tqdm import tqdm
# import re
#
# # Configuration
# PDF_PATH = r"..\1_data\bridge_defect_report.pdf"
# INPUT_CSV_PATH = r"..\3_evaluation\bridge_inspection_QA6.csv"
# OUTPUT_CSV_PATH = "queries_with_responses_deepseek6.csv"
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MAX_ROWS = 3 # Set to None to process all rows
#
# os.environ['HF_TOKEN'] = "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"
#
# def initialize_components():
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )
#
#     llm = HuggingFaceLLM(
#         model_name=LLAMA_MODEL,
#         tokenizer_name=LLAMA_MODEL,
#         context_window=8192,
#         max_new_tokens=150,
#         device_map="auto",
#         model_kwargs={
#             "quantization_config": quantization_config,
#             "torch_dtype": torch.float16,
#             "trust_remote_code": True
#         },
#         generate_kwargs={
#             "temperature": 0.2,
#             "do_sample": True,
#             "repetition_penalty": 1.2
#         }
#     )
#
#     embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#
#     Settings.node_parser = SentenceSplitter(
#         chunk_size=512,
#         chunk_overlap=100
#     )
#     Settings.embed_model = embed_model
#     Settings.llm = llm
#
#     return llm, embed_model
#
# def build_rag_pipeline():
#     documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
#     print(f"Loaded {len(documents)} document chunks")
#
#     if len(documents) == 0:
#         raise ValueError("No documents loaded - check PDF file path")
#     print("\nSample document content:")
#     print(documents[0].text[:200] + "...")
#
#     index = VectorStoreIndex.from_documents(documents)
#
#     query_engine = index.as_query_engine(
#         similarity_top_k=2,
#         node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
#         response_mode="compact"
#     )
#
#     return query_engine
#
# def parse_response(response_text):
#     """Separate the concise repair solution from additional notes"""
#     # Split at the first line break followed by potential whitespace
#     parts = re.split(r'\n\s*\n', response_text.strip(), maxsplit=1)
#
#     if len(parts) == 1:
#         # If no clear separation, try to split at first sentence
#         sentences = re.split(r'(?<=[.!?])\s+', response_text.strip(), maxsplit=1)
#         if len(sentences) > 1:
#             return sentences[0], sentences[1]
#         return response_text, ""
#
#     # Clean up the parts
#     solution = parts[0].strip()
#     notes = parts[1].strip() if len(parts) > 1 else ""
#
#     return solution, notes
#
# def generate_repair_solution(query, query_engine):
#     """Generate a repair solution using the RAG pipeline"""
#     structured_query = (
#         f"Provide the repair method for this bridge defect in one concise paragraph, "
#         f"followed by any additional notes: {query}"
#     )
#
#     try:
#         response = query_engine.query(structured_query)
#         return str(response).strip()
#     except Exception as e:
#         print(f"Error generating solution: {str(e)}")
#         return "Unable to determine repair method", "Error occurred"
#
# def process_csv(input_path, output_path, query_engine, max_rows=None):
#     df = pd.read_csv(input_path)
#     if max_rows:
#         df = df.head(max_rows)
#
#     df['generated_response'] = ""
#
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
#         query = row['UserInput']
#
#         response_text = generate_repair_solution(query, query_engine)
#         solution, _ = parse_response(response_text)  # We only keep the solution now
#
#         df.at[idx, 'generated_response'] = solution
#
#         print(f"\nQuery {idx + 1}: {query[:50]}...")
#         print(f"Solution: {solution}")
#
#     # Only keep the columns we want in the output
#     output_df = df[['defect_count', 'UserInput', 'Response', 'generated_response']]
#     output_df.to_csv(output_path, index=False)
#     print(f"\nSaved results to {output_path}")
#
# if __name__ == "__main__":
#     print("Initializing components...")
#     llm, embed_model = initialize_components()
#
#     print("Building RAG pipeline...")
#     query_engine = build_rag_pipeline()
#
#     print("\nProcessing queries...")
#     process_csv(
#         input_path=INPUT_CSV_PATH,
#         output_path=OUTPUT_CSV_PATH,
#         query_engine=query_engine,
#         max_rows=MAX_ROWS
#     )
#
#
# import os
# import torch
# import pandas as pd
# import re
# from pathlib import Path
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from transformers import BitsAndBytesConfig
# from tqdm import tqdm
#
# # Configuration
# PDF_PATH = r"..\GYData\defect_report_llama_v2.pdf"
# INPUT_CSV = r"..\GYData\separated_llama3_response_v2.csv"
# OUTPUT_CSV = "LlamaNLlama_structured_v2.csv"
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MAX_ROWS = 2 # Set to None to process all rows
# # "BAAI/bge-small-en-v1.5"
#
# os.environ['HF_TOKEN'] = "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"
# # SciBERT
# # nvidia/NV-Embed-v2
# def initialize_components():
#     """Initialize models with optimized settings"""
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#     )
#
#     llm = HuggingFaceLLM(
#         model_name=LLAMA_MODEL,
#         tokenizer_name=LLAMA_MODEL,
#         system_prompt="You are a bridge engineering assistant providing precise repair solutions./"
#                       "you are provided with historical bridge inspection records/"
#                       "you should recommend repair proposal based on these historical records",
#         # system_prompt=(
#         #     "You are a bridge engineering expert providing concise repair solutions. "
#         #     "Only provide the technical repair method in 1 sentence. "
#         #     "If referencing a Finding, format as: 'Finding [X]: [solution]'"
#         # ),
#         context_window=8192,
#         max_new_tokens=256,  # Strict length limit
#         device_map="auto",
#         model_kwargs={
#             "quantization_config": quantization_config,
#             "torch_dtype": torch.float16,
#             "trust_remote_code": True
#         },
#         generate_kwargs={
#             "temperature": 0.3,  # More deterministic
#             "do_sample": False,
#
#         }
#     )
#
#     embed_model = HuggingFaceEmbedding(
#         model_name=EMBED_MODEL,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
#
#     Settings.node_parser = SentenceSplitter(
#         chunk_size=300,
#         chunk_overlap=50,
#         paragraph_separator = "\nFinding ",  # Critical for your document structure
#         secondary_chunking_regex = r"Finding \d+:"  # Split at each Finding
#     )
#     Settings.embed_model = embed_model
#     Settings.llm = llm
#
#     return llm, embed_model
#
#
# def load_pdf_documents():
#     """Load PDF with validation"""
#     print(f"Loading PDF from: {PDF_PATH}")
#     if not PDF_PATH.exists():
#         raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
#
#     return SimpleDirectoryReader(input_files=[str(PDF_PATH)]).load_data()
#
#
# def build_rag_pipeline():
#     """Build optimized RAG pipeline"""
#     documents = load_pdf_documents()
#     index = VectorStoreIndex.from_documents(documents)
#
#     return index.as_query_engine(
#         similarity_top_k=4,
#         node_postprocessors=[
#             SimilarityPostprocessor(similarity_cutoff=0.7)
#         ],
#         response_mode="compact"
#     )
#
#
# def generate_repair_solution(query, query_engine):
#     """Generate standardized repair solution"""
#     prompt = f"""
#     DEFECT: {query}
#
#     REQUIREMENTS:
#     1. Provide ONLY the technical repair method
#     2. Maximum 1 sentence
#     3. No explanations or disclaimers
#     4. If matching a Finding, use format: "Finding [X]: [solution]"
#     """
#
#     try:
#         response = query_engine.query(prompt)
#         return str(response).strip()
#     except Exception as e:
#         print(f"Error processing query: {str(e)}")
#         return "Error: Could not generate solution"
#
#
# def clean_response(response_text):
#     """Enforce standardized output format"""
#     # Remove any disclaimers/analysis
#     clean_text = re.split(r'(Note:|However|Analysis:|Please)', response_text)[0]
#
#     # Extract first complete sentence
#     sentences = re.findall(r'(.*?[.!?])', clean_text.strip())
#     if sentences:
#         solution = sentences[0]
#     else:
#         solution = clean_text.strip()
#
#     # Standardize Finding references
#     if "Finding" in solution:
#         solution = re.sub(r'Finding (\d+)[:\-]?\s*', r'Finding \1: ', solution)
#
#     return solution
#
#
# def process_csv(input_path, output_path, query_engine, max_rows=None):
#     """Process CSV with strict output control"""
#     df = pd.read_csv(input_path)
#     if max_rows:
#         df = df.head(max_rows)
#
#     df['Standardized_Response'] = ""
#
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         query = str(row['UserInput']).strip()
#         if not query:
#             df.at[idx, 'Standardized_Response'] = "Empty query"
#             continue
#
#         try:
#             # Generate and clean response
#             raw_response = generate_repair_solution(query, query_engine)
#             final_response = clean_response(raw_response)
#
#             df.at[idx, 'Standardized_Response'] = final_response
#
#             # Debug output
#             print(f"\nQuery {idx + 1}: {query[:60]}...")
#             print(f"Response: {final_response}")
#
#         except Exception as e:
#             print(f"Failed on row {idx}: {str(e)}")
#             df.at[idx, 'Standardized_Response'] = f"Error: {str(e)}"
#
#     # Save only essential columns
#     output_cols = ['UserInput', 'Response', 'Standardized_Response']
#     df[output_cols].to_csv(output_path, index=False)
#     print(f"\nSaved {len(df)} standardized responses to {output_path}")
#
#
# if __name__ == "__main__":
#     try:
#         print("Initializing components...")
#         llm, embed_model = initialize_components()
#
#         print("Building RAG pipeline...")
#         query_engine = build_rag_pipeline()
#
#         print("\nProcessing CSV with strict formatting...")
#         process_csv(
#             input_path=INPUT_CSV_PATH,
#             output_path=OUTPUT_CSV_PATH,
#             query_engine=query_engine,
#             max_rows=MAX_ROWS
#         )
#     except Exception as e:
#         print(f"\nFatal error: {str(e)}")
#

import os
import torch
import pandas as pd
import re
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
from tqdm import tqdm

# Configuration

# defect_report_separated_llama3_response_v3.pdf
# defect_report_separated_gpt4_response_counted_v2.pdf
PDF_PATH = Path(r"..\GYData\defect_report_separated_gpt4_response_counted_v2.pdf")
# INPUT_CSV_PATH = Path(r"..\GYData\question_sets\question_set_b_linguistic_robustness.csv")
INPUT_CSV_PATH = Path(r"C:\Users\Gx\Desktop\FineTune\GYData\question_sets_llama_generalisation\question_set_d_test_rag.csv")
OUTPUT_CSV_PATH = Path("generalisation_eva/defect_report_separated_gpt4_response_counted_v2_question_set_d_test_rag_UAE.csv")
EMBED_MODEL = "WhereIsAI/UAE-Large-V1"
# intfloat/e5-large-v2


# BAAI/bge-small-en-v1.5
# sentence-transformers/all-mpnet-base-v2
# allenai/scibert_scivocab_uncased
# Marqo/dunzhang-stella_en_400M_v5
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_ROWS = None  # Set to None to process all rows

os.environ['HF_TOKEN'] = "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"


def initialize_components():
    """Initialize models with optimized settings"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    llm = HuggingFaceLLM(
        model_name=LLAMA_MODEL,
        tokenizer_name=LLAMA_MODEL,
        system_prompt="You are a bridge engineering assistant providing precise repair solutions./"
                      "you are provided with historical bridge inspection records/"
                      "you should recommend repair proposal based on these historical records",
        context_window=8192,
        max_new_tokens=256,
        device_map="auto",
        model_kwargs={
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "trust_remote_code": True
        },
        generate_kwargs={
            "temperature": 0.3,
            "do_sample": False,
        }
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_folder="./.cache"
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50,
        paragraph_separator="\nFinding ",
        secondary_chunking_regex=r"Finding \d+:"
    )
    Settings.embed_model = embed_model
    Settings.llm = llm

    return llm, embed_model


def load_pdf_documents():
    """Load PDF with validation"""
    print(f"Loading PDF from: {PDF_PATH}")
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    return SimpleDirectoryReader(input_files=[str(PDF_PATH)]).load_data()


def build_rag_pipeline():
    """Build optimized RAG pipeline"""
    documents = load_pdf_documents()
    index = VectorStoreIndex.from_documents(documents)

    return index.as_query_engine(
        similarity_top_k=4,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ],
        response_mode="compact"
    )


def generate_repair_solution(query, query_engine):
    """Generate standardized repair solution"""
    prompt = f"""
    DEFECT: {query}

    REQUIREMENTS:
    1. Provide ONLY the technical repair method
    2. Maximum 1 sentence
    3. No explanations or disclaimers
    4. If matching a Finding, use format: "Finding [X]: [solution]"
    """

    try:
        response = query_engine.query(prompt)
        return str(response).strip()
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return "Error: Could not generate solution"


def clean_response(response_text):
    """Enforce standardized output format"""
    clean_text = re.split(r'(Note:|However|Analysis:|Please)', response_text)[0]
    sentences = re.findall(r'(.*?[.!?])', clean_text.strip())
    if sentences:
        solution = sentences[0]
    else:
        solution = clean_text.strip()

    if "Finding" in solution:
        solution = re.sub(r'Finding (\d+)[:\-]?\s*', r'Finding \1: ', solution)

    return solution


def process_csv(input_path, output_path, query_engine, max_rows=None):
    """Process CSV with strict output control"""
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if max_rows:
        df = df.head(max_rows)

    # Check if Cluster_ID exists, if not create empty column
    if 'Cluster_ID' not in df.columns:
        df['Cluster_ID'] = ""

    df['Standardized_Response'] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        query = str(row['UserInput']).strip()
        if not query:
            df.at[idx, 'Standardized_Response'] = "Empty query"
            continue

        try:
            raw_response = generate_repair_solution(query, query_engine)
            final_response = clean_response(raw_response)
            df.at[idx, 'Standardized_Response'] = final_response

            print(f"\nQuery {idx + 1}: {query[:60]}...")
            print(f"Response: {final_response}")

        except Exception as e:
            print(f"Failed on row {idx}: {str(e)}")
            df.at[idx, 'Standardized_Response'] = f"Error: {str(e)}"

    # Save with Cluster_ID included
    output_cols = ['Cluster_ID', 'UserInput', 'Response', 'Standardized_Response']
    df[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} standardized responses to {output_path}")


if __name__ == "__main__":
    try:
        print("Initializing components...")
        llm, embed_model = initialize_components()

        print("Building RAG pipeline...")
        query_engine = build_rag_pipeline()

        print("\nProcessing CSV with strict formatting...")
        process_csv(
            input_path=INPUT_CSV_PATH,
            output_path=OUTPUT_CSV_PATH,
            query_engine=query_engine,
            max_rows=MAX_ROWS
        )
    except Exception as e:
        print(f"\nFatal error: {str(e)}")