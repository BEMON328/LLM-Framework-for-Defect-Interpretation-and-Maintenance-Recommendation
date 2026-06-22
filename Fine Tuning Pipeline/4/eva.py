import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import re
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation.log'
)

# Configuration
data_path = "scored_results_test.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # Faster alternative to BAAI/bge-small-en-v1.5

# Domain-specific configuration (must come before metric functions)
REQUIRED_ACTIONS = ["replace", "weld", "repair", "bolt", "cut", "fill"]
DEFECT_COMPONENTS = ["flange", "web", "gusset", "bearing", "diaphragm", "abutment"]
SEEN_COMPONENTS = ["girder", "diaphragm"]  # Components in training data
UNSEEN_COMPONENTS = ["bearing", "abutment"]  # Held-out components


# ================= METRIC CALCULATIONS =================
def evaluate_correctness(ground_truth, generated):
    """Binary evaluation of critical action match"""
    gt_actions = set(a for a in REQUIRED_ACTIONS if a in ground_truth.lower())
    gen_actions = set(a for a in REQUIRED_ACTIONS if a in generated.lower())
    return int(gt_actions == gen_actions)


def evaluate_completeness(ground_truth, generated):
    """Measure if all defect aspects are addressed"""
    gt_components = set(c for c in DEFECT_COMPONENTS if c in ground_truth.lower())
    gen_components = set(c for c in DEFECT_COMPONENTS if c in generated.lower())
    return len(gt_components & gen_components) / max(1, len(gt_components))


def evaluate_robustness(df_group):
    """Consistency across query variants (grouped by defect type)"""
    if len(df_group) < 2:  # Need at least 2 queries per defect for robustness
        return np.nan
    unique_answers = df_group["generated"].nunique()
    return 1 - (unique_answers / len(df_group))


# ================= CORE FUNCTIONS =================
def log_hardware_info():
    """Log system hardware information"""
    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"CPU cores: {os.cpu_count()}")
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")


def load_data(filepath):
    """Load and preprocess the dataset with progress tracking"""
    try:
        logging.info(f"Loading dataset from: {Path(filepath).absolute()}")
        df = pd.read_csv(filepath, header=None,
                         names=["query", "ground_truth", "generated", "similarity"])

        # Extract defect information
        df["defect_type"] = df["query"].str.extract(
            r'(crack|corrosion|section loss|hole)', flags=re.IGNORECASE)
        df["component"] = df["query"].str.extract(
            r'(girder|bearing|abutment|diaphragm)', flags=re.IGNORECASE)

        logging.info(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        raise


def initialize_model():
    """Initialize the embedding model with error handling"""
    try:
        logging.info(f"Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        if torch.cuda.is_available():
            model = model.to('cuda')
            logging.info("Model moved to GPU")
        return model
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        raise


def calculate_similarities(model, df):
    """Calculate semantic similarities with progress tracking"""
    try:
        logging.info("Starting similarity calculations...")

        # Process in batches to monitor progress
        batch_size = 32
        similarities = []

        for i in tqdm(range(0, len(df), batch_size), desc="Calculating similarities"):
            batch = df.iloc[i:i + batch_size]
            gt_embs = model.encode(
                batch["ground_truth"].tolist(),
                convert_to_tensor=True,
                show_progress_bar=False
            )
            gen_embs = model.encode(
                batch["generated"].tolist(),
                convert_to_tensor=True,
                show_progress_bar=False
            )
            batch_sims = cosine_similarity(
                gt_embs.cpu().numpy() if torch.cuda.is_available() else gt_embs.numpy(),
                gen_embs.cpu().numpy() if torch.cuda.is_available() else gen_embs.numpy()
            ).diagonal()
            similarities.extend(batch_sims)

        return similarities
    except Exception as e:
        logging.error(f"Similarity calculation failed: {str(e)}")
        raise


def run_evaluation(data_path):
    """Main evaluation pipeline with enhanced logging"""
    try:
        log_hardware_info()
        df = load_data(data_path)
        model = initialize_model()

        # Calculate similarities if missing
        if "similarity" not in df.columns or df["similarity"].isna().all():
            df["similarity"] = calculate_similarities(model, df)

        # Calculate metrics
        df["correctness"] = df.apply(lambda x: evaluate_correctness(x["ground_truth"], x["generated"]), axis=1)
        df["completeness"] = df.apply(lambda x: evaluate_completeness(x["ground_truth"], x["generated"]), axis=1)

        # Grouped metrics
        robustness = df.groupby("defect_type", group_keys=False).apply(evaluate_robustness).mean()
        generalization = {
            "seen_acc": df[df["component"].isin(SEEN_COMPONENTS)]["correctness"].mean(),
            "unseen_acc": df[df["component"].isin(UNSEEN_COMPONENTS)]["correctness"].mean()
        }
        generalization["transfer_rate"] = (
            generalization["unseen_acc"] / generalization["seen_acc"]
            if generalization["seen_acc"] > 0 else np.nan
        )

        return df, {
            "Correctness": f"{df['correctness'].mean() * 100:.1f}%",
            "Completeness": f"{df['completeness'].mean() * 100:.1f}%",
            "Robustness": f"{robustness * 100:.1f}%" if not np.isnan(robustness) else "N/A",
            "Generalization": f"{generalization['transfer_rate'] * 100:.1f}%",
            "Similarity": f"{df['similarity'].mean() * 100:.1f}%"
        }
    except Exception as e:
        logging.error(f"Evaluation pipeline failed: {str(e)}")
        raise


# ================= VISUALIZATION =================
def visualize_results(df, results):
    """Generate evaluation visualizations"""
    try:
        # Metrics table
        print("\n📊 Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric:<15}: {value}")

        # Error analysis
        errors = df[df["correctness"] == 0]
        if not errors.empty:
            print("\n🔍 Top Errors Analysis:")
            for idx, row in errors.head(3).iterrows():
                print(f"\nQUERY: {row['query']}")
                print(f"EXPECTED: {row['ground_truth']}")
                print(f"GENERATED: {row['generated']}")
                print(f"SIMILARITY: {row['similarity']:.2f}")
        else:
            print("\n🎉 No incorrect recommendations found!")

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        df.groupby("defect_type")["correctness"].mean().sort_values().plot(kind="barh")
        plt.title("Correctness by Defect Type")
        plt.xlabel("Accuracy")

        plt.subplot(1, 2, 2)
        plt.scatter(df["completeness"], df["similarity"], alpha=0.5)
        plt.xlabel("Completeness")
        plt.ylabel("Similarity")
        plt.title("Response Quality")

        plt.tight_layout()
        plt.savefig("evaluation_metrics.png")
        print("\n📈 Visualizations saved to evaluation_metrics.png")

    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        raise


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    try:
        print("🚀 Starting bridge maintenance evaluation")
        df, results = run_evaluation(data_path)
        visualize_results(df, results)
        print("\n✅ Evaluation completed successfully")
        print("💡 Check evaluation.log for detailed processing information")
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        print("⚠️ Check evaluation.log for error details")