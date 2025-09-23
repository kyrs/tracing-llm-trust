import sys
import os
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from utils.performance_functions.ethics_dataset import get_ethics_contrastive_dataset
from tqdm import tqdm
from utils.performance_functions.ethics_dataset import ethics_contrastive_loss


BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
CACHE_DIR = os.getenv("HUB_CACHE")
print(f"Using cache directory: {CACHE_DIR}")
if CACHE_DIR:
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

MODEL_NAMES = [..., ...]
results = {}

for model_path in MODEL_NAMES:
    print("=" * 80)
    print(f"Evaluating Model: {model_path}")

    # Load model
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR).to(DEVICE)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        continue
    
    # Load tokenizer for this specific model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset with the correct tokenizer
    print("Preparing dataset...")
    dataset = get_ethics_contrastive_dataset(tokenizer, split="test")
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator
    )

    # Compute loss
    loss = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Ethics Loss"):
            batch_loss = ethics_contrastive_loss(model, tokenizer, batch)    
            loss += batch_loss.item()
    
    avg_loss = loss / len(dataloader)
    results[model_path] = avg_loss
    
    print(f"Average Ethics Loss: {avg_loss:.6f}")
    print()
    
    # Clear GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

# Final summary
print("=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
for model_path, loss in results.items():
    print(f"{model_path}: {loss:.6f}")
print("=" * 80)