import sys
import os
from typing import Dict
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)

from utils.performance_functions.truth_dataset import get_truthfulness_contrastive_dataset, truthfulness_contrastive_loss
BATCH = Dict[str, torch.Tensor]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
CACHE_DIR = os.getenv("HUB_CACHE")
print(f"Using cache directory: {CACHE_DIR}")
if CACHE_DIR:
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

MODEL_NAMES = [..., ...]

BATCH_SIZE = 8
results = {}
for MODEL_NAME in MODEL_NAMES:
    print("model_name", MODEL_NAME)
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()

    # Load model with memory optimizations
    try:
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto"  # Automatically handle device placement
        )
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        continue

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_truthfulness_contrastive_dataset(tokenizer, split = 'test')
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator
    )

    loss = 0

    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_loss = truthfulness_contrastive_loss(model=model, batch=batch)
            loss += batch_loss.item()  # Convert to Python scalar to save memory

    results[MODEL_NAME] = loss/len(dataloader)

# Final summary
print("=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
for model_path, loss in results.items():
    print(f"{model_path}: {loss:.6f}")
print("=" * 80)