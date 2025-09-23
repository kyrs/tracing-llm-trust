import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)
from utils.performance_functions.bias_dataset import get_bias_agreement_dataset, bias_agreement_nll_loss
from typing import Dict
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()
hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
cache_dir = os.getenv("HUB_CACHE")
print(f"Using cache directory: {cache_dir}")
if cache_dir:
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = Dict[str, torch.Tensor]

model_names = [..., ...]

BATCH_SIZE = 8

results = {}

for MODEL_NAME in model_names:
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()

    # Load model with memory optimizations
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            device_map="auto"  # Automatically handle device placement
        )
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        continue
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_bias_agreement_dataset(tokenizer, split='test')
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator
    )

    loss = 0

    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_loss = bias_agreement_nll_loss(model=model, tokenizer=tokenizer, batch=batch)
            loss += batch_loss.item()  # Convert to Python scalar to save memory

    print(colored(f"{MODEL_NAME}: {loss / len(dataloader):.4f}", 'yellow'))
    results[MODEL_NAME] = loss / len(dataloader)

# Final summary
print("=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
for model_path, loss in results.items():
    print(f"{model_path}: {loss:.6f}")
print("=" * 80)