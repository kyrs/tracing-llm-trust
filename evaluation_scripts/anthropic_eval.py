import argparse
import os
import sys
from typing import Dict
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dotenv import load_dotenv
import torch
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)

from utils.performance_functions.anthropic import get_anthropic_dataset, compute_mean_loss


# Load environment variables from .env file
load_dotenv()
hub_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
cache_dir = os.getenv("HUB_CACHE")
if cache_dir:
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = Dict[str, torch.Tensor]
BATCH_SIZE = 8
losses = {}
def main():
    MODEL_NAMES = [..., ...]

    if not MODEL_NAMES:
        print(colored("No models found in the config file.", "red"))
        return

    for model_name in MODEL_NAMES:
        print(colored(f"Evaluating model: {model_name}", 'cyan', attrs=['bold']))
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, token=hub_token, cache_dir=cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            dataset = get_anthropic_dataset(tokenizer, split='test').shuffle(seed=42)
            dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator
            )
            
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=hub_token, cache_dir=cache_dir)
            model.to(DEVICE)
            
            mean_loss = compute_mean_loss(model, dataloader, DEVICE=DEVICE)
            losses[model_name] = mean_loss
            print(colored(f"  Mean token-level cross-entropy loss on Anthropic: {mean_loss:.4f}\n", 'yellow'))
            del model
            del dataloader
            del dataset
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(colored(f"  Failed to evaluate model {model_name}. Error: {e}\n", 'red'))

    print(colored(losses, 'green'))

if __name__ == "__main__":
    main()