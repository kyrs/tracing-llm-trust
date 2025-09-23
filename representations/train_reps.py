import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

# Device placement – prefer GPU when available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_CACHE = os.getenv("HUB_CACHE")

# Model / dataset
MODEL_NAME = ... # hf_username/model_name
DATASET_NAME = "Dahoas/static-hh"
SPLIT = "train"

# Representation parameters
MAX_LENGTH = 1024
BATCH_SIZE = 16
DTYPE = torch.float32

SAVE_PATH = f"{MODEL_NAME.split('/')[-1]}_sumreps.pt"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def aggregate_hidden(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Expand mask to match hidden size
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)  # (batch, seq_len, 1)
    token_sum = (hidden_state * mask).sum(dim=1)               # (batch, hidden)
    token_count = mask.sum(dim=1)                               # (batch, 1)
    # Avoid division by zero just in case
    token_count = torch.clamp(token_count, min=1.0)
    return token_sum / token_count


# ─── Load model & tokenizer ────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=HF_CACHE,
)
# Ensure pad token exists for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=HF_CACHE,
    torch_dtype=DTYPE,
    device_map="auto",  # dispatch layers across available GPUs/CPU automatically
)

print(model)
model.eval()  # inference mode

# ─── Load dataset ──────────────────────────────────────────────────────────────

dataset = load_dataset(DATASET_NAME, split=SPLIT)
num_samples = len(dataset)
print(f"Loaded {num_samples} samples from '{DATASET_NAME}' split '{SPLIT}'.")

# ─── Representation extraction ────────────────────────────────────────────────

all_reps: List[torch.Tensor] = []
prompts: List[str] = []
completions: List[str] = []
for start_idx in tqdm(range(0, num_samples, BATCH_SIZE), desc="Batches"):
    batch = dataset[start_idx : start_idx + BATCH_SIZE]

    # Build the raw text – concatenate prompt + completion as in SFT script
    raw_texts = [p + " " + c for p, c in zip(batch["prompt"], batch["chosen"])]

    # Tokenise
    encodings = tokenizer(
        raw_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    from termcolor import colored
    with torch.no_grad():
        outputs = model(**encodings, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # shape (batch, seq_len, hidden)
        reps = aggregate_hidden(last_hidden, encodings["attention_mask"])  # (batch, hidden)
    
        # Move to CPU (fp32) to free GPU quickly
        reps = reps.float().cpu()
    if torch.isnan(reps).any():
        print(colored(torch.isnan(reps).any(), "red"))
        raise Exception("NaN in reps")
    if torch.isinf(reps).any():
        print(colored(torch.isinf(reps).any(), "red"))
        raise Exception("Inf in reps")

    all_reps.append(reps)
    prompts.extend(batch["prompt"])
    completions.extend(batch["chosen"])

# ─── Save representations tensor ──────────────────────────────────────────────

rep_tensor = torch.cat(all_reps, dim=0)  # (N, hidden)
assert rep_tensor.shape[0] == num_samples, "Mismatch in representation count!"

# Save everything in a single file (dictionary)
data_dict = {
    "representations": rep_tensor,  # tensor of shape (N, hidden)
    "prompts": prompts,            # list[str]
    "completions": completions,    # list[str]
}

torch.save(data_dict, SAVE_PATH)
print(f"Saved representations and texts to single file: {SAVE_PATH}")