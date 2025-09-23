from typing import Optional
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ── Dataset ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def get_anthropic_dataset(tokenizer, split: str, indices=None, max_tokens: Optional[int] = None): # max_tokens defaults to 512 inside the function.
    """
    This will take the argument of max_tokens to return only the training samples that actually are used in training.
    """
    if split not in ['train', 'test']:
        raise Exception ('split must be either train or test.')
    raw = load_dataset("Dahoas/static-hh", split=split)       # prompt / chosen / rejected …

    if max_tokens is not None:
        original_size = len(raw)

        def is_within_max_tokens(ex):
            full_text = ex["prompt"] + ex["chosen"] + tokenizer.eos_token
            return len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) <= max_tokens

        raw = raw.filter(is_within_max_tokens, num_proc=4, desc=f"Filtering out samples > {max_tokens} tokens")
        # filtering is done by the basis of prompt + chosen.
        new_size = len(raw)
        if original_size > 0:
            percentage_excluded = (original_size - new_size) / original_size * 100
            print(f"{BColors.WARNING}Filtered out {original_size - new_size} samples ({percentage_excluded:.2f}%) that were longer than {max_tokens} tokens.{BColors.ENDC}")

    def tokenize(ex):
        prompt      = ex["prompt"]            
        completion  = ex["chosen"] + tokenizer.eos_token
        full_text   = prompt + completion

        current_max_length = max_tokens if max_tokens is not None else 1024
        tokens = tokenizer(
            full_text,
            max_length=current_max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False           # ← ­important
        )

        labels = tokens["input_ids"].copy()

        # mask the prompt
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=current_max_length,
            truncation=True
        )["input_ids"]
        
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        # mask the pads
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        tokens["labels"] = labels

        '''
        returns a list of dict with keys --> dict_keys(['input_ids', 'attention_mask', 'labels']).
        
        '''
        return tokens

    ds = raw.map(tokenize,
                 remove_columns=raw.column_names,
                 batched=False,
                 desc="tokenising HH chosen answers")

    if indices is not None:
        ds = ds.select(indices)

    return ds  # dict(input_ids, attention_mask, labels)

# ── Evaluation loop ────────────────────────────────────────────────────────────
def compute_mean_loss(model: nn.Module, dataloader: DataLoader, DEVICE: str) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss  = model(**batch).loss 
            losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("nan") 

# ── Evaluation loop ────────────────────────────────────────────────────────────
def compute_batch_loss(model: nn.Module, batch: dict, DEVICE: str, requires_grad: bool = False):
    model.eval()
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        loss = model(**batch).loss
    return loss if requires_grad else loss.item()