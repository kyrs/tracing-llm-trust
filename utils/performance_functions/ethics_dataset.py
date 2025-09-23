from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Dataset paths -------------------------------------------------------------
# ---------------------------------------------------------------------------
# Directory that contains the pre-generated splits (train / test)
SPLIT_DIR = Path("../../data/ethics")

# Mapping from user-facing ``split`` names to the corresponding CSV files.
SPLIT_CSV_FILES = {
    "train": SPLIT_DIR / "cm_train.csv",
    "test": SPLIT_DIR / "cm_test.csv",
}

# ---------------------------------------------------------------------------
# Dataset builder -----------------------------------------------------------
# ---------------------------------------------------------------------------

def get_ethics_dataset(
    tokenizer: PreTrainedTokenizer,
    *,
    split: str,
    max_length: int = 256,
    balance: bool = True,
) -> Dataset:

    # ------------------------------------------------------------------
    # 1. Load and balance the dataset -----------------------------------
    # ------------------------------------------------------------------
    data_file = SPLIT_CSV_FILES.get(split)
    if data_file is None or not data_file.exists():
        raise ValueError(f"Split file {data_file} not found")

    ds = load_dataset("csv", data_files=str(data_file), split="train")

    # Keep only short scenarios
    ds = ds.filter(lambda ex: ex["is_short"] in [True, "True", 1, "1"])

    ds_zeros = ds.filter(lambda ex: int(ex["label"]) == 0)
    ds_ones = ds.filter(lambda ex: int(ex["label"]) == 1)

    if balance:
        # Separate by label and downsample the majority class to balance
        min_size = min(len(ds_zeros), len(ds_ones))
        ds_zeros = ds_zeros.shuffle(seed=42).select(range(min_size))
        ds_ones = ds_ones.shuffle(seed=42).select(range(min_size))

    # Create a new dataset of paired statements
    paired_ds = Dataset.from_dict({
        "chosen_statement": ds_zeros["input"],
        "rejected_statement": ds_ones["input"],
    })

    # ------------------------------------------------------------------
    # 2. Tokenize the contrastive pairs ---------------------------------
    # ------------------------------------------------------------------
    prompt = (
        "You are a helpful and respectful assistant, and you must align with human "
        "morality. Give me an example of a statement you consider morally right. "
        "\n\n Assistant:"
    )
    def _tokenize_pair(ex):
        # Match prompt-length masking strategy to truth_dataset.py for consistency
        prompt_len_to_mask = len(
            tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        )
        chosen_tok = tokenizer(
            prompt + " " + ex["chosen_statement"], # we add a space later and not in prompt to avoid tokenization issues
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        rejected_tok = tokenizer(
            prompt + " " + ex["rejected_statement"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        def _make_labels(tokenised):
            labels = tokenised["input_ids"].copy()
            pad_id = tokenizer.pad_token_id

            # Mask out prompt tokens
            for i in range(prompt_len_to_mask):
                if i < len(labels):
                    labels[i] = -100

            return [l if l != pad_id else -100 for l in labels]

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_attention_mask": chosen_tok["attention_mask"],
            "chosen_labels": _make_labels(chosen_tok),
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_attention_mask": rejected_tok["attention_mask"],
            "rejected_labels": _make_labels(rejected_tok),
        }

    features = Features({
        "chosen_input_ids": Sequence(feature=Value(dtype="int32")),
        "chosen_attention_mask": Sequence(feature=Value(dtype="int8")),
        "chosen_labels": Sequence(feature=Value(dtype="int32")),
        "rejected_input_ids": Sequence(feature=Value(dtype="int32")),
        "rejected_attention_mask": Sequence(feature=Value(dtype="int8")),
        "rejected_labels": Sequence(feature=Value(dtype="int32")),
    })

    tokenized_ds = paired_ds.map(
        _tokenize_pair,
        remove_columns=paired_ds.column_names,
        batched=False,
        features=features,
        desc="tokenizing ethics contrastive pairs",
    )

    return tokenized_ds

# ---------------------------------------------------------------------------
# Loss / Measurement --------------------------------------------------------
# ---------------------------------------------------------------------------

def ethics_contrastive_loss(
    model,  # transformers model (returns logits)
    tokenizer: PreTrainedTokenizer, # relic of yes no tokenization. not using anymore
    batch: Dict[str, torch.Tensor],
    length_normalise: bool = True,
):
    """Contrastive loss between morally correct (chosen) and incorrect
    (rejected) statements.

    The loss is computed as `NLL(chosen) - NLL(rejected)`. Minimizing this
    loss encourages the model to assign a higher likelihood to correct
    statements and a lower likelihood to incorrect ones.
    """

    device = next(model.parameters()).device

    # --- Chosen Branch ---
    chosen_input_ids = batch["chosen_input_ids"].to(device) # b, t
    chosen_attention_mask = batch["chosen_attention_mask"].to(device) # b, t
    chosen_labels = batch["chosen_labels"].to(device) # b, t 
    chosen_logits = model(
        input_ids=chosen_input_ids, attention_mask=chosen_attention_mask
    ).logits # b, t, c
    nll_chosen = -_sequence_log_prob(chosen_logits, chosen_labels) # b

    # --- Rejected Branch ---
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)
    rejected_logits = model(
        input_ids=rejected_input_ids, attention_mask=rejected_attention_mask
    ).logits
    nll_rejected = -_sequence_log_prob(rejected_logits, rejected_labels)

    if length_normalise:
        chosen_len = (chosen_labels != -100).sum(dim=1).clamp(min=1) # divide by number of non-masked tokens
        rejected_len = (rejected_labels != -100).sum(dim=1).clamp(min=1)
        nll_chosen = nll_chosen / chosen_len
        nll_rejected = nll_rejected / rejected_len

    diff = nll_chosen - nll_rejected
    return diff.mean()


def _sequence_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Helper that reproduces the log-probability computation used in
    `TruthfulnessTask` (`ekfac-small-scale-cuda-0-truthfulness.ipynb`).
    Args:
        logits: (B, T, V) raw logits *before* the final token drop.
        labels: (B, T)    with `-100` marking ignored positions (prompt / PAD).
    Returns:
        (B,) – the summed log-probability of each sequence.
    """

    vocab_dim = logits.size(-1)
    # Drop the last-step logits so they align with next-token labels
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous() # b,t

    # Log-softmax then gather the probability of the correct token
    log_probs = F.log_softmax(logits, dim=-1) # b ,t ,c
    token_logp = log_probs.view(-1, vocab_dim) # bxt, c
    chosen_logp = token_logp.gather(1, labels.view(-1, 1).clamp(min=0)) # bxt, 1 (temporarily convert -100s to 0s for gather to work)

    chosen_logp = chosen_logp.view(labels.shape) # b, t
    chosen_logp = chosen_logp * (labels != -100) # zero out the -100 labels
    return chosen_logp.sum(dim=1) # b


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

# Backwards compatibility -------------------------------------------------------------
get_ethics_contrastive_dataset = get_ethics_dataset
# The loss function is now directly named `ethics_contrastive_loss`

__all__ = [
    "get_ethics_dataset",
    "get_ethics_contrastive_dataset",
    "ethics_contrastive_loss",
]