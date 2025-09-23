from datasets import load_dataset
from transformers import PreTrainedTokenizer
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
# Default path (full dataset – kept as fallback)
TRUTHFUL_CSV_PATH = "../../data/truthfulness/TruthfulQA.csv"

# Directory that contains the pre-generated splits created by
# ``data/truthfulness/split_dataset.py``.
SPLIT_DIR = Path("../../data/truthfulness/splits")

# Mapping from user-facing ``split`` names to the corresponding CSV files.
# Only ``train`` and ``test`` are strictly required by the current workflow,
# but we add the other two for completeness.
SPLIT_CSV_FILES = {
    "train": SPLIT_DIR / "TruthfulQA_train.csv",
    "test": SPLIT_DIR / "TruthfulQA_test.csv",
}


def get_truthfulness_contrastive_dataset(
    tokenizer: PreTrainedTokenizer,
    *,
    split: str,
    indices=None,
    max_length: int = 512,
):
    """Return a **contrastive** TruthfulQA dataset with the fields expected by
    `TruthfulnessTask.compute_measurement`, namely the six keys:
        - chosen_input_ids / chosen_attention_mask / chosen_labels
        - rejected_input_ids / rejected_attention_mask / rejected_labels

    For every example we use the (single) *best* correct answer as the
    positive branch and the first *incorrect* answer as the negative branch.
    """

    if split is None:
        raise ValueError("'split' argument must be provided (e.g. 'train', 'test').")

    # ------------------------------------------------------------------
    # Select the correct CSV depending on the requested split.
    # If a pre-generated split file exists, we load it directly; otherwise we
    # fall back to the original behaviour (loading the full CSV and relying on
    # the split argument).
    # ------------------------------------------------------------------

    split_lower = split.lower()
    data_file = SPLIT_CSV_FILES.get(split_lower)

    if data_file is not None and data_file.exists():
        # The split file is present – load it as a single 'train' split.
        raw_ds = load_dataset("csv", data_files=str(data_file), split="train")
    else:
        # Fallback to original monolithic CSV (e.g. for 'validation', etc.)
        raw_ds = load_dataset("csv", data_files=TRUTHFUL_CSV_PATH, split=split_lower)


    def _tokenize_pair(example):
        prompt = example["Question"].strip() + "\n\nAnswer:" # No trailing space

        # Preferred / positive branch – use the best (or first correct) answer
        chosen_answer = " " + ( # Prepend space
            example["Best Answer"].strip()
            if example["Best Answer"]
            else example["Correct Answers"].split(";")[0].strip()
        )

        # Rejected / negative branch – take the first incorrect answer
        rejected_answer = " " + example["Incorrect Answers"].split(";")[0].strip()

        chosen_text = prompt + chosen_answer
        rejected_text = prompt + rejected_answer

        chosen_tok = tokenizer(
            chosen_text, max_length=max_length, truncation=True, padding="max_length"
        )
        rejected_tok = tokenizer(
            rejected_text, max_length=max_length, truncation=True, padding="max_length"
        )

        # Helper to build labels (mask out prompt + pad tokens)
        def _make_labels(tokenised):
            labels = tokenised["input_ids"].copy()
            prompt_len = len(
                tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
            )
            labels[:prompt_len] = [-100] * prompt_len  # ignore prompt tokens
            pad_id = tokenizer.pad_token_id
            labels = [l if l != pad_id else -100 for l in labels]  # ignore pads
            return labels

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_attention_mask": chosen_tok["attention_mask"],
            "chosen_labels": _make_labels(chosen_tok),
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_attention_mask": rejected_tok["attention_mask"],
            "rejected_labels": _make_labels(rejected_tok),
        }

    ds = raw_ds.map(
        _tokenize_pair,
        remove_columns=raw_ds.column_names,
        batched=False,
        desc="tokenizing TruthfulQA contrastive pairs",
        features=Features({
            "chosen_input_ids": Sequence(feature=Value(dtype='int32')),
            "chosen_attention_mask": Sequence(feature=Value(dtype='int8')),
            "chosen_labels": Sequence(feature=Value(dtype='int32')),
            "rejected_input_ids": Sequence(feature=Value(dtype='int32')),
            "rejected_attention_mask": Sequence(feature=Value(dtype='int8')),
            "rejected_labels": Sequence(feature=Value(dtype='int32')),
        })
    )

    if indices is not None:
        ds = ds.select(indices)

    # Print the dataset type in red for quick verification
    print(f"\033[91mReturned dataset type: {type(ds)}\033[0m")

    return ds

# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset, Features, Sequence, Value

BATCH = Dict[str, torch.Tensor]  # alias used for type-hints below


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
    logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
    labels = labels[..., 1:].contiguous()        # (B, T-1)

    # Log-softmax then gather the probability of the correct token
    log_probs = F.log_softmax(logits, dim=-1)    # (B, T-1, V)
    token_logp = log_probs.view(-1, vocab_dim)   # flatten for gather
    chosen_logp = token_logp.gather(
        1, labels.view(-1, 1).clamp(min=0)       # clamp keeps -100 rows 0
    ).view(labels.shape)

    chosen_logp = chosen_logp * (labels != -100)  # mask prompt / PAD tokens
    return chosen_logp.sum(dim=1)                 # (B,)


def truthfulness_contrastive_loss(
    model: PreTrainedModel,
    batch: BATCH,
    length_normalise: bool = True, # compute the mean nll by default
) -> torch.Tensor:
    """Contrastive loss: NLL(chosen) − NLL(rejected).

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The language model to evaluate (in `grad` mode here).
    batch : Dict[str, torch.Tensor]
        A batch produced by :func:`get_truthfulness_contrastive_dataset`.
    length_normalise : bool, default = False
        If *True*, each sequence-level negative log-likelihood is divided by
        the number of non-masked tokens so that the score corresponds to the
        **average** per-token NLL rather than the **sum**.  This removes any
        bias towards shorter continuations.

    Returns
    -------
    torch.Tensor
        A scalar – the mean (over the batch) of the chosen-minus-rejected
        (possibly length-normalised) NLLs.  Negative values indicate the model
        prefers the chosen (truthful) answer.
    """

    device = next(model.parameters()).device

    # Ensure tensors are on the right device
    chosen_input_ids = batch["chosen_input_ids"].to(device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)

    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    # Forward passes
    chosen_logits = model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask,
    ).logits
    rejected_logits = model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask,
    ).logits

    # Negative log-likelihoods (sums by default)
    nll_chosen = -_sequence_log_prob(chosen_logits, chosen_labels)       # (B,)
    nll_rejected = -_sequence_log_prob(rejected_logits, rejected_labels)  # (B,)

    if length_normalise:
        # number of valid (non-masked) tokens in each sequence – clamp avoids 0
        chosen_len = (chosen_labels != -100).sum(dim=1).clamp(min=1)    # (B,)
        rejected_len = (rejected_labels != -100).sum(dim=1).clamp(min=1)  # (B,)

        nll_chosen = nll_chosen / chosen_len
        nll_rejected = nll_rejected / rejected_len

    diff = nll_chosen - nll_rejected  # (B,)
    return diff.mean()


# Exportable symbols from this module
__all__ = [
    "get_truthfulness_contrastive_dataset",
    "truthfulness_contrastive_loss",
]