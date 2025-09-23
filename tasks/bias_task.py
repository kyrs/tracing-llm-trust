from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from kronfluence.task import Task
from utils.performance_functions.bias_dataset import bias_agreement_nll_loss

BATCH_TYPE = Dict[str, torch.Tensor]

class BiasTask(Task):
    def __init__(self, tokenizer: AutoTokenizer, num_transformer_blocks: int, model_family: str, num_blocks: int = 1):
        self.num_blocks = num_blocks
        if num_blocks!=1:
            raise Warning("num_blocks must be 1")
        self.num_transformer_blocks = num_transformer_blocks
        self.tokenizer = tokenizer
        self.model_family = model_family
        if model_family not in ["pythia", "qwen"]:
            raise NotImplementedError(f"Model family must be one of ['pythia', 'qwen'], got {model_family}")

    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False,) -> torch.Tensor: # not factor_args.use_empirical_fisher.
    #use_empirical_fisher is False by default for fit_all_factors in fator_args. This means that sample = True will be passed while fitting the analyzer => FIM will be computed using samples from the model, not true labels
        logits = model(
            input_ids=batch["input_ids"], # prompt + chosen tokenized
            attention_mask=batch["attention_mask"], # a list of 11111100000
        ).logits #  B, T, 50k
        logits = logits[..., :-1, :].contiguous() # B, T-1, 50k
        logits = logits.view(-1, logits.size(-1)) # B*(T-1), 50k

        if not sample: # copmute loss by teacher forcing.
            labels = batch["labels"] # prompt + chosen tokenized, but prompt tokens forced to -100
            labels = labels[..., 1:].contiguous() # B, T-1
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return bias_agreement_nll_loss(model, self.tokenizer, batch)


    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        if self.model_family == "pythia":
            for i in range(self.num_transformer_blocks - self.num_blocks, self.num_transformer_blocks):
                print(i, end=" ")
                total_modules.append(f"gpt_neox.layers.{i}.attention.query_key_value")
                total_modules.append(f"gpt_neox.layers.{i}.attention.dense")
                total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_h_to_4h")
                total_modules.append(f"gpt_neox.layers.{i}.mlp.dense_4h_to_h")
        elif self.model_family == "qwen":
            for i in range(self.num_transformer_blocks - self.num_blocks, self.num_transformer_blocks):
                print(i, end=" ")
                total_modules.append(f"model.layers.{i}.self_attn.q_proj")
                total_modules.append(f"model.layers.{i}.self_attn.k_proj")
                total_modules.append(f"model.layers.{i}.self_attn.v_proj")
                total_modules.append(f"model.layers.{i}.self_attn.o_proj")
                total_modules.append(f"model.layers.{i}.mlp.gate_proj")
                total_modules.append(f"model.layers.{i}.mlp.up_proj")
                total_modules.append(f"model.layers.{i}.mlp.down_proj")
        else:
            raise NotImplementedError(f"Model family must be one of ['pythia', 'qwen'], got {self.model_family}")
        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]