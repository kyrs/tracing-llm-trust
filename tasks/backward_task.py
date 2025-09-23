from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from kronfluence.task import Task
from utils.performance_functions.anthropic import compute_batch_loss

BATCH_TYPE = Dict[str, torch.Tensor]

class BackwardTask(Task):
    def __init__(self, num_transformer_blocks: int, model_family: str, num_blocks: int = 1):
        self.num_blocks = num_blocks
        if num_blocks!=1:
            raise Warning("num_blocks must be 1")
        self.num_transformer_blocks = num_transformer_blocks
        self.model_family = model_family
        if model_family not in ["pythia", "qwen"]:
            raise NotImplementedError(f"Model family must be one of ['pythia', 'qwen'], got {model_family}")

    def compute_train_loss(self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False,) -> torch.Tensor: # not factor_args.use_empirical_fisher.
    #use_empirical_fisher is False by default for fit_all_factors in fator_args. This means that sample = True will be passed while fitting the analyzer => FIM will be computed using samples from the model, not true labels
        raise NotImplementedError("If used properly, this task should not be used.")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return compute_batch_loss(model = model, batch = batch, DEVICE = "cuda", requires_grad = True)


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