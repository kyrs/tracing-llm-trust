'''
This file loads the factors and the scores and computes gradients corresponding to BOTTOM_K points
'''

import logging
import os
import sys
from pathlib import Path
import time
import json
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
from typing import Dict
import torch
from transformers import default_data_collator, AutoTokenizer
from dotenv import load_dotenv

from kronfluence.module.utils import (
    get_tracked_module_names, set_mode, prepare_modules,
    synchronize_modules, finalize_all_iterations,
    update_factor_args, set_factors
)
from kronfluence.utils.constants import ACCUMULATED_PRECONDITIONED_GRADIENT_NAME, LAMBDA_MATRIX_NAME
from kronfluence.module.tracked_module import TrackedModule, ModuleMode
from accelerate.utils import send_to_device        
from kronfluence.utils.state import no_sync
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from torch.cuda.amp import autocast
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM 
from termcolor import colored
from utils.performance_functions.anthropic import get_anthropic_dataset
from tasks.backward_task import BackwardTask
import random
logging.basicConfig(level=logging.INFO)
BATCH_TYPE = Dict[str, torch.Tensor]
load_dotenv()


##############################################################################
'''In order to run this script, you need only change the following.'''
NUM_TRANSFORMER_BLOCKS_LIST = [24, 32, 32]
MODEL_NAMES = [{POST-SFT-pythia-1.4b}, {POST-SFT-pythia-2.8b}, {POST-SFT-pythia-6.9b}] # hf_username/model_name
TASK_NAME = "bias" # can be "bias" "ethics" "truth"
MODEL_FAMILY = "pythia"
BOTTOM_K  = 100
SCORES_TYPE = "pure_ekfac" # can be "pure_ekfac" "dpp" "random"
IF_WEIGHT = 0
GRADIENT_REDUCTION_TYPE = "mean" # can be "mean" or "sum"
SEED = 42
##############################################################################

NUM_BLOCKS = 1
QUERY_GRADIENT_RANK = -1
USE_HALF_PRECISION = True
FACTOR_STRATEGY = "ekfac"
USE_COMPILE = False
QUERY_BATCH_SIZE = 2
TRAIN_BATCH_SIZE = 2
PROFILE = False
CACHE_DIR = os.getenv("HUB_CACHE")
COMPUTE_PER_TOKEN_SCORES = False

assert TASK_NAME == "bias" if SCORES_TYPE == "random" else True, "TASK_NAME must be 'bias' if SCORES_TYPE is 'random'"
assert len(MODEL_NAMES) == len(NUM_TRANSFORMER_BLOCKS_LIST), "MODEL_NAMES and NUM_TRANSFORMER_BLOCKS_LIST must have the same length"

for i in range(len(NUM_TRANSFORMER_BLOCKS_LIST)):
    print(colored(f"Processing {MODEL_NAMES[i]}", "green"))
    sft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[i], cache_dir = CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[i], cache_dir = CACHE_DIR)

    ANALYSIS_NAME = f"{TASK_NAME}_{MODEL_NAMES[i].split('/')[-1]}"
    # Define task and prepare model.
    task = BackwardTask(num_transformer_blocks=NUM_TRANSFORMER_BLOCKS_LIST[i], model_family=MODEL_FAMILY)
    model = prepare_model(sft_model, task)
    if USE_COMPILE:
        model = torch.compile(model)
    analyzer = Analyzer(
        analysis_name=ANALYSIS_NAME, # folder in output dir 
        model=model,
        task=task,
        profile=PROFILE,
        output_dir=f"ekfac_experiments/influence_results-{SEED}" if SEED != 42 else "ekfac_experiments/influence_results",
    )

    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    # Compute influence factors.
    factors_name = FACTOR_STRATEGY
    factor_args = FactorArguments(strategy=FACTOR_STRATEGY) # use empirical fisher is false by default.
    if USE_HALF_PRECISION:
        factor_args = all_low_precision_factor_arguments(strategy=FACTOR_STRATEGY, dtype=torch.bfloat16)
        factors_name += "_half"
    if USE_COMPILE:
        factors_name += "_compile"
    
    # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    if USE_HALF_PRECISION:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    if USE_COMPILE:
        scores_name += "_compile"
    if COMPUTE_PER_TOKEN_SCORES:
        score_args.compute_per_token_scores = True
        scores_name += "_per_token"
    rank = QUERY_GRADIENT_RANK if QUERY_GRADIENT_RANK != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"
    score_args.aggregate_query_gradients = True
                
    # ---------------
    # 1. Reuse what Analyzer already created
    # ---------------
    model  = analyzer.model
    task   = analyzer.task
    state  = analyzer.state
    
    factors_dir = analyzer.factors_output_dir(factors_name)
    if not factors_dir.exists():
        print(colored(f"Factors directory not found at {factors_dir}", "red"))
        continue

    factor_args = analyzer.load_factor_args(factors_name)         # loads the saved FactorArguments
    score_args  = score_args
    
    print(colored("Performing checks...", "green"))
    loaded_factors = analyzer.load_all_factors(factors_name)
    
    # Sanity-check – do we have λ for every module we are going to track?
    missing = [
        m for m in get_tracked_module_names(model)
        if m not in loaded_factors[LAMBDA_MATRIX_NAME]
    ]
    assert len(missing) == 0, f"λ not found for: {missing}"
    
    # Push **all** factors into the wrappers
    for factor_name, per_module_dict in loaded_factors.items():
        set_factors(model, factor_name, per_module_dict, clone=True)
    
    # ---------------
    # 2. Prepare modules & put them in GRADIENT_AGGREGATION mode (aggregate first)
    # ------
    
    # 0) make sure the modules know the (low-preci) args you used
    update_factor_args(model, factor_args)
    
    # 1) load every factor that EKFAC needs and stuff them into the modules
    loaded_factors = analyzer.load_all_factors(factors_name)
    for factor_name, per_module_dict in loaded_factors.items():
        set_factors(model, factor_name, per_module_dict, clone=True)
    tracked = get_tracked_module_names(model)
    prepare_modules(model, tracked, state.device)
    set_mode(model, ModuleMode.GRADIENT_AGGREGATION, tracked, release_memory=False)
    
    dataset = get_anthropic_dataset(tokenizer, split='train')
    # ─────────── 1. pick top-K indices by influence score ────────────────────
    if SCORES_TYPE == "pure_ekfac":
        scores_path = Path(
            f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/scores_ekfac_half/pairwise_scores.safetensors"
        ) if SEED != 42 else Path(
            f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/scores_ekfac_half/pairwise_scores.safetensors"
        )
        if scores_path.is_file():
            print(f"Loading influence scores from {scores_path} …")
            # Load according to file type (.pt / .safetensors)
            if scores_path.suffix == ".safetensors":
                loaded = safe_load_file(str(scores_path))  # returns Dict[str, Tensor]
            else:
                loaded = torch.load(scores_path, map_location="cpu")
        
            # `analyzer.save_pairwise_scores` stores a dict; grab the tensor if needed.
            if isinstance(loaded, dict) and "scores" in loaded:
                scores_tensor = loaded["scores"]
            elif isinstance(loaded, dict) and "all_modules" in loaded:
                scores_tensor = loaded["all_modules"]
            else:
                scores_tensor = loaded  # assume the tensor itself
        
            scores = scores_tensor.to(torch.float32).squeeze()  # (N,)
        else:
            raise FileNotFoundError(colored(f"Scores file not found at {scores_path}", "red"))
        
        indices = torch.topk(scores, k=BOTTOM_K, largest=False).indices.tolist()
    elif SCORES_TYPE == "dpp":
    # ---------------------- 1.1 Use the indices from the DDP function
        with open(f"dpp/{ANALYSIS_NAME}_indices_{BOTTOM_K}_ifweight{IF_WEIGHT}.json", "r") as f:
            indices = json.load(f)["indices"]
        
    elif SCORES_TYPE == "random":
        random.seed(42)  # ensure reproducibility across runs
        indices = random.sample(range(len(dataset)), BOTTOM_K)

    else:
        raise NotImplementedError(f"SCORES_TYPE {SCORES_TYPE} not implemented.")
    print(f"Bottom-{BOTTOM_K} influential training indices:", indices[:BOTTOM_K], "...")

    # Rebuild dataloader to iterate only over the selected indices
    query_loader = analyzer._get_dataloader(
        dataset=dataset,
        per_device_batch_size=1,
        indices=indices,
        dataloader_params=analyzer._configure_dataloader(None),
        allow_duplicates=True,
    )
    
    # ---------------
    # 4. Forward / backward pass to accumulate raw summed gradients (faster)
    # ---------------
    
    enable_amp = score_args.amp_dtype is not None
    
    # Process only the filtered batches
    # GPU synchronize before measuring time to ensure accurate timing of GPU ops
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_timestamp = time.time()
    model.zero_grad(set_to_none=True)
    num_points_used_for_gradients = 0
    for index, batch in enumerate(query_loader):
        # Avoid accumulating parameter grads across samples (saves memory)
        model.zero_grad(set_to_none=True)
        batch = send_to_device(batch, state.device)
        with no_sync(model, state):
            with autocast(enabled=enable_amp, dtype=score_args.amp_dtype):
                measurement = task.compute_measurement(batch, model)
            measurement.backward()
        synchronize_modules(model, tracked, state.num_processes)
        # No per-sample preconditioning, just keep accumulating aggregated gradient inside modules
        num_points_used_for_gradients += 1
    assert num_points_used_for_gradients == len(indices) == BOTTOM_K, "Number of points used for gradients does not match the number of indices"
    # 4.1 Switch to preconditioning aggregated gradient once after the loop and compute
    set_mode(model, ModuleMode.PRECONDITION_GRADIENT, tracked, release_memory=False)
    finalize_all_iterations(model, tracked)
    
    # 5. Read the pre-conditioned query gradients
    # ---------------
    query_grads = {}
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked:
            pg = module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
            if pg is not None:
                # This script is only configured for full-rank gradients.
                # Raise an error if low-rank gradients are detected.
                if isinstance(pg, list):
                    raise TypeError(
                        f"Detected low-rank gradient for module {module.name}. "
                        "This script is configured for full-rank gradients only. "
                        "Please set QUERY_GRADIENT_RANK = -1."
                    )

                # The aggregated gradient is a *sum*. The original implementation
                # used a *mean*. To make it equivalent, we divide by the number of samples.
                if GRADIENT_REDUCTION_TYPE == "mean":
                    pg.div_(len(indices))
                elif GRADIENT_REDUCTION_TYPE == "sum":
                    pg.div_(100) # this will ensure that the gradient is the same as the original implementation
                else:
                    raise NotImplementedError(f"GRADIENT_REDUCTION_TYPE {GRADIENT_REDUCTION_TYPE} not implemented.")
                query_grads[module.name] = pg.clone().cpu()
    
    print({k: v.shape for k, v in query_grads.items()})

    # 6. Save aggregated, preconditioned query gradients per (model, task, score)
    # --------------------------------------------------------------------------
    if SCORES_TYPE == "pure_ekfac":
        save_dir = Path(
        f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
    ) if SEED != 42 else Path(
        f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
    )
    elif SCORES_TYPE == "dpp":
        save_dir = Path(
        f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_ifweight{IF_WEIGHT}_bottom{BOTTOM_K}"
    ) if SEED != 42 else Path(
        f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_ifweight{IF_WEIGHT}_bottom{BOTTOM_K}"
    )
    elif SCORES_TYPE == "random":
        save_dir = Path(
        f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
    ) if SEED != 42 else Path(
        f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
    )
    else:
        raise NotImplementedError(f"SCORES_TYPE {SCORES_TYPE} not implemented.")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "aggregated_query_grads.safetensors"
    safe_save_file(query_grads, str(save_path))
    print(colored(f"Saved aggregated query gradients to {save_path}", "green"))

    # Record elapsed time for computing gradients over bottom-K points
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.time() - start_timestamp
    timing_info = {
        "model_name": MODEL_NAMES[i],
        "analysis_name": ANALYSIS_NAME,
        "scores_type": SCORES_TYPE,
        "bottom_k": BOTTOM_K,
        "num_indices": len(indices),
        "elapsed_seconds": float(elapsed_seconds),
    }
    timing_path = save_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(colored(f"Saved timing to {timing_path} ({elapsed_seconds:.2f}s)", "green"))
    del query_grads
    del model
    del task
    del analyzer
    del state
    del factor_args
    del score_args
    del loaded_factors
    del tracked
    del dataset
    