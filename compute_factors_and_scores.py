import logging
import os
import sys
from typing import Dict
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
from transformers import default_data_collator, AutoTokenizer
from dotenv import load_dotenv

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM 
from termcolor import colored
from utils.performance_functions.anthropic import get_anthropic_dataset
logging.basicConfig(level=logging.INFO)
BATCH_TYPE = Dict[str, torch.Tensor]

load_dotenv()


##############################################################################
'''In order to run this script, you need only change the following.'''
NUM_TRANSFORMER_BLOCKS_LIST = [24, 32, 32]
MODEL_NAMES = ["username/POST-SFT-pythia-1.4b", "username/POST-SFT-pythia-2.8b", "username/POST-SFT-pythia-6.9b"] # Replace username with actual HuggingFace username
TASK_NAME = "bias" # can be "bias" "ethics" "truth"
MODEL_FAMILY = "pythia" # can be "pythia" or "qwen"
seed = 42
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


assert len(MODEL_NAMES) == len(NUM_TRANSFORMER_BLOCKS_LIST), "MODEL_NAMES and NUM_TRANSFORMER_BLOCKS_LIST must have the same length"

for i in range(len(NUM_TRANSFORMER_BLOCKS_LIST)):
    # Prepare the tokenizer and dataset.
    print(colored(f"Loading model {MODEL_NAMES[i]}...", "green"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[i], cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[i], cache_dir=CACHE_DIR)

    '''The following block of code checks if our number of transformer blocks actually match the final block.'''
    if MODEL_FAMILY == "pythia":
        num_layers = len(model.gpt_neox.layers)
    elif MODEL_FAMILY == "qwen":
        num_layers = len(model.model.layers)
    else:
        raise NotImplementedError(f"Model family {MODEL_FAMILY} not supported for layer number check.")
    assert NUM_TRANSFORMER_BLOCKS_LIST[i] == num_layers, f"NUM_TRANSFORMER_BLOCKS_LIST[{i}] must be equal to the number of transformer blocks in the model, got {NUM_TRANSFORMER_BLOCKS_LIST[i]} and {num_layers}"
    
    anthropic_dataset = get_anthropic_dataset(tokenizer, split='train')
    anthropic_dataset_10k_split = anthropic_dataset.shuffle(seed=seed).select(range(10000)) # random 10k subset
    # Define task and prepare model.
    
    if TASK_NAME == "bias":
        from tasks.bias_task import BiasTask
        from utils.performance_functions.bias_dataset import get_bias_agreement_dataset
        TASK = BiasTask
        QUERY_DATASET = get_bias_agreement_dataset(tokenizer, split='train')
    elif TASK_NAME == "ethics":
        from tasks.ethics_task import EthicsTask
        from utils.performance_functions.ethics_dataset import get_ethics_contrastive_dataset
        TASK = EthicsTask
        QUERY_DATASET = get_ethics_contrastive_dataset(tokenizer, split='train')
    elif TASK_NAME == "truth":
        from tasks.truth_task import TruthTask
        from utils.performance_functions.truth_dataset import get_truthfulness_contrastive_dataset
        TASK = TruthTask
        QUERY_DATASET = get_truthfulness_contrastive_dataset(tokenizer, split='train')
    else:
        raise NotImplementedError(f"TASK_NAME must be 'bias' or 'ethics' or 'truth', got {TASK_NAME}")

    print(colored(f"Found task {TASK_NAME}", "cyan"))
    if TASK_NAME in ["bias", "ethics"]:
        task = TASK(tokenizer=tokenizer, num_transformer_blocks=NUM_TRANSFORMER_BLOCKS_LIST[i], model_family = MODEL_FAMILY, num_blocks=NUM_BLOCKS)
    elif TASK_NAME == "truth":
        task = TASK(num_transformer_blocks=NUM_TRANSFORMER_BLOCKS_LIST[i], model_family = MODEL_FAMILY, num_blocks=NUM_BLOCKS)
    else:
        raise NotImplementedError(f"TASK_NAME must be 'bias' or 'ethics' or 'truth', got {TASK_NAME}")

    model = prepare_model(model, task)
    if USE_COMPILE:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name=f"{TASK_NAME}_{MODEL_NAMES[i].split('/')[-1]}",
        model=model,
        task=task,
        profile=PROFILE,
        output_dir=f"ekfac_experiments/influence_results-{seed}" if seed !=42 else "ekfac_experiments/influence_results",
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

    print(colored("Fitting factors...", "green"))

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=anthropic_dataset_10k_split,
        per_device_batch_size=None,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=8,  
        overwrite_output_dir=False,
    )

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

    score_args.aggregate_query_gradients=True

    print(colored("Computing pairwise scores...", "green"))
    QUERY_BATCH_SIZE = 2
    TRAIN_BATCH_SIZE = 2
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=QUERY_DATASET,
        train_dataset=anthropic_dataset,
        per_device_query_batch_size=QUERY_BATCH_SIZE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        overwrite_output_dir=False
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")
    del model
    del tokenizer
    del analyzer
    del task
    del QUERY_DATASET
    del anthropic_dataset
    del anthropic_dataset_10k_split
    del factors_name
    del factor_args
    del score_args
    del scores_name