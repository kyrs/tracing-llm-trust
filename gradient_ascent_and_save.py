import os
import sys
import json
from pathlib import Path
import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from dotenv import load_dotenv
load_dotenv()
############################################################################
MODEL_NAME = ... # hf_username/model_name
TASK_NAME = "bias"  # "ethics", "bias", "truth", "combined"etc.
SCORES_TYPE = "pure_ekfac" # "pure_ekfac" "dpp" "random"
BOTTOM_K = 100 # number of points used to compute gradients. This will be necessary in fetching the grads. 
LEARNING_RATE = 0.012 # generally a decent start
PUSH_TO_HUB = False
SAVE_TO_DISK = False

COMPUTE_ANTHROPIC_EVAL = True
COMPUTE_TASK_EVAL = True
SEED = 42
############################################################################

HUB_USERNAME = ...  # Replace with your Hugging Face username
HUGGING_FACE_HUB_TOKEN = os.getenv("HUB_TOKEN")
CACHE_DIR = os.getenv("HUB_CACHE")

DEVICE = "cuda"

if SCORES_TYPE == "dpp":
    print(colored("Running gradient ascent with DPP scores...", "yellow"))
    # Uncomment the following lines if you want to add confirmation:
    # print(colored("Are you sure you want to run gradient ascent with DPP scores? (y/n)", "red"))
    # if input() != "y":
    #     exit()

if SCORES_TYPE == "random":
    print(colored("Running gradient ascent with random scores...", "yellow"))
    # Uncomment the following lines if you want to add confirmation:
    # print(colored("Are you sure you want to run gradient ascent with random scores? (y/n)", "red"))
    # if input() != "y":
    #     exit()

def find_submodule(model, dotted_name):
    """
    Recursively find a submodule in a model given its dotted name.
    """
    current = model
    for part in dotted_name.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def main():
    """
    Main function to load a model, apply gradients, and save the result.
    """

    # 1. Load model and tokenizer
    print(colored(f"Loading model: {MODEL_NAME}", "cyan"))
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded.")

    # 2. Load saved query gradients
    if TASK_NAME == "combined":
        ANALYSIS_NAME = f"{TASK_NAME}_bias_ethics_truth_{MODEL_NAME.split('/')[-1]}"
        grad_dir = Path(
            f"ekfac_experiments/influence_results-{SEED}/combined_scores/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        ) if SEED != 42 else Path(
            f"ekfac_experiments/influence_results/combined_scores/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        )
    elif SCORES_TYPE == "random":
        # the factors won't matter, so we've decided to use the bias analysis name
        ANALYSIS_NAME = f"bias_{MODEL_NAME.split('/')[-1]}"
        grad_dir = Path(
            f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        ) if SEED != 42 else Path(
            f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        )
    else:
        ANALYSIS_NAME = f"{TASK_NAME}_{MODEL_NAME.split('/')[-1]}"
        grad_dir = Path(
            f"ekfac_experiments/influence_results-{SEED}/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        ) if SEED != 42 else Path(
            f"ekfac_experiments/influence_results/{ANALYSIS_NAME}/query_grads_{SCORES_TYPE}_bottom{BOTTOM_K}"
        )
        
    grad_path = grad_dir / "aggregated_query_grads.safetensors"
    
    if not grad_path.is_file():
        print(colored(f"Gradient file not found at {grad_path}. Exiting.", "red"))
        return
        
    print(f"Loading gradients from {grad_path}")
    query_grads = safe_load_file(str(grad_path))

    # 3. Apply gradient ascent
    print(colored(f"Applying gradient ascent with LR={LEARNING_RATE}", "yellow"))
    with torch.no_grad():
        for mod_name, pg in query_grads.items():
            # grad = reduce_fn(pg, dim=0).to(DEVICE)
            grad = torch.squeeze(pg, dim=0).to(DEVICE)
            module = find_submodule(model, mod_name)
            if hasattr(module, "base_layer"):
                module = module.base_layer

            if module.bias is not None:
                weight_grad = grad[:, :-1]
                bias_grad = grad[:, -1]

                assert module.weight.shape == weight_grad.shape, f"Weight shape mismatch for {mod_name}"
                assert module.bias.shape == bias_grad.shape, f"Bias shape mismatch for {mod_name}"
                
                module.weight.data += LEARNING_RATE * weight_grad.to(module.weight.dtype)
                module.bias.data += LEARNING_RATE * bias_grad.to(module.bias.dtype)
            else:
                weight_grad = grad
                print(colored(f"Module {mod_name} has no bias. Applying only weight gradients.", "yellow"))
                
                assert module.weight.shape == weight_grad.shape, f"Weight shape mismatch for {mod_name}"
                
                module.weight.data += LEARNING_RATE * weight_grad.to(module.weight.dtype)

    print("Gradient ascent applied.")


    # 4. Save the modified model
    save_dir_name = f"{ANALYSIS_NAME}_bottom{BOTTOM_K}_lr{LEARNING_RATE}"
    if SAVE_TO_DISK:
        save_dir = "gradient_ascent_models" / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_TO_DISK:
        print(colored(f"Saving updated model to {save_dir}", "green"))
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
 
        # 5. Save the configuration
        config_data = {
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "scores_type": SCORES_TYPE,
            "bottom_k": BOTTOM_K,
            "learning_rate": LEARNING_RATE,
            "reduce_fnc": "mean",
            "device": DEVICE,
        }
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
    
        print("Model, tokenizer, and config saved successfully.")

    # 6. Push to Hugging Face Hub if enabled
    if PUSH_TO_HUB:
        if not HUGGING_FACE_HUB_TOKEN:
            print(colored("HUGGING_FACE_HUB_TOKEN not found in environment. Cannot push to Hub.", "red"))
            return
            
        repo_name = f"{HUB_USERNAME}/{save_dir.name}"
        print(colored(f"Pushing model to Hugging Face Hub: {repo_name}", "cyan"))
        
        try:
            model.push_to_hub(
                repo_name,
                use_auth_token=HUGGING_FACE_HUB_TOKEN,
                safe_serialization=True
            )
            tokenizer.push_to_hub(
                repo_name,
                use_auth_token=HUGGING_FACE_HUB_TOKEN,
            )
            print(colored("Successfully pushed model and tokenizer to Hub.", "green"))
        except Exception as e:
            print(colored(f"Failed to push to Hub. Error: {e}", "red"))
        
    if COMPUTE_ANTHROPIC_EVAL:
        from utils.performance_functions.anthropic import get_anthropic_dataset, compute_mean_loss
        from transformers import default_data_collator
        from torch.utils.data import DataLoader
        
        print("Evaluating Anthropic token-level mean loss...")
        anthropic_test = get_anthropic_dataset(tokenizer, split='test').shuffle(seed=42)
        loader = DataLoader(anthropic_test, batch_size=8, shuffle=False, collate_fn=default_data_collator)
        mean_token_loss = compute_mean_loss(model, loader, DEVICE=DEVICE)
        print(colored(f"Anthropic token-level mean loss: {mean_token_loss:.4f}", "green"))
    
    if COMPUTE_TASK_EVAL:
        from transformers import default_data_collator
        from torch.utils.data import DataLoader
        
        # ------------------------------------------------------------------
        # Minimal task-specific evaluation
        # ------------------------------------------------------------------

        if TASK_NAME == "bias":
            from utils.performance_functions.bias_dataset import (
                get_bias_agreement_dataset,
                bias_agreement_nll_loss,
            )

            print("Evaluating bias-agreement mean loss (NLL)...")

            # Prepare dataset & dataloader
            bias_test = get_bias_agreement_dataset(tokenizer, split="test")
            loader = DataLoader(
                bias_test, batch_size=8, shuffle=False, collate_fn=default_data_collator
            )

            # Accumulate loss across evaluation set
            total_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch_loss = bias_agreement_nll_loss(
                        model=model, tokenizer=tokenizer, batch=batch
                    )
                    total_loss += batch_loss.item()

            mean_bias_loss = total_loss / len(loader)
            print(colored(f"Bias agreement mean loss: {mean_bias_loss:.4f}", "green"))
        elif TASK_NAME == "ethics":
            from utils.performance_functions.ethics_dataset import get_ethics_contrastive_dataset, ethics_contrastive_loss
            from transformers import default_data_collator
            from torch.utils.data import DataLoader

            print("Evaluating Ethics loss...")
            dataset = get_ethics_contrastive_dataset(tokenizer, split='test')
            dataloader = DataLoader(
                dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator
            )

            loss = 0
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch_loss = ethics_contrastive_loss(model=model, tokenizer=tokenizer, batch=batch)
                    loss += batch_loss.item()
            
            mean_loss = loss / len(dataloader)
            print(colored(f"Ethics loss: {mean_loss:.4f}", "green"))

        elif TASK_NAME == "truth":
            from utils.performance_functions.truth_dataset import (
                get_truthfulness_contrastive_dataset,
                truthfulness_contrastive_loss,
            )

            print("Evaluating truthfulness contrastive mean loss...")

            # Prepare dataset & dataloader
            truth_test = get_truthfulness_contrastive_dataset(tokenizer, split="test")
            loader = DataLoader(
                truth_test,
                batch_size=8,
                shuffle=False,
                collate_fn=default_data_collator,
            )

            # Accumulate loss across evaluation set
            total_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch_loss = truthfulness_contrastive_loss(
                        model=model, batch=batch
                    )
                    total_loss += batch_loss.item()

            mean_truth_loss = total_loss / len(loader)
            print(
                colored(
                    f"Truthfulness contrastive mean loss: {mean_truth_loss:.4f}", "green"
                )
            )
        elif TASK_NAME == "combined":
            from utils.performance_functions.bias_dataset import (
                get_bias_agreement_dataset,
                bias_agreement_nll_loss,
            )

            print("Evaluating bias-agreement mean loss (NLL)...")

            # Prepare dataset & dataloader
            bias_test = get_bias_agreement_dataset(tokenizer, split="test")
            loader = DataLoader(
                bias_test, batch_size=8, shuffle=False, collate_fn=default_data_collator
            )

            # Accumulate loss across evaluation set
            total_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch_loss = bias_agreement_nll_loss(
                        model=model, tokenizer=tokenizer, batch=batch
                    )
                    total_loss += batch_loss.item()

            mean_bias_loss = total_loss / len(loader)
            print(colored(f"Bias agreement mean loss: {mean_bias_loss:.4f}", "green"))
            
            from utils.performance_functions.ethics_dataset import get_ethics_contrastive_dataset, ethics_contrastive_loss
            from transformers import default_data_collator
            from torch.utils.data import DataLoader

            print("Evaluating Ethics loss...")
            dataset = get_ethics_contrastive_dataset(tokenizer, split='test')
            dataloader = DataLoader(
                dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator
            )

            loss = 0
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch_loss = ethics_contrastive_loss(model=model, tokenizer=tokenizer, batch=batch)
                    loss += batch_loss.item()
            
            mean_loss = loss / len(dataloader)
            print(colored(f"Ethics loss: {mean_loss:.4f}", "green"))

            from utils.performance_functions.truth_dataset import (
                get_truthfulness_contrastive_dataset,
                truthfulness_contrastive_loss,
            )

            print("Evaluating truthfulness contrastive mean loss...")

            # Prepare dataset & dataloader
            truth_test = get_truthfulness_contrastive_dataset(tokenizer, split="test")
            loader = DataLoader(
                truth_test,
                batch_size=8,
                shuffle=False,
                collate_fn=default_data_collator,
            )

            # Accumulate loss across evaluation set
            total_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch_loss = truthfulness_contrastive_loss(
                        model=model, batch=batch
                    )
                    total_loss += batch_loss.item()

            mean_truth_loss = total_loss / len(loader)
            print(
                colored(
                    f"Truthfulness contrastive mean loss: {mean_truth_loss:.4f}", "green"
                )
            )
        else:
            raise ValueError(f"Task evaluation is currently only supported for 'bias', 'ethics', 'truth' and 'combined' tasks, but TASK_NAME is '{TASK_NAME}'.")


if __name__ == "__main__":
    main()