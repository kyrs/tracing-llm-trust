from pathlib import Path
import torch
import json
from termcolor import colored
from safetensors.torch import load_file as safe_load_file
from typing import List, Dict, Optional
    
def get_indices_for_combined_scores(SCORES_TYPE: str, ANALYSIS_NAMES: List[str], BOTTOM_K: int, IF_WEIGHT: float, METRIC_WEIGHTS: Optional[Dict[str, float]] = None):
    print(colored("This method is implemented for only truth, bias, and ethics tasks", "yellow"))
    
    if SCORES_TYPE == "pure_ekfac":
        scores = {}
        for ANALYSIS_NAME in ANALYSIS_NAMES:
            scores_path = Path(
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

                scores[ANALYSIS_NAME] = scores_tensor.to(torch.float32).squeeze()  # (N,)
            else:
                raise FileNotFoundError(colored(f"Scores file not found at {scores_path}", "red"))
        # Normalize scores for each task using Min-Max scaling
        normalized_scores = {}
        for task_name, task_scores in scores.items():
            min_val = torch.min(task_scores)
            max_val = torch.max(task_scores)
            normalized_scores[task_name] = (task_scores - min_val) / (max_val - min_val)

        # Combine scores with optional metric weights per base metric name (bias/truth/ethics)
        metric_keys = ["bias", "truth", "ethics"]
        def infer_metric_key(name: str):
            lname = name.lower()
            for k in metric_keys:
                if k in lname:
                    return k
            return None

        # Default all weights to 1.0 if not provided
        weights_to_use: Dict[str, float] = {k: 1.0 for k in metric_keys}
        if METRIC_WEIGHTS is not None:
            for k in metric_keys:
                if k in METRIC_WEIGHTS:
                    weights_to_use[k] = float(METRIC_WEIGHTS[k])

        # Print weights applied
        print(colored(f"Using metric weights: {weights_to_use}", "cyan"))

        # Weighted sum of normalized scores
        first_tensor = next(iter(normalized_scores.values()))
        combined_scores = torch.zeros_like(first_tensor)
        for task_name, task_scores in normalized_scores.items():
            mk = infer_metric_key(task_name)
            w = weights_to_use.get(mk, 1.0) if mk is not None else 1.0
            combined_scores = combined_scores + (w * task_scores)
        indices = torch.topk(combined_scores, k=BOTTOM_K, largest=False).indices.tolist()

        # Compute per-task ranks (1 = lowest score) and print a table for selected indices
        ranks_per_task = {}
        for task_name, task_scores in scores.items():
            order = torch.argsort(task_scores, descending=False)
            ranks = torch.empty_like(order, dtype=torch.long)
            ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.long)
            ranks_per_task[task_name] = ranks

        task_names = list(scores.keys())
        print("Ranks within each metric (1 = lowest score):")
        print(" | ".join(["index"] + task_names))
        for idx in indices:
            row = [str(idx)] + [str(int(ranks_per_task[t][idx].item())) for t in task_names]
            print(" | ".join(row))

        # Report overlap between combined subset and bottom-100 per-metric indices
        for metric_key in ["bias", "ethics", "truth"]:
            task_name = next((n for n in task_names if metric_key in n.lower()), None)
            if task_name is not None:
                k = min(100, scores[task_name].numel())
                bottom_indices = torch.topk(scores[task_name], k=k, largest=False).indices.tolist()
                overlap_count = len(set(indices).intersection(bottom_indices))
                print(colored(f"Overlap with {metric_key} bottom-{k}: {overlap_count} out of {len(indices)} in combined subset", "cyan"))
            else:
                print(colored(f"Could not find a {metric_key} task key to compute overlap.", "red"))
    else:
        raise NotImplementedError(f"SCORES_TYPE {SCORES_TYPE} not implemented.")
    print(f"Bottom-{BOTTOM_K} influential training indices:", indices[:100], "...")
    
    return indices