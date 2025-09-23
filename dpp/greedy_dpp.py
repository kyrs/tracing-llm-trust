
import math, heapq, argparse, random, time
from typing import List, Tuple, Optional
import torch
from typing import Optional, Union
import numpy as np 
from datasets import load_dataset
from safetensors.torch import load_file as safe_load_file
from pathlib import Path

class RFF(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 D: int,
                 gamma: float,
                 *,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        W = torch.randn(in_dim, D, generator=gen, device=device, dtype=dtype)
        b = torch.rand(D,            generator=gen, device=device, dtype=dtype)
        W.mul_(math.sqrt(2 * gamma))      # N(0, 2γ I)
        b.mul_(2 * math.pi)               # U[0, 2π)

        self.register_buffer("W", W, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self.scale = math.sqrt(2.0 / D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:   # (n,d) → (n,D)
        return self.scale * torch.cos(X @ self.W + self.b)


# ── helpers with vectorised dot-products ────────────────────────────────────
def _k_row(Z: torch.Tensor, idx: int, S: List[int]) -> torch.Tensor:
    """Return (Z[S]·Z[idx]) as a 1-D tensor without Python loops."""
    return Z[S] @ Z[idx]             # (|S|,)

def marginal_gain_dpp(Z: torch.Tensor,
                      idx: int,
                      selected: List[int],
                      L: torch.Tensor,
                      eps: float) -> float:
    z_i = Z[idx]
    if not selected:
        return math.log(max(torch.dot(z_i, z_i).item() +1, eps)) ## eigen value +1 

    k_iS = _k_row(Z, idx, selected)                    # vectorised
    y = torch.linalg.solve_triangular(L,
                                      k_iS[:, None],
                                      upper=False).squeeze(1)
    delta = torch.dot(z_i, z_i)+1 - torch.dot(y, y) ## eigen value +1
    return math.log(max(delta.item(), eps))

def cholesky_update(Z: torch.Tensor,
                    selected: List[int],
                    L_prev: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """Rank-1 Cholesky update in O(k²)."""
    if L_prev.numel() == 0:                            # first element
        diag = math.sqrt(max(torch.dot(Z[selected[0]],
                                       Z[selected[0]]).item()+1, eps)) ## eigen value +1
        return Z.new_full((1, 1), diag)

    z_new = Z[selected[-1]]
    v     = _k_row(Z, selected[-1], selected[:-1])     # vectorised
    w     = torch.linalg.solve_triangular(L_prev,
                                          v[:, None],
                                          upper=False).squeeze(1)
    alpha = torch.dot(z_new, z_new)+1 - torch.dot(w, w) ## eigen value +1
    alpha = max(alpha.item(), eps)

    k  = L_prev.size(0)
    L  = torch.empty(k + 1, k + 1, device=Z.device, dtype=Z.dtype)
    L[:k, :k] = L_prev
    L[k, :k]  = w
    L[k, k]   = math.sqrt(alpha)
    return L

# ── fast lazy-greedy ────────────────────────────────────────────────────────
def lazy_greedy(Z: torch.Tensor,
                k: int,
                if_score_avg: torch.Tensor,
                ifWeight: float = 1.0,
                *,
                eps: float = 1e-12,
                verbose: bool = False) -> Tuple[List[int], float]:
    """
    Select k items with lazy-greedy DPP + additive IF score.
    Vectorised, heap stores pure Python floats → ~20–30× faster.
    """
    n = Z.size(0)
    # ||z_i||² as Python float list
    gains0  = (Z * Z).sum(1).cpu().tolist()
    gains0  = [g + 1.0 for g in gains0] # eigen value +1
    if_avg  = if_score_avg.cpu().tolist()              #  → Python floats

    assert (len(flag_weird_values(gains0))==0), "Check the embedding their is issue in it."
    assert(len(flag_weird_values(if_avg))==0), "Check the if weight their is an issue in it."

    # initial upper bounds in the heap (negated for max-heap behaviour)
    heap = [(-(math.log(max(g, eps)) + ifWeight*math.log1p(if_avg[i])), i, 0)       # (bound, idx, stale)
            for i, g in enumerate(gains0)]
    
    heapq.heapify(heap)
    
    selected: List[int] = []
    L   = Z.new_empty(0, 0)        # Cholesky factor of K_S
    W   = 0.0                      # running sum of IF scores
    dppGain = 0.0
    ifGain  = 0.0
    exact   = 0.0
    old_gain = float('inf')
    def mod_gain(j: int, W_: float) -> float:
        return math.log1p(W_ + if_avg[j]) - (math.log1p(W_) if W_ else 0.0) # for stability
    for t in range(k):
        # tighten bounds …
        while True:
            neg_ub, i, stale = heapq.heappop(heap)
            if stale == t:
                break
            bound = marginal_gain_dpp(Z, i, selected, L, eps) + ifWeight * mod_gain(i, W)
            heapq.heappush(heap, (-bound, i, t))

        # ----- i is the item we are about to add -----
        true_dpp   = marginal_gain_dpp(Z, i, selected, L, eps)
        true_if    = ifWeight * mod_gain(i, W)
        true_gain  = true_dpp + true_if
         
        assert(old_gain >= true_gain), f" Submodulariity getting violeted : old_gain {old_gain} < true_gain {true_gain} for t= {t}"
        print(f"[{t+1:4}] gain={true_gain: .6f} old_gain={old_gain: .6f}  dpp={true_dpp: .6f}  if={true_if: .6f}   W={W: .6f}  |S|={len(selected):>3}", end="\r", flush=True)
        old_gain = true_gain
        selected.append(i)
        W += if_avg[i]
        L  = cholesky_update(Z, selected, L, eps)

    
    logdet = (2 * torch.log(torch.diag(L)).sum()).item()
    return selected, logdet

def load_scores(scores_path):
    dataset = load_dataset("Dahoas/static-hh")
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
        assert len(scores) == len(
            dataset["train"]
        ), "Scores size ≠ #train examples, please check `scores_path`."

    else:
        raise FileNotFoundError(
            f"Influence scores not found at {scores_path}.\n"
            "Either generate them first or update `scores_path`."
        )
    return scores

# ────────────────────── heuristics & diagnostics ───────────────────────────

def median_heuristic_gamma(X: torch.Tensor, m: int = 1_000) -> float:
    """median heuristic for gamma"""
    idx = torch.randperm(X.size(0))[:m]
    subset = X[idx]
    dists = torch.cdist(subset, subset)  # Euclidean
    median = torch.median(dists).item()
    return 1.0 / (median ** 2 + 1e-12)


def quick_test(device="cpu"):
    n, d, k = 2_000, 30, 30
    X = torch.randn(n, d, device=device)
    γ = median_heuristic_gamma(X)
    Z = rff_transform(X, γ, D=512)
    sel_g, ld_g = lazy_greedy(Z, k)
    sel_r = torch.randperm(n)[:k].tolist()
    ld_r = 2 * torch.log(torch.diag(torch.linalg.cholesky(Z[sel_r] @ Z[sel_r].T))).sum().item()
    print("Quick-test results (RFF log-det):")
    print(f"  greedy: {ld_g:.3f}\n  random: {ld_r:.3f}\n  lift  : {ld_g - ld_r:.3f}")

# ────────────────────────────── main ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Lazy-greedy DPP with RBF kernel (RFF approx)")
    # p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--d", type=int, default=2_048)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--ifweight", type=float, default=10)
    p.add_argument("--gamma", type=float, default=None,
                   help="RBF γ; if omitted we use the median heuristic")
    p.add_argument("--D", type=int, default=2_048, help="#Random Fourier features")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick-test", action="store_true", help="Run small sanity test and exit")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    torch.manual_seed(42)

    if args.quick_test:
        quick_test(device=args.device)
        return

    MODEL_NAME = ... # hf_username/model_name
    TASK_NAME = "bias" # "ethics", "bias", "truth", "combined" etc.


    ANALYSIS_NAME = f"{TASK_NAME}_{MODEL_NAME.split('/')[-1]}" 
    REPRESENTATIONS_PATH = f"../representations/{MODEL_NAME.split('/')[-1]}_sumreps.pt"
    data_dict = torch.load(REPRESENTATIONS_PATH, map_location="cpu")
    X = data_dict["representations"].to(args.device)  # Tensor [N, hidden]
    # Sanity-check the hidden state dimensionality before proceeding.
    assert X.shape[1] == args.d, (
        f"Hidden dimension mismatch: expected {args.d} (from --d) but got {X.shape[1]} in the loaded representations. "
        "Make sure you pass the correct --d argument or load representations with the matching hidden size."
    )
    print(X.shape)

    scores_path = Path(f"../ekfac_experiments/influence_results/{ANALYSIS_NAME}/scores_ekfac_half/pairwise_scores.safetensors")
    
    EkFACScore = load_scores(scores_path)
    EkFACScore = EkFACScore.view(-1, 1) # add the fairness dimension
    print(EkFACScore.shape)

    EkFKFCScoreAvg = torch.sum(EkFACScore, dim=1)          # (n,)

    EkFKFCScoreAvg = -1 * EkFKFCScoreAvg

    # normalise to [0, 1]
    EkFKFCScoreAvg = (EkFKFCScoreAvg - EkFKFCScoreAvg.min()) / (
                    EkFKFCScoreAvg.max() - EkFKFCScoreAvg.min() + 1e-12) 

    γ = args.gamma if args.gamma is not None else median_heuristic_gamma(X)
    print(f"Using γ = {γ:.3e}")

    print("Computing Random Fourier Features …")
    rff = torch.compile(RFF(args.d, args.D, γ, device=args.device))
    Z   = rff(X)

    start = time.time()
    print("running lazy-greedy DPP with modular fn …")
    S, logdet = lazy_greedy(Z, args.k, EkFKFCScoreAvg, ifWeight=args.ifweight, verbose=args.verbose)
    dt = time.time() - start
    print(f"Done in {dt/60:.1f} min  —  log|K_S|≈{logdet:.2f}")
    import json

    with open(f"{ANALYSIS_NAME}_indices_{args.k}_ifweight{args.ifweight}.json", "w") as f:
        json.dump({"indices": S}, f)
    print(f"Selected indices saved")

if __name__ == "__main__":
    main()