#!/usr/bin/env python3
"""
RE-IRL minimal script with:
- fixed VAL_MONITOR across rounds,
- pairwise Platt calibration + reliability,
- single-text Platt calibration + reliability,
- single-text midpoint decision threshold + confusion matrix,
- strict checks/prints for sorted dataset sizes.

This is a full replacement for the prior script.
"""

import os, sys, json, math
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions import constraints
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


try:
    from sklearn.metrics import roc_auc_score, confusion_matrix
except Exception:
    roc_auc_score = None
    confusion_matrix = None

from irl_pipeline.irl.ground_truth import GroundTruthGenerator
from irl_pipeline.irl.reward_computer import SimplifiedIRLRewardComputer
from irl_pipeline.irl.bt_likelihood import _log_likelihood_bradley_terry


# -------------------------- utils --------------------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def sigmoid_np(x: np.ndarray):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def stratified_holdout_by_label(labels: np.ndarray, seed=42,
                                frac_cal=0.10, frac_test=0.10):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    classes = np.unique(labels)
    cal_idx, test_idx, pool_idx = [], [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_cal = int(np.floor(frac_cal * n))
        n_test = int(np.floor(frac_test * n))
        cal_idx.append(idx_c[:n_cal])
        test_idx.append(idx_c[n_cal:n_cal + n_test])
        pool_idx.append(idx_c[n_cal + n_test:])
    cal = np.concatenate(cal_idx); rng.shuffle(cal)
    test = np.concatenate(test_idx); rng.shuffle(test)
    pool = np.concatenate(pool_idx); rng.shuffle(pool)
    return cal, test, pool

def stratified_take_from_indices(indices: np.ndarray, labels: np.ndarray,
                                 frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels_full = np.asarray(labels).astype(int)
    idx = np.asarray(indices)
    take, rest = [], []
    for c in np.unique(labels_full[idx]):
        idx_c = idx[labels_full[idx] == c]
        rng.shuffle(idx_c)
        n_take = int(np.floor(frac * len(idx_c)))
        take.append(idx_c[:n_take])
        rest.append(idx_c[n_take:])
    take = np.concatenate(take) if take else np.array([], dtype=int)
    rest = np.concatenate(rest) if rest else np.array([], dtype=int)
    rng.shuffle(take); rng.shuffle(rest)
    return take, rest

def stratified_kfold_indices_by_label(labels: np.ndarray, k: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    classes = np.unique(labels)
    per_class_splits: Dict[int, List[np.ndarray]] = {}
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        per_class_splits[c] = np.array_split(idx_c, k)
    folds = []
    for i in range(k):
        fold_parts = [per_class_splits[c][i] for c in classes]
        fold = np.concatenate(fold_parts) if len(fold_parts) else np.array([], dtype=int)
        rng.shuffle(fold)
        folds.append(fold)
    return folds

def expected_pairwise_probs(delta_phi: torch.Tensor, theta_samples: torch.Tensor,
                            T: float, bs: int = 4096, alpha: float = 1.0) -> np.ndarray:
    device = theta_samples.device
    delta_phi = delta_phi.to(device)
    out = []
    for s in range(0, delta_phi.shape[0], bs):
        chunk = delta_phi[s:s + bs]
        logits = alpha * (theta_samples @ chunk.T) / T   # <-- include alpha
        out.append(torch.sigmoid(logits).mean(0).detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def expected_scores(phi: torch.Tensor, theta_samples: torch.Tensor) -> np.ndarray:
    phi = phi.to(theta_samples.device)
    return (theta_samples @ phi.T).mean(0).detach().cpu().numpy()

def platt_calibrate(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    def nll(ab):
        a, b = ab
        z = np.clip(a * logits + b, -40, 40)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-7
        return -np.mean(targets * np.log(np.clip(p, eps, 1 - eps)) + (1 - targets) * np.log(np.clip(1 - p, eps, 1 - eps)))
    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(0.0, None), (None, None)])
    return float(res.x[0]), float(res.x[1])

def temp_calibrate(logits: np.ndarray, targets: np.ndarray) -> float:
    def ce(T):
        z = logits / T
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        eps = 1e-7
        return -np.mean(targets * np.log(np.clip(p, eps, 1 - eps)) + (1 - targets) * np.log(np.clip(1 - p, eps, 1 - eps)))
    res = minimize_scalar(ce, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)

def plot_reliability_curve_quantile(p_hat: np.ndarray, y: np.ndarray, title: str, path: Path, n_bins: int = 10, color: str = None):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p_hat, qs))
    xs, ys = [], []
    for i in range(len(edges) - 1):
        m = (p_hat >= edges[i]) & (p_hat <= (edges[i + 1] if i + 1 == len(edges) - 1 else edges[i + 1]))
        if m.any():
            xs.append(p_hat[m].mean()); ys.append(y[m].mean())
    
    # Save reliability curve data for combined plots
    reliability_data = {'xs': xs, 'ys': ys, 'title': title}
    data_path = path.parent / f"{path.stem}_data.npy"
    np.save(data_path, reliability_data, allow_pickle=True)
    
    plt.figure(figsize=(5.2, 4))
    plt.plot([0, 1], [0, 1], '--', alpha=0.6)
    plt.plot(xs, ys, 'o-', color=color)
    plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=220); plt.close()

def get_model_color(model_name: str) -> str:
    """Get consistent color for model based on name."""
    if "SmolLM-360M" in model_name or "360M" in model_name:
        return "#1f77b4"  # blue
    elif "SmolLM-135M" in model_name or "135M" in model_name:
        return "#ff7f0e"  # orange
    elif "pythia-70m" in model_name or "70m" in model_name:
        return "#2ca02c"  # green
    elif "pythia-410m" in model_name or "410m" in model_name:
        return "#d62728"  # red
    elif "Llama" in model_name or "llama" in model_name:
        return "#9467bd"  # purple
    elif "pythia-1b" in model_name or "1b" in model_name:
        return "#8c564b"  # brown
    else:
        return None  # default color

def plot_reward_distribution_violin(R_tox: np.ndarray, R_nont: np.ndarray, model_name: str, path: Path):
    """Create violin plot for reward distributions (toxic vs non-toxic) with decision boundary."""
    import seaborn as sns
    
    # Calculate decision boundary (midpoint between means)
    decision_boundary = 0.5 * (R_tox.mean() + R_nont.mean())
    
    # Prepare data for violin plot
    all_rewards = np.concatenate([R_tox, R_nont])
    categories = ['Toxic'] * len(R_tox) + ['Non-Toxic'] * len(R_nont)
    
    # Create DataFrame
    df = pd.DataFrame({'Category': categories, 'Reward Score R(o)': all_rewards})
    
    # Create violin plot with box overlay
    plt.figure(figsize=(6, 6))
    ax = sns.violinplot(
        x="Category", y="Reward Score R(o)", data=df, hue="Category",
        palette={"Toxic": "#EEC8B0", "Non-Toxic": "#C8DAF0"},
        inner=None, saturation=1, legend=False
    )
    
    # Add box plot overlay
    sns.boxplot(
        x="Category", y="Reward Score R(o)", data=df, hue="Category",
        width=0.3, showcaps=True,
        boxprops={'facecolor':'none', 'edgecolor':'k'},
        showfliers=False, whiskerprops={'linewidth':2}, ax=ax,
        palette={"Toxic": "#EEC8B0", "Non-Toxic": "#C8DAF0"},
        saturation=1, legend=False
    )
    
    # Add decision boundary line
    ax.axhline(y=decision_boundary, color='red', linestyle='--', alpha=0.6, linewidth=2, 
               label=f'Decision Boundary ({decision_boundary:.2f})')
    
    # Styling
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Reward Score R(o)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f"Reward Score Distribution: {model_name.split('/')[-1]}", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()

def plot_combined_reliability_curves(p_single: np.ndarray, y_single: np.ndarray, 
                                   p_pair: np.ndarray, y_pair: np.ndarray,
                                   model_name: str, path: Path, n_bins: int = 10):
    """Create combined reliability curves for single-text and pairwise."""
    color = get_model_color(model_name)
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Single-text reliability
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p_single, qs))
    xs, ys = [], []
    for i in range(len(edges) - 1):
        m = (p_single >= edges[i]) & (p_single <= (edges[i + 1] if i + 1 == len(edges) - 1 else edges[i + 1]))
        if m.any():
            xs.append(p_single[m].mean()); ys.append(y_single[m].mean())
    
    ax1.plot([0, 1], [0, 1], '--', alpha=0.6, color='gray')
    ax1.plot(xs, ys, 'o-', color=color)
    ax1.set_xlabel('Predicted probability'); ax1.set_ylabel('Observed frequency')
    ax1.set_title(f'Single-Text Reliability Curve for {model_short}')
    ax1.grid(alpha=0.3)
    
    # Pairwise reliability
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p_pair, qs))
    xs, ys = [], []
    for i in range(len(edges) - 1):
        m = (p_pair >= edges[i]) & (p_pair <= (edges[i + 1] if i + 1 == len(edges) - 1 else edges[i + 1]))
        if m.any():
            xs.append(p_pair[m].mean()); ys.append(y_pair[m].mean())
    
    ax2.plot([0, 1], [0, 1], '--', alpha=0.6, color='gray')
    ax2.plot(xs, ys, 'o-', color=color)
    ax2.set_xlabel('Predicted probability'); ax2.set_ylabel('Observed frequency')
    ax2.set_title(f'Pairwise Reliability Curve for {model_short}')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()

def plot_confmat(cm: np.ndarray, title: str, path: Path, class_names=('toxic','non-toxic')):
    plt.figure(figsize=(4.4, 4.0))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names); plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean((p - y) ** 2))

def ece_quantile(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # quantile bin edges based on predicted probs
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    if len(edges) < 2:
        return 0.0
    ece = 0.0
    N = len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1] if i + 1 < len(edges) - 1 else edges[i + 1] + 1e-12
        mask = (p >= lo) & (p < hi) if i + 1 < len(edges) - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask): 
            continue
        p_bin = p[mask].mean()
        y_bin = y[mask].mean()
        w = mask.mean()  # weight by fraction of samples in the bin
        ece += w * abs(p_bin - y_bin)
    return float(ece)


def pairwise_acc(x: np.ndarray, num_pairs: int = 5000, seed: int = 0) -> tuple[float, float]:
    """
    P(x_i > x_j) for random pairs (i != j) drawn from the SAME set x,
    plus the mean absolute gap |x_i - x_j|.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    j = np.where(j == i, (j + 1) % n, j)
    wins = (x[i] > x[j]).mean()
    gap = np.abs(x[i] - x[j]).mean()
    return float(wins), float(gap)





# -------------------------- VI --------------------------

def train_vi_bt(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    dp_train: torch.Tensor,
    pgt_train: np.ndarray,
    dp_val_fixed: torch.Tensor,
    pgt_val_fixed: np.ndarray,
    d: int,
    prior_mean: torch.Tensor | None,
    prior_std: torch.Tensor | None,
    n_steps=3000,
    lr=1e-2,
    alpha=1.0,
    learn_T=True,
    eval_every=100,
    device="cpu",
):
    pyro.clear_param_store()
    demos = [(b.to(device), g.to(device)) for b, g in demos]
    if prior_mean is None: prior_mean = torch.zeros(d, device=device)
    if prior_std  is None: prior_std  = torch.ones(d, device=device)

    def model():
        theta = pyro.sample("theta", dist.Normal(prior_mean, prior_std).to_event(1))
        T = 1.0
        if learn_T:
            logT = pyro.param("logT", torch.tensor(math.log(2.0), device=device))
            pyro.factor("logT_prior", dist.Normal(math.log(2.0), 1.0).log_prob(logT))
            T = torch.exp(logT)
        ll = _log_likelihood_bradley_terry(theta, demos, alpha, temperature=T)
        pyro.factor("bt_ll", ll)
        return theta

    def guide():
        mu_q = pyro.param("mu_q", prior_mean.clone())
        sig_q = pyro.param("sig_q", prior_std.clone(), constraint=constraints.positive)
        pyro.sample("theta", dist.Normal(mu_q, sig_q).to_event(1))
        if learn_T: _ = pyro.param("logT")

    opt = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, opt, loss=Trace_ELBO())

    history = {"step": [], "loss": [], "acc_train": [], "acc_val_fixed": [], "mu_norm": []}

    for t in range(1, n_steps + 1):
        loss = svi.step()
        if t % eval_every == 0 or t == 1 or t == n_steps:
            mu = pyro.param("mu_q").detach(); sig = pyro.param("sig_q").detach()
            S = 64
            theta_s = dist.Normal(mu, sig).sample((S,))
            T_eval = float(torch.exp(pyro.param("logT")).item()) if learn_T else 1.0

            p_tr = expected_pairwise_probs(dp_train,     theta_s, T_eval, alpha=alpha)
            p_vf = expected_pairwise_probs(dp_val_fixed, theta_s, T_eval, alpha=alpha)
            acc_tr = float(((p_tr >= 0.5) == (pgt_train >= 0.5)).mean()) if len(p_tr) else float("nan")
            acc_vf = float(((p_vf >= 0.5) == (pgt_val_fixed >= 0.5)).mean()) if len(p_vf) else float("nan")

            history["step"].append(t)
            history["loss"].append(float(loss))
            history["acc_train"].append(acc_tr)
            history["acc_val_fixed"].append(acc_vf)
            history["mu_norm"].append(float(mu.norm().item()))
            print(f"[step {t:5d}] loss={loss:,.1f} | acc_tr={acc_tr:.3f} acc_val_fixed={acc_vf:.3f} | ||mu||={history['mu_norm'][-1]:.3f}")

    mu = pyro.param("mu_q").detach().cpu()
    sig = pyro.param("sig_q").detach().cpu()
    T = float(torch.exp(pyro.param("logT")).item()) if learn_T else 1.0
    return {"mu": mu, "sig": sig, "T": T, "history": history}


# -------------------------- helper functions --------------------------

def get_dataset_paths(model_name: str, detox_name: str,
                      cache_dir: str = "datasets",
                      train_samples: int | None = None) -> Dict[str, str]:
    """
    Prefer exact filenames if train_samples is given (…_{N}_samples_…),
    otherwise fall back to the first file matching …_*_samples_….
    """
    import glob, os
    def _first(pat: str) -> str | None:
        c = sorted(glob.glob(pat))
        return c[0] if c else None

    model_safe = model_name.replace('/', '_')
    detox_safe = detox_name.replace('/', '_')

    if train_samples is not None:
        exact_orig  = os.path.join(cache_dir, f"{model_safe}_{train_samples}_samples_original.json")
        exact_detox = os.path.join(cache_dir, f"{detox_safe}_{train_samples}_samples_detoxified.json")
        print(f"Using dataset according to train_samples: {train_samples}")
    else:
        exact_orig = exact_detox = None

    orig  = exact_orig  if (exact_orig  and os.path.exists(exact_orig))  else _first(os.path.join(cache_dir, f"{model_safe}_*_samples_original.json"))
    detox = exact_detox if (exact_detox and os.path.exists(exact_detox)) else _first(os.path.join(cache_dir, f"{detox_safe}_*_samples_detoxified.json"))
    return {
        'original_dataset_path': orig or os.path.join(cache_dir, f"{model_safe}_*_samples_original.json"),
        'detoxified_dataset_path': detox or os.path.join(cache_dir, f"{detox_safe}_*_samples_detoxified.json"),
        'sorted_toxic_dataset_path': os.path.join(cache_dir, f"sorted_toxic_dataset_{model_safe}.json"),
        'sorted_non_toxic_dataset_path': os.path.join(cache_dir, f"sorted_non_toxic_dataset_{model_safe}.json"),
    }


def get_model_short_name(model_name: str) -> str:
    """Get short name for output folders."""
    return model_name.replace('/', '_').replace('-', '_')


def _auto_fill_model_params(cfg):
    """
    If model.hidden_size (or detox_name) is missing, pull it from cfg.model.models
    by matching model.base_model_name.
    """
    base = cfg.model.base_model_name
    # Try to find an entry with matching name
    match = None
    for m in getattr(cfg.model, "models", []):
        if m.get("name") == base:
            match = m
            break

    # Fill hidden_size if missing
    if getattr(cfg.model, "hidden_size", None) in (None, 0) and match and "hidden_size" in match:
        cfg.model.hidden_size = match["hidden_size"]

    # Sanity check
    if getattr(cfg.model, "hidden_size", None) in (None, 0):
        raise ValueError(
            f"model.hidden_size is not set and could not be inferred for base_model_name={base}. "
            f"Add it to configs/re_irl_config.yaml -> model.models or pass model.hidden_size=..."
        )


def create_individual_plots(summary: List[Dict], out_dir: Path):
    """Create plots for individual model runs across rounds."""
    plots_dir = out_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    rounds = [s["round"] for s in summary]

    # --- Always build per-model combined top-k files, even if only 1 round
    create_topk_uncertainty_summary(out_dir)

    # If only 1 round, skip the trend figures but keep histograms below
    if len(summary) <= 1:
        pass  # fall through to histograms (and we already built top-k)
    else:
        # Posterior uncertainty plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Posterior contraction (trace)
        axes[0,0].plot(rounds, [s["posterior_trace_Sigma"] for s in summary], 'o-')
        axes[0,0].set_title("Posterior Contraction (Trace)")
        axes[0,0].set_xlabel("Round"); axes[0,0].set_ylabel("trace(Σ)")
        axes[0,0].grid(alpha=0.3)

        # Log det Σ
        axes[0,1].plot(rounds, [s["posterior_logdet_Sigma"] for s in summary], 'o-', color='orange')
        axes[0,1].set_title("Posterior Tightness")
        axes[0,1].set_xlabel("Round"); axes[0,1].set_ylabel("log det(Σ)")
        axes[0,1].grid(alpha=0.3)

        # Epistemic uncertainty (MI medians) — add Sorted-pair MI if present
        axes[1,0].plot(rounds, [s["pred_single_p_MI_median"] for s in summary], 'o-', label='Single-text MI')
        axes[1,0].plot(rounds, [s["pred_pair_p_MI_median"] for s in summary], 'o-', label='Pairwise (orig–detox) MI')

        # Show/hide Sorted-pair line by env toggle and data availability
        show_sorted = bool(int(os.environ.get("SHOW_SORTED_PAIR_PLOTS", "1")))
        if show_sorted and any("sorted_pair_p_MI_median" in s for s in summary):
            axes[1,0].plot(rounds,
                           [s.get("sorted_pair_p_MI_median", float('nan')) for s in summary],
                           's--', label='Sorted-pair (nontox vs tox) MI')

        axes[1,0].set_title("Epistemic Uncertainty (MI)")
        axes[1,0].set_xlabel("Round"); axes[1,0].set_ylabel("Median MI")
        axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

        # Performance metrics
        axes[1,1].plot(rounds, [s["single_auroc_TEST"] for s in summary], 'o-', label='AUROC')
        axes[1,1].plot(rounds, [s["single_acc_midpoint_TEST"] for s in summary], 'o-', label='Single Acc')
        axes[1,1].plot(rounds, [s["pair_acc_platt"] for s in summary], 'o-', label='Pair Acc')
        axes[1,1].plot(rounds, [s["single_f1_macro_TEST"] for s in summary], 'o-', label='F1')
        axes[1,1].set_title("Performance Metrics")
        axes[1,1].set_xlabel("Round"); axes[1,1].set_ylabel("Score")
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "uncertainty_performance_trends.png", dpi=220, bbox_inches='tight')
        plt.close()

        # Calibration metrics
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(rounds, [s["single_brier_platt_test"] for s in summary], 'o-', label='Single Brier')
        plt.plot(rounds, [s["pair_brier_platt_test"] for s in summary], 'o-', label='Pair Brier')
        plt.title("Brier Scores"); plt.xlabel("Round"); plt.ylabel("Brier Score")
        plt.legend(); plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(rounds, [s["single_ece_platt_q_test"] for s in summary], 'o-', label='Single ECE')
        plt.plot(rounds, [s["pair_ece_platt_q_test"] for s in summary], 'o-', label='Pair ECE')
        plt.title("ECE Scores"); plt.xlabel("Round"); plt.ylabel("ECE")
        plt.legend(); plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "calibration_trends.png", dpi=220, bbox_inches='tight')
        plt.close()

    # --- Predictive entropy distributions per round (rows)
    try:
        show_sorted = bool(int(os.environ.get("SHOW_SORTED_PAIR_PLOTS", "1")))
        have_sorted_row = show_sorted and any((out_dir / f"round_{r}" / "sorted_pair_H_total_TEST.npy").exists()
                                              for r in rounds)
        nrows = 3 if have_sorted_row else 2

        fig, axes = plt.subplots(nrows, len(summary), figsize=(4*len(summary), 4*nrows))
        axes = np.array(axes).reshape(nrows, len(summary))

        for i, r in enumerate(rounds):
            rd = out_dir / f"round_{r}"

            # Row 1: single-text entropy
            try:
                single_H_total = np.load(rd / "single_H_total_TEST.npy")
                axes[0, i].hist(single_H_total, bins=20, density=True, alpha=0.7)
                axes[0, i].set_title(f"Round {r}\nSingle H_total")
            except FileNotFoundError:
                axes[0, i].text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=axes[0, i].transAxes)

            # Row 2: pairwise (orig–detox)
            try:
                pair_H_total = np.load(rd / "pair_H_total_TEST.npy")
                axes[1, i].hist(pair_H_total, bins=20, density=True, alpha=0.7, color="orange")
                axes[1, i].set_title("Pair H_total")
            except FileNotFoundError:
                axes[1, i].text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=axes[1, i].transAxes)

            # Row 3: sorted pairs (nontox vs tox)
            if nrows == 3:
                try:
                    sorted_pair_H_total = np.load(rd / "sorted_pair_H_total_TEST.npy")
                    axes[2, i].hist(sorted_pair_H_total, bins=20, density=True, alpha=0.7)
                    axes[2, i].set_title("Sorted-pair H_total")
                    axes[2, i].set_xlabel("Total Entropy")
                except FileNotFoundError:
                    axes[2, i].text(0.5, 0.5, "No data", ha="center", va="center",
                                    transform=axes[2, i].transAxes)

            for rrow in range(nrows):
                if i == 0:
                    axes[rrow, i].set_ylabel("Density")
                axes[rrow, i].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "entropy_distributions_per_round.png", dpi=220, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create entropy distributions: {e}")

    print(f"📈 Individual analysis plots saved to {plots_dir}")



        
    # Create top-k uncertainty examples summary
    create_topk_uncertainty_summary(out_dir)
    
    print(f"📈 Individual analysis plots saved to {plots_dir}")

def create_topk_uncertainty_summary(out_dir: Path):
    """Create summary of top-k uncertainty examples across rounds."""
    try:
        topk_dir = out_dir / "analysis_plots" / "topk_examples"
        topk_dir.mkdir(exist_ok=True)
        
        # Collect top-k examples from all rounds
        all_single_ep, all_single_al = [], []
        all_pair_ep, all_pair_al = [], []
        
        for round_dir in out_dir.glob("round_*"):
            # Load top-k files if they exist
            for file_name, target_list in [
                ("topk_single_epistemic.json", all_single_ep),
                ("topk_single_aleatoric.json", all_single_al),
                ("topk_pair_epistemic.json", all_pair_ep),
                ("topk_pair_aleatoric.json", all_pair_al)
            ]:
                file_path = round_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Add round info to each example
                        round_num = int(round_dir.name.split('_')[-1])
                        for item in data:
                            item['round'] = round_num
                        target_list.extend(data)
        
        # Save combined top-k summaries
        for data, name in [
            (all_single_ep, "combined_single_epistemic_topk.json"),
            (all_single_al, "combined_single_aleatoric_topk.json"), 
            (all_pair_ep, "combined_pair_epistemic_topk.json"),
            (all_pair_al, "combined_pair_aleatoric_topk.json")
        ]:
            if data:
                with open(topk_dir / name, 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        
        if any([all_single_ep, all_single_al, all_pair_ep, all_pair_al]):
            print(f"📋 Top-k uncertainty examples saved to {topk_dir}")
    except Exception as e:
        print(f"⚠️  Could not create top-k summary: {e}")




def create_combined_analysis_plots(all_results: Dict, combined_dir: Path):
    """Create combined analysis plots across all models."""
    analysis_dir = combined_dir / "analysis_plots"
    analysis_dir.mkdir(exist_ok=True)

    # Sort models by size for consistent legend ordering
    model_sizes = {name: get_model_size_order(name) for name in all_results.keys()}
    sorted_models = sorted(all_results.keys(), key=lambda x: model_sizes[x])

    # Use consistent model colors instead of tab10
    colors = [get_model_color(model_name) for model_name in sorted_models]
    show_sorted = bool(int(os.environ.get("SHOW_SORTED_PAIR_PLOTS", "1")))

    # Combined posterior uncertainty trends
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, model_name in enumerate(sorted_models):
        if not all_results[model_name]:
            continue
        summary = all_results[model_name]
        rounds = [s["round"] for s in summary]
        label = model_name.split('/')[-1]
        color = colors[i] if colors[i] is not None else plt.cm.tab10(i / len(sorted_models))

        # Trace
        axes[0,0].plot(rounds, [s["posterior_trace_Sigma"] for s in summary], 'o-',
                      color=color, label=label, alpha=0.7)
        # Log det
        axes[0,1].plot(rounds, [s["posterior_logdet_Sigma"] for s in summary], 'o-',
                      color=color, label=label, alpha=0.7)
        # MI medians
        axes[1,0].plot(rounds, [s["pred_single_p_MI_median"] for s in summary], 'o-',
                      color=color, label=f'{label} (Single)', alpha=0.7)
        axes[1,0].plot(rounds, [s["pred_pair_p_MI_median"] for s in summary], 's--',
                      color=color, label=f'{label} (Pair)', alpha=0.7)
        if show_sorted and any("sorted_pair_p_MI_median" in s for s in summary):
            axes[1,0].plot(rounds,
                           [s.get("sorted_pair_p_MI_median", float('nan')) for s in summary],
                           'd:', color=color, label=f'{label} (Sorted-pair)', alpha=0.7)

        # Performance
        axes[1,1].plot(rounds, [s["single_auroc_TEST"] for s in summary], 'o-',
                      color=color, label=label, alpha=0.7)

    axes[0,0].set_title("Posterior Contraction"); axes[0,0].set_ylabel("trace(Σ)")
    axes[0,1].set_title("Posterior Tightness");   axes[0,1].set_ylabel("log det(Σ)")
    axes[1,0].set_title("Epistemic Uncertainty"); axes[1,0].set_ylabel("Median MI")
    axes[1,1].set_title("AUROC Performance");     axes[1,1].set_ylabel("AUROC")

    for ax in axes.flat:
        ax.set_xlabel("Round"); ax.grid(alpha=0.3); ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(analysis_dir / "combined_uncertainty_trends.png", dpi=220, bbox_inches='tight')
    plt.close()

    # Combined entropy distributions comparison (add Sorted-pair as a 3rd panel if files exist)
    try:
        # Check if at least one model has sorted-pair files
        has_sorted_any = False
        for model_name in sorted_models:
            if not all_results[model_name]:
                continue
            model_dir = Path("re_irl_min_stratified_plots") / get_model_short_name(model_name)
            for s in all_results[model_name]:
                r = s["round"]
                if (model_dir / f"round_{r}" / "sorted_pair_H_total_TEST.npy").exists():
                    has_sorted_any = True
                    break
            if has_sorted_any:
                break

        ncols = 3 if show_sorted and has_sorted_any else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

        # Ensure axes is iterable
        if ncols == 2:
            ax_single, ax_pair = axes
            ax_sorted = None
        else:
            ax_single, ax_pair, ax_sorted = axes

        for i, model_name in enumerate(sorted_models):
            if not all_results[model_name]:
                continue
            summary = all_results[model_name]
            label = model_name.split('/')[-1]
            color = colors[i] if colors[i] is not None else plt.cm.tab10(i / len(sorted_models))

            model_dir = Path("re_irl_min_stratified_plots") / get_model_short_name(model_name)
            all_single_H, all_pair_H, all_sorted_H = [], [], []

            for round_data in summary:
                round_num = round_data["round"]
                round_dir = model_dir / f"round_{round_num}"
                try:
                    single_H = np.load(round_dir / 'single_H_total_TEST.npy')
                    all_single_H.extend(single_H)
                except Exception:
                    pass
                try:
                    pair_H = np.load(round_dir / 'pair_H_total_TEST.npy')
                    all_pair_H.extend(pair_H)
                except Exception:
                    pass
                if ax_sorted is not None:
                    try:
                        sorted_H = np.load(round_dir / 'sorted_pair_H_total_TEST.npy')
                        all_sorted_H.extend(sorted_H)
                    except Exception:
                        pass

            if all_single_H:
                ax_single.hist(all_single_H, bins=30, alpha=0.5, density=True, label=label, color=color)
            if all_pair_H:
                ax_pair.hist(all_pair_H, bins=30, alpha=0.5, density=True, label=label, color=color)
            if ax_sorted is not None and all_sorted_H:
                ax_sorted.hist(all_sorted_H, bins=30, alpha=0.5, density=True, label=label, color=color)

        ax_single.set_title("Single-text Entropy")
        ax_single.set_xlabel("Total Entropy"); ax_single.set_ylabel("Density"); ax_single.legend(); ax_single.grid(alpha=0.3)

        ax_pair.set_title("Pairwise Entropy (orig–detox)")
        ax_pair.set_xlabel("Total Entropy"); ax_pair.set_ylabel("Density"); ax_pair.legend(); ax_pair.grid(alpha=0.3)

        if ax_sorted is not None:
            ax_sorted.set_title("Sorted-pair Entropy (nontox vs tox)")
            ax_sorted.set_xlabel("Total Entropy"); ax_sorted.set_ylabel("Density"); ax_sorted.legend(); ax_sorted.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(analysis_dir / "combined_entropy_distributions.png", dpi=220, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create combined entropy distributions: {e}")

    # Combined top-k uncertainty examples (unchanged)
    create_combined_topk_summary(all_results, analysis_dir)
    print(f"📊 Combined analysis plots saved to {analysis_dir}")





def create_combined_topk_summary(all_results: Dict, analysis_dir: Path):
    """Create combined top-k uncertainty examples across all models."""
    try:
        topk_dir = analysis_dir / "combined_topk_examples"
        topk_dir.mkdir(exist_ok=True)
        
        combined_data = {
            "single_epistemic": [],
            "single_aleatoric": [],
            "pair_epistemic": [],
            "pair_aleatoric": []
        }
        
        for model_name in all_results.keys():
            if not all_results[model_name]:
                continue
            
            model_dir = Path("re_irl_min_stratified_plots") / get_model_short_name(model_name)
            topk_model_dir = model_dir / "analysis_plots" / "topk_examples"
            
            # Load combined top-k files for this model
            file_mapping = {
                "combined_single_epistemic_topk.json": "single_epistemic",
                "combined_single_aleatoric_topk.json": "single_aleatoric",
                "combined_pair_epistemic_topk.json": "pair_epistemic",
                "combined_pair_aleatoric_topk.json": "pair_aleatoric"
            }
            
            for file_name, key in file_mapping.items():
                file_path = topk_model_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Add model info to each example
                        for item in data:
                            item['model'] = model_name
                        combined_data[key].extend(data)
        
        # Save combined results across all models
        for key, data in combined_data.items():
            if data:
                # Sort by uncertainty score and take top examples
                if "epistemic" in key:
                    data.sort(key=lambda x: x.get('MI', 0), reverse=True)
                else:
                    data.sort(key=lambda x: x.get('H_cond', 0), reverse=True)
                
                with open(topk_dir / f"all_models_{key}_topk.json", 'w') as f:
                    json.dump(data[:50], f, ensure_ascii=False, indent=2)  # Top 50 across all models
        
        if any(combined_data.values()):
            print(f"📋 Combined top-k examples saved to {topk_dir}")
    except Exception as e:
        print(f"⚠️  Could not create combined top-k summary: {e}")

def get_model_size_order(model_name: str) -> int:
    """Get ordering by model size for consistent legend."""
    if "135M" in model_name: return 1
    elif "360M" in model_name or "125m" in model_name: return 2  
    elif "70m" in model_name: return 3
    elif "410m" in model_name: return 4
    elif "1b" in model_name or "1B" in model_name: return 5
    return 6

def create_combined_plots_and_summary(all_results: Dict, combined_dir: Path):
    """Create combined plots and summary table from all model results."""
    print(f"\n📊 Creating combined results in {combined_dir}")
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    table_data = []
    for model_name, results in all_results.items():
        if results:  # Check if results exist for this model
            result = results[0]  # Use first round data
            
            # Extract model size from name for sorting
            if "135M" in model_name:
                size_order = 1
                size_display = "135M"
            elif "360M" in model_name or "125m" in model_name:
                size_order = 2
                size_display = "360M" if "360M" in model_name else "125M"
            elif "70m" in model_name:
                size_order = 3
                size_display = "70M"
            elif "410m" in model_name:
                size_order = 4
                size_display = "410M"
            elif "1b" in model_name or "1B" in model_name:
                size_order = 5
                size_display = "1B"
            else:
                size_order = 6
                size_display = "Unknown"
            
            table_data.append({
                "Model": model_name.split('/')[-1],
                "Size": size_display,
                "Size_Order": size_order,
                "Pairwise_Acc_Calibrated": result.get("pair_acc_platt", 0),
                "Pairwise_ECE_Calibrated": result.get("pair_ece_platt_q_test", 0),
                "Single_Text_Acc": result.get("single_acc_midpoint_TEST", 0),
                "AUROC": result.get("single_auroc_TEST", 0),
                "Single_Text_F1": result.get("single_f1_macro_TEST", 0),
                "Single_Text_ECE": result.get("single_ece_platt_q_test", 0)
            })
    
    # Sort by model size
    table_data.sort(key=lambda x: x["Size_Order"])
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    df = df.drop("Size_Order", axis=1)  # Remove sorting column
    
    # Format numbers for display
    for col in ["Pairwise_Acc_Calibrated", "Pairwise_ECE_Calibrated", "Single_Text_Acc", "AUROC", "Single_Text_F1", "Single_Text_ECE"]:
        df[col] = df[col].round(3)
    
    df.to_csv(combined_dir / "model_comparison_table.csv", index=False)
    print(f"📋 Saved comparison table to {combined_dir / 'model_comparison_table.csv'}")
    
    # Save combined summary JSON
    with open(combined_dir / "combined_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"📄 Saved combined summary to {combined_dir / 'combined_summary.json'}")
    
    # Create combined reliability plot using saved reliability data
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        single_plot_count = 0
        pair_plot_count = 0
        
        for model_name, results in all_results.items():
            if results:
                model_short = model_name.split('/')[-1]
                color = get_model_color(model_name)
                
                # Find the last available round
                try:
                    model_dir = Path("re_irl_min_stratified_plots") / get_model_short_name(model_name)
                    round_dirs = [d for d in model_dir.glob("round_*") if d.is_dir()]
                    if round_dirs:
                        last_round = sorted(round_dirs, key=lambda x: int(x.name.split('_')[1]))[-1]
                        round_dir = last_round
                        print(f"📊 Using {round_dir} for {model_short}")
                    else:
                        round_dir = model_dir / "round_1"
                        print(f"📊 Fallback to round_1 for {model_short}")
                    
                    # Load saved reliability curve data
                    single_data_file = round_dir / "single_text_reliability_data.npy"
                    pair_data_file = round_dir / "pairwise_reliability_data.npy"
                    
                    print(f"📄 Checking files for {model_short}:")
                    print(f"   Single: {single_data_file.exists()} - {single_data_file}")
                    print(f"   Pair: {pair_data_file.exists()} - {pair_data_file}")
                    
                    if single_data_file.exists():
                        single_data = np.load(single_data_file, allow_pickle=True).item()
                        print(f"   ✅ Loaded single data: {len(single_data['xs'])} points")
                        ax1.plot(single_data['xs'], single_data['ys'], 'o-', color=color, 
                                label=model_short, alpha=0.8, markersize=4)
                        single_plot_count += 1
                    else:
                        print(f"   ❌ Single reliability data not found for {model_short}")
                    
                    if pair_data_file.exists():
                        pair_data = np.load(pair_data_file, allow_pickle=True).item()
                        print(f"   ✅ Loaded pair data: {len(pair_data['xs'])} points")
                        ax2.plot(pair_data['xs'], pair_data['ys'], 'o-', color=color, 
                                label=model_short, alpha=0.8, markersize=4)
                        pair_plot_count += 1
                    else:
                        print(f"   ❌ Pairwise reliability data not found for {model_short}")
                        
                except Exception as e:
                    print(f"⚠️  Could not load reliability data for {model_short}: {e}")
                    continue
        
        print(f"📈 Combined reliability plot: {single_plot_count} single curves, {pair_plot_count} pair curves")
        
        # Configure single-text plot (same styling as individual plots)
        ax1.plot([0, 1], [0, 1], '--', alpha=0.6, color='gray')
        ax1.set_xlabel('Predicted probability')
        ax1.set_ylabel('Observed frequency')
        ax1.set_title('Single-Text Reliability Comparison')
        ax1.grid(alpha=0.3)
        
        # Configure pairwise plot (same styling as individual plots)
        ax2.plot([0, 1], [0, 1], '--', alpha=0.6, color='gray')
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Observed frequency')
        ax2.set_title('Pairwise Reliability Comparison')
        ax2.grid(alpha=0.3)
        
        # Only add legends if we have data
        if single_plot_count > 0:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax1.text(0.5, 0.5, 'No reliability data available', ha='center', va='center', transform=ax1.transAxes)
        
        if pair_plot_count > 0:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'No reliability data available', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(combined_dir / 'combined_reliability_comparison.png', dpi=220, bbox_inches='tight')
        plt.close()
        
        if single_plot_count > 0 or pair_plot_count > 0:
            print(f"📈 ✅ Saved combined reliability plot with {single_plot_count} single + {pair_plot_count} pair curves")
        else:
            print(f"📈 ⚠️  Saved empty combined reliability plot - no data files found")
            print("💡 Tip: Run individual model evaluations first to generate reliability data files")
            
    except Exception as e:
        print(f"⚠️  Could not create combined reliability plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Create combined violin plot for reward distributions
    try:
        import seaborn as sns
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        all_data = []
        all_labels = []
        all_models = []
        
        for model_name, results in all_results.items():
            if results:
                model_short = model_name.split('/')[-1]
                
                # Find the last available round
                try:
                    model_dir = Path("re_irl_min_stratified_plots") / get_model_short_name(model_name)
                    round_dirs = [d for d in model_dir.glob("round_*") if d.is_dir()]
                    if round_dirs:
                        last_round = sorted(round_dirs, key=lambda x: int(x.name.split('_')[1]))[-1]
                        round_dir = last_round
                    else:
                        round_dir = model_dir / "round_1"
                    
                    # Load reward data
                    R_tox_file = round_dir / "R_tox_TEST.npy"
                    R_nont_file = round_dir / "R_nont_TEST.npy"
                    
                    if R_tox_file.exists() and R_nont_file.exists():
                        R_tox = np.load(R_tox_file)
                        R_nont = np.load(R_nont_file)
                        
                        # Add toxic rewards
                        all_data.extend(R_tox)
                        all_labels.extend(['Toxic'] * len(R_tox))
                        all_models.extend([model_short] * len(R_tox))
                        
                        # Add non-toxic rewards
                        all_data.extend(R_nont)
                        all_labels.extend(['Non-Toxic'] * len(R_nont))
                        all_models.extend([model_short] * len(R_nont))
                        
                except Exception as e:
                    print(f"⚠️  Could not load reward data for {model_short}: {e}")
                    continue
        
        if all_data:
            # Create DataFrame
            df = pd.DataFrame({
                'Reward': all_data, 
                'Category': all_labels, 
                'Model': all_models
            })
            
            # Create violin plot with box overlay
            sns.violinplot(data=df, x='Model', y='Reward', hue='Category', 
                          palette={"Toxic": "#EEC8B0", "Non-Toxic": "#C8DAF0"}, 
                          inner=None, saturation=1, ax=ax)
            
            # Add box plot overlay
            sns.boxplot(data=df, x='Model', y='Reward', hue='Category',
                       width=0.3, showcaps=True,
                       boxprops={'facecolor':'none', 'edgecolor':'k'},
                       showfliers=False, whiskerprops={'linewidth':2}, ax=ax,
                       palette={"Toxic": "#EEC8B0", "Non-Toxic": "#C8DAF0"},
                       saturation=1)
            
            # Calculate and add overall decision boundary
            toxic_data = df[df['Category'] == 'Toxic']['Reward']
            nontoxic_data = df[df['Category'] == 'Non-Toxic']['Reward']
            decision_boundary = 0.5 * (toxic_data.mean() + nontoxic_data.mean())
            ax.axhline(y=decision_boundary, color='red', linestyle='--', alpha=0.6, linewidth=2,
                      label=f'Overall Decision Boundary ({decision_boundary:.2f})')
            
            # Styling
            ax.set_title('Model Comparison: Reward Score Distributions', fontsize=14)
            ax.set_xlabel('Model Architecture', fontsize=12)
            ax.set_ylabel('Reward Score R(o)', fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(alpha=0.3)
            ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(combined_dir / 'combined_reward_distribution_violin.png', dpi=220, bbox_inches='tight')
            plt.close()
            print(f"📈 Saved combined violin plot to {combined_dir / 'combined_reward_distribution_violin.png'}")
        
    except Exception as e:
        print(f"⚠️  Could not create combined violin plot: {e}")
    
    print(f"📊 Combined analysis complete!")
    
    # Create combined analysis plots
    create_combined_analysis_plots(all_results, combined_dir)

def run_single_model(cfg: DictConfig, model_config: Dict, combined_results: Dict):
    """Run evaluation for a single model."""
    model_name = model_config['name']
    detox_name = model_config['detox_name']
    hidden_size = model_config['hidden_size']
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING MODEL: {model_name}")
    print(f"📦 Base Model: {model_name}")
    print(f"🧠 Hidden Size: {hidden_size}")
    print(f"🔄 Detox Model: {detox_name}")
    print(f"{'='*80}")
    
    # Get dataset paths
    n_train = OmegaConf.select(cfg, "dataset.train_samples", None)
    paths = get_dataset_paths(model_name, detox_name,
                          cache_dir=OmegaConf.select(cfg, "dataset.cache_dir", "datasets"),
                          train_samples=n_train)
    print(f"📁 Dataset Paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
        if not os.path.exists(path):
            print(f"  ⚠️  WARNING: {path} not found!")
            return None
    
    # Update config for this model
    cfg.model.base_model_name = model_name
    cfg.model.hidden_size = hidden_size
    cfg.dataset.original_dataset_path = paths['original_dataset_path']
    cfg.dataset.detoxified_dataset_path = paths['detoxified_dataset_path']
    cfg.dataset.sorted_toxic_dataset_path = paths['sorted_toxic_dataset_path']
    cfg.dataset.sorted_non_toxic_dataset_path = paths['sorted_non_toxic_dataset_path']
    
    # Create model-specific output directory
    model_short = get_model_short_name(model_name)
    model_out_dir = Path("re_irl_min_stratified_plots") / model_short
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output Directory: {model_out_dir}")
    
    # Run the main evaluation logic (existing code)
    result = run_evaluation_logic(cfg, model_out_dir)
    
    # Store results for combined analysis
    combined_results[model_name] = result
    
    return result

# -------------------------- main --------------------------

def run_evaluation_logic(cfg: DictConfig, out_dir: Path):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); pyro.set_rng_seed(cfg.seed)

    original_path = cfg.dataset.original_dataset_path
    detox_path    = cfg.dataset.detoxified_dataset_path
    sorted_toxic_path    = cfg.dataset.sorted_toxic_dataset_path
    sorted_nontoxic_path = cfg.dataset.sorted_non_toxic_dataset_path

    # 👇 Add this block
    print("📦 IRL will use the following datasets:")
    print(f"   original:       {original_path}")
    print(f"   detoxified:     {detox_path}")
    print(f"   sorted_toxic:   {sorted_toxic_path}")
    print(f"   sorted_non_toxic:{sorted_nontoxic_path}")


    for p in [original_path, detox_path, sorted_toxic_path, sorted_nontoxic_path]:
        assert os.path.exists(p), f"Missing: {p}"

    # Load aligned pairwise data
    orig  = load_json(original_path)
    detox = load_json(detox_path)
    assert len(orig) == len(detox)
    for i in range(min(10, len(orig))):
        assert orig[i]["prompt"] == detox[i]["prompt"]
    texts_orig  = [d["output"] for d in orig]
    texts_detox = [d["output"] for d in detox]

    # Teacher for pairwise truth
    print("Building teacher rewards...")
    gt = GroundTruthGenerator(classifier_name=cfg.model.reward_classifier, device=device)
    r_star_orig  = gt.compute_ground_truth_rewards(texts_orig)
    r_star_detox = gt.compute_ground_truth_rewards(texts_detox)
    pgt_pairs    = sigmoid_np(r_star_detox - r_star_orig)
    y_toxic_orig = (r_star_orig < 0.0).astype(np.int32)
    print(f"Class ratio ALL (toxic proportion) = {y_toxic_orig.mean():.3f}")

    # Feature extraction
    print(f"Extracting features with {cfg.model.base_model_name} ...")
    feat = SimplifiedIRLRewardComputer(
        artifact_name=None,
        base_model_name=cfg.model.base_model_name,
        likelihood_type="bradley_terry",
        normalization_strategy=cfg.training.normalization_strategy,
        n_posterior_samples=1,
        device=device,
        theta_samples=torch.randn(1, cfg.model.hidden_size, device=device),
    )
    with torch.no_grad():
        phi_orig_raw  = feat.extract_features(texts_orig,  max_length=cfg.training.max_length, batch_size=cfg.training.batch_size)
        phi_detox_raw = feat.extract_features(texts_detox, max_length=cfg.training.max_length, batch_size=cfg.training.batch_size)

    # Pairwise CAL/TEST + fixed VAL_MONITOR
    cal_idx, test_idx, pool_idx = stratified_holdout_by_label(
        y_toxic_orig, seed=cfg.seed, frac_cal=cfg.training.cal_fraction, frac_test=cfg.training.val_fraction
    )
    monitor_frac = OmegaConf.select(cfg, "training.monitor_fraction", default=0.10)
    val_monitor_idx, pool_idx = stratified_take_from_indices(pool_idx, y_toxic_orig, frac=monitor_frac, seed=cfg.seed)

    print(f"Pairwise splits: CAL={len(cal_idx)}, TEST={len(test_idx)}, VAL_MONITOR={len(val_monitor_idx)}, TRAIN-POOL={len(pool_idx)}")
    print("Toxic proportions:",
          f"CAL={y_toxic_orig[cal_idx].mean():.3f}",
          f"TEST={y_toxic_orig[test_idx].mean():.3f}",
          f"VAL_MONITOR={y_toxic_orig[val_monitor_idx].mean():.3f}",
          f"POOL={y_toxic_orig[pool_idx].mean():.3f}")

    # K disjoint training folds from pool
    folds_local  = stratified_kfold_indices_by_label(y_toxic_orig[pool_idx], k=cfg.training.k_folds, seed=cfg.seed)
    folds_global = [pool_idx[f] for f in folds_local]

    def assert_disjoint(*arrays):
        seen = set()
        for a in arrays:
            a_set = set(map(int, a))
            overlap = seen & a_set
            assert not overlap, f"Split leakage of {len(overlap)} indices!"
            seen |= a_set

    # Disjointness checks
    assert_disjoint(cal_idx, test_idx, val_monitor_idx, *folds_global)

    # Freeze normalization across rounds
    trainpool_stack = torch.vstack([phi_orig_raw[pool_idx], phi_detox_raw[pool_idx]])
    global_mean = trainpool_stack.mean(0, keepdim=True)
    global_std  = trainpool_stack.std(0, keepdim=True).clamp_min(1e-6)
    print("Using GLOBAL z-scoring (train-pool only).")

    # NEW — persist TRAIN-POOL normalization + indices so other scripts can reuse them
    np.save(out_dir / "global_mean.npy", global_mean.cpu().numpy())
    np.save(out_dir / "global_std.npy",  global_std.cpu().numpy())
    np.save(out_dir / "train_pool_idx.npy", np.asarray(pool_idx, dtype=np.int64))
    print(f"[SAVED] {out_dir/'global_mean.npy'}, {out_dir/'global_std.npy'}, {out_dir/'train_pool_idx.npy'}")




    # ---------- sorted single-text data (BALANCE: keep only base-model rows, then 1:1 balance) ----------
    sorted_toxic = load_json(sorted_toxic_path)
    sorted_nont  = load_json(sorted_nontoxic_path)
    print(f"[SORTED] loaded toxic={len(sorted_toxic)}  non-toxic={len(sorted_nont)}")

    rng = np.random.default_rng(cfg.seed)
    base_name = cfg.model.base_model_name

    def _filter_by_model(rows, name):
        # First try exact match
        kept = [r for r in rows if r.get("model_name", "") == name]
        if kept:
            return kept
        # If no exact match, try partial match (for compatibility)
        model_short = name.split('/')[-1] if '/' in name else name
        kept = [r for r in rows if model_short in r.get("model_name", "")]
        if kept:
            return kept
        # If still no match, return all (fallback for compatibility)
        print(f"⚠️  No model_name filter match for '{name}', using all rows")
        return rows

    # keep only base model (not the detox model)
    sorted_toxic_bm = _filter_by_model(sorted_toxic, base_name)
    sorted_nont_bm  = _filter_by_model(sorted_nont,  base_name)
    print(f"[SORTED] kept by model_name={base_name}: toxic={len(sorted_toxic_bm)}  non-toxic={len(sorted_nont_bm)}")
    
    # Relaxed assertion - just need some data
    if len(sorted_toxic_bm) == 0 or len(sorted_nont_bm) == 0:
        print(f"⚠️  Warning: Limited sorted data - toxic={len(sorted_toxic_bm)}, non-toxic={len(sorted_nont_bm)}")
        if len(sorted_toxic_bm) == 0:
            sorted_toxic_bm = sorted_toxic[:10] if sorted_toxic else []
        if len(sorted_nont_bm) == 0:
            sorted_nont_bm = sorted_nont[:10] if sorted_nont else []

    # balance to the same count on both sides
    n_bal = min(len(sorted_toxic_bm), len(sorted_nont_bm))
    idx_tox_keep  = rng.choice(len(sorted_toxic_bm),  n_bal, replace=False)
    idx_nont_keep = rng.choice(len(sorted_nont_bm), n_bal, replace=False)

    sorted_toxic_bal = [sorted_toxic_bm[i] for i in idx_tox_keep]
    sorted_nont_bal  = [sorted_nont_bm[i]  for i in idx_nont_keep]

    tox_texts_bal  = [x["output"] for x in sorted_toxic_bal]
    nont_texts_bal = [x["output"] for x in sorted_nont_bal]
    print(f"[SORTED] using balanced sets (by model_name): toxic={len(tox_texts_bal)}  non-toxic={len(nont_texts_bal)}")


    # Extract features ONLY for the balanced sets
    with torch.no_grad():
        phi_tox_raw  = feat.extract_features(tox_texts_bal,  max_length=cfg.training.max_length,
                                             batch_size=cfg.training.batch_size)
        phi_nont_raw = feat.extract_features(nont_texts_bal, max_length=cfg.training.max_length,
                                             batch_size=cfg.training.batch_size)

    # Single-text labels (teacher or membership) ON THE BALANCED SETS
    use_teacher_labels_for_sorted = OmegaConf.select(cfg, "evaluation.use_teacher_labels_for_sorted", default=True)
    if use_teacher_labels_for_sorted:
        r_star_tox_bal  = gt.compute_ground_truth_rewards(tox_texts_bal)
        r_star_nont_bal = gt.compute_ground_truth_rewards(nont_texts_bal)
        # non-toxic = 1, toxic = 0
        y_tox_labels  = (r_star_tox_bal  >= 0.0).astype(int) # when r_star_tox_bal is negative, it is toxic
        y_nont_labels = (r_star_nont_bal >= 0.0).astype(int) # when r_star_nont_bal is negative, it is non-toxic
    else:
        y_tox_labels  = np.zeros(len(tox_texts_bal), dtype=int)
        y_nont_labels = np.ones(len(nont_texts_bal), dtype=int)






    summary = []
    prior_mean, prior_std = None, None

    for round_idx, train_fold_idx in enumerate(folds_global, start=1):
        print("\n" + "=" * 70)
        print(f"RE-IRL Round {round_idx}/{cfg.training.k_folds}  (train size = {len(train_fold_idx)})")
        print("=" * 70)
        print(f"🧩 Using base_model={cfg.model.base_model_name}  hidden_size={cfg.model.hidden_size}  "
                f"original={cfg.dataset.original_dataset_path}  detox={cfg.dataset.detoxified_dataset_path}")


        # Standardize with GLOBAL stats (frozen across rounds)
        mean, std = global_mean, global_std

        phi_o = (phi_orig_raw - mean) / std
        phi_d = (phi_detox_raw - mean) / std
        delta_phi_all = phi_d - phi_o

        demos_train = [(phi_o[i], phi_d[i]) for i in train_fold_idx]
        dp_train    = delta_phi_all[train_fold_idx]; pgt_train = pgt_pairs[train_fold_idx]
        dp_val_fixed  = delta_phi_all[val_monitor_idx]; pgt_val_fixed = pgt_pairs[val_monitor_idx]
        dp_cal   = delta_phi_all[cal_idx];  pgt_cal  = pgt_pairs[cal_idx]
        dp_test  = delta_phi_all[test_idx]; pgt_test = pgt_pairs[test_idx]

        # VI
        d = phi_o.shape[1]
        vi = train_vi_bt(
            demos=demos_train,
            dp_train=dp_train, pgt_train=pgt_train,
            dp_val_fixed=dp_val_fixed, pgt_val_fixed=pgt_val_fixed,
            d=d,
            prior_mean=prior_mean, prior_std=prior_std,
            n_steps=cfg.training.n_steps, lr=cfg.training.learning_rate,
            alpha=cfg.training.alpha, learn_T=cfg.training.use_learnable_temperature,
            eval_every=max(50, cfg.training.n_steps // 50), device=device,
        )
        mu, sig, T_vi = vi["mu"], vi["sig"], vi["T"]
        print(f"[VI DONE] ||mu||={float(mu.norm()):.4f}  T={T_vi:.3f}")

        round_dir = out_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        np.save(round_dir / "mu.npy",  mu.numpy())
        np.save(round_dir / "sig.npy", sig.numpy())
        with open(round_dir / "train_history.json", "w") as f:
            json.dump(vi["history"], f, indent=2)

        # Posterior samples
        S = OmegaConf.select(cfg, "training.n_samples", default=256)

        theta_samples = dist.Normal(mu.to(device), sig.to(device)).sample((S,))

        # ---------- Pairwise calibration ----------
        p_cal_raw = expected_pairwise_probs(dp_cal, theta_samples, T_vi, alpha=cfg.training.alpha)
        p_cal_raw = np.clip(p_cal_raw, 1e-7, 1 - 1e-7)
        logits_cal = np.log(p_cal_raw / (1 - p_cal_raw))
        y_cal = np.clip(pgt_cal, 1e-7, 1 - 1e-7)

        a_pl, b_pl = platt_calibrate(logits_cal, y_cal)
        T_cal = temp_calibrate(logits_cal, y_cal)

        p_test_raw   = expected_pairwise_probs(dp_test, theta_samples, T_vi, alpha=cfg.training.alpha)
        p_test_raw   = np.clip(p_test_raw, 1e-7, 1 - 1e-7)
        logits_test  = np.log(p_test_raw / (1 - p_test_raw))
        p_test_platt = 1 / (1 + np.exp(-np.clip(a_pl * logits_test + b_pl, -40, 40)))
        p_test_tcal  = 1 / (1 + np.exp(-np.clip(logits_test / T_cal, -40, 40)))

        acc_raw   = float(((p_test_raw   >= 0.5) == (pgt_test >= 0.5)).mean())
        acc_platt = float(((p_test_platt >= 0.5) == (pgt_test >= 0.5)).mean())
        acc_tcal  = float(((p_test_tcal  >= 0.5) == (pgt_test >= 0.5)).mean())
        print(f"[PAIRWISE TEST] acc_raw={acc_raw:.3f}  acc_platt={acc_platt:.3f}  acc_temp={acc_tcal:.3f}")

        # Pairwise calibration metrics on TEST (after calibration)
        pair_brier_platt_test = brier_score(p_test_platt, pgt_test)
        pair_ece_platt_q_test = ece_quantile(p_test_platt, pgt_test, n_bins=15)
        print(f"[PAIRWISE TEST] Brier(Platt)={pair_brier_platt_test:.4f}  ECE(q15, Platt)={pair_ece_platt_q_test:.4f}")


        plot_reliability_curve_quantile(
            p_test_platt, pgt_test,
            title="Pairwise reliability (TEST)",
            path=round_dir / 'pairwise_reliability.png',
            n_bins=10,
            color=get_model_color(cfg.model.base_model_name)
        )
        plt.figure(figsize=(6.2, 3.6))
        common_edges = np.linspace(0, 1, 26)
        plt.hist(pgt_test,     bins=common_edges, density=True, alpha=0.5, label='p_gt (teacher)')
        plt.hist(p_test_raw,   bins=common_edges, density=True, alpha=0.5, label='p_hat raw')
        plt.hist(p_test_platt, bins=common_edges, density=True, alpha=0.7, label='p_hat Platt')
        plt.legend(); plt.xlabel('Probability'); plt.ylabel('Density')
        plt.title('Pairwise prob distributions (TEST)')
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(round_dir / 'pairwise_prob_hists.png', dpi=220); plt.close()

        # ---------- Single-text ----------
        phi_tox  = (phi_tox_raw  - mean) / std
        phi_nont = (phi_nont_raw - mean) / std
        R_tox  = expected_scores(phi_tox,  theta_samples)
        R_nont = expected_scores(phi_nont, theta_samples)

        # --- Same-class & safe-band diagnostics (runs on the full balanced sorted sets) ---

        # Teacher probabilities for the balanced sets (if you computed teacher rewards)
        if use_teacher_labels_for_sorted:
            p_tox_bal  = sigmoid_np(r_star_tox_bal)   # shape: len(tox_texts_bal)
            p_nont_bal = sigmoid_np(r_star_nont_bal)  # shape: len(nont_texts_bal)
        else:
            # Fallback: use membership as a crude 0/1 signal if no teacher probs exist
            p_tox_bal  = np.zeros(len(R_tox), dtype=float)
            p_nont_bal = np.ones(len(R_nont), dtype=float)

        # (1) Non-toxic vs non-toxic: do we over-rank inside the safe region?
        acc_nn, gap_nn = pairwise_acc(R_nont, num_pairs=5000)
        rho_nn = float(spearmanr(R_nont, p_nont_bal).correlation) if spearmanr is not None else float("nan")
        print(f"[SAFE-INTRA] non-toxic vs non-toxic: pairwiseAcc={acc_nn:.3f}  mean|ΔR|={gap_nn:.3f}  Spearman(R, p_teacher)={rho_nn:.3f}")

        # (2) Toxic vs toxic symmetry (optional but useful)
        acc_tt, gap_tt = pairwise_acc(R_tox, num_pairs=5000)
        rho_tt = float(spearmanr(R_tox, p_tox_bal).correlation) if spearmanr is not None else float("nan")
        print(f"[TOX-INTRA] toxic vs toxic:       pairwiseAcc={acc_tt:.3f}  mean|ΔR|={gap_tt:.3f}  Spearman(R, p_teacher)={rho_tt:.3f}")

        # (3) “Safe band” inside non-toxic (e.g., teacher prob ≥ τ)
        tau = 0.90
        mask_safe = (p_nont_bal >= tau)
        if mask_safe.sum() >= 5:
            R_safe = R_nont[mask_safe]
            p_safe = p_nont_bal[mask_safe]
            acc_safe, gap_safe = pairwise_acc(R_safe, num_pairs=5000)
            rho_safe = float(spearmanr(R_safe, p_safe).correlation) if spearmanr is not None else float("nan")
            print(f"[SAFE-BAND τ={tau:.2f}] size={mask_safe.sum()}  pairwiseAcc={acc_safe:.3f}  mean|ΔR|={gap_safe:.3f}  Spearman(R, p_teacher)={rho_safe:.3f}")
        else:
            acc_safe = gap_safe = rho_safe = float("nan")
            print(f"[SAFE-BAND τ={tau:.2f}] not enough safe examples")





        # --- NEW: Pairwise accuracy on the entire balanced sorted set ---
        # Probability a random non-toxic scores higher than a random toxic (AUC == pairwise accuracy)
        if roc_auc_score is not None:
            y_balanced = np.concatenate([np.zeros(len(R_tox), dtype=int),
                                        np.ones(len(R_nont), dtype=int)])
            scores_balanced = np.concatenate([R_tox, R_nont])
            pairwise_acc_bal = float(roc_auc_score(y_balanced, scores_balanced))
        else:
            pairwise_acc_bal = float('nan')
        print(f"[SORTED PAIRWISE] accuracy (balanced ALL pairs) = {pairwise_acc_bal:.3f}")


        # CAL/TEST splits within sorted sets (per class)
        rng = np.random.default_rng(cfg.seed)
        idx_tox  = np.arange(len(R_tox));  rng.shuffle(idx_tox)
        idx_nont = np.arange(len(R_nont)); rng.shuffle(idx_nont)
        frac_cal_single = 0.30
        n_cal_tox  = int(np.floor(frac_cal_single * len(idx_tox)))
        n_cal_nont = int(np.floor(frac_cal_single * len(idx_nont)))
        cal_tox,  test_tox  = idx_tox[:n_cal_tox],   idx_tox[n_cal_tox:]
        cal_nont, test_nont = idx_nont[:n_cal_nont], idx_nont[n_cal_nont:]

        # Labels (teacher by default)
        y_single_cal  = np.concatenate([y_tox_labels[cal_tox],  y_nont_labels[cal_nont]])
        y_single_test = np.concatenate([y_tox_labels[test_tox], y_nont_labels[test_nont]])

        # --- (1) Platt calibration on single-text for probabilities ---
        logits_single_cal  = np.concatenate([R_tox[cal_tox],  R_nont[cal_nont]])
        logits_single_test = np.concatenate([R_tox[test_tox], R_nont[test_nont]])
        a_s, b_s = platt_calibrate(logits_single_cal, y_single_cal)
        z_test = np.clip(a_s * logits_single_test + b_s, -40, 40)
        p_single = 1 / (1 + np.exp(-z_test))

        # Reliability (TEST)
        plot_reliability_curve_quantile(
            p_single, y_single_test,
            title="Single-text reliability (TEST)",
            path=round_dir / 'single_text_reliability.png',
            n_bins=10,
            color=get_model_color(cfg.model.base_model_name)
        )
        
        # Combined reliability curves
        plot_combined_reliability_curves(
            p_single, y_single_test,
            p_test_platt, pgt_test,
            model_name=cfg.model.base_model_name,
            path=round_dir / 'combined_reliability_curves.png',
            n_bins=10
        )

        # Single-text calibration metrics on TEST (after Platt)
        single_brier_platt_test = brier_score(p_single, y_single_test)
        single_ece_platt_q_test = ece_quantile(p_single, y_single_test, n_bins=15)
        print(f"[SINGLE TEST]  Brier(Platt)={single_brier_platt_test:.4f}  ECE(q15, Platt)={single_ece_platt_q_test:.4f}")


        # --- (2) Midpoint threshold for classification ---
        R_tox_cal_mean  = float(np.mean(R_tox[cal_tox]))   if len(cal_tox)  else 0.0
        R_nont_cal_mean = float(np.mean(R_nont[cal_nont])) if len(cal_nont) else 0.0
        R_star_mid = 0.5 * (R_tox_cal_mean + R_nont_cal_mean)

        # Evaluate classification on TEST using midpoint
        R_test   = np.concatenate([R_tox[test_tox], R_nont[test_nont]])
        y_pred_mid = (R_test >= R_star_mid).astype(int)  # >= => non-toxic
        acc_mid = float((y_pred_mid == y_single_test).mean())

        # Evaluate AUROC on raw scores
        auroc = float(roc_auc_score(y_single_test, R_test)) if roc_auc_score else float('nan')

        # Separability: TEST-only mean gap
        separation = float(R_nont[test_nont].mean() - R_tox[test_tox].mean())

        # Scale-free separation (Cohen's d) on TEST
        pooled_sd = np.sqrt(0.5*(R_nont[test_nont].var() + R_tox[test_tox].var()))
        sep_std = float((R_nont[test_nont].mean() - R_tox[test_tox].mean()) / (pooled_sd + 1e-12))

        print(f"[SINGLE-TEXT] separation(TEST)={separation:.3f}  acc_midpoint(TEST)={acc_mid:.3f}  AUROC(TEST)={auroc:.3f}")
        print(f"[SINGLE-TEXT] sep_std(TEST)={sep_std:.3f}  pooled_sd={pooled_sd:.3f}")

        # Hist + thresholds
        plt.figure(figsize=(6.7, 3.9))
        model_short = cfg.model.base_model_name.split('/')[-1] if '/' in cfg.model.base_model_name else cfg.model.base_model_name
        plt.hist(R_tox,  bins=40, alpha=0.6, density=True, label='toxic')
        plt.hist(R_nont, bins=40, alpha=0.6, density=True, label='non-toxic')
        # draw midpoint threshold only
        plt.axvline(R_star_mid, color='k', lw=2, label=f'Decision Boundary={R_star_mid:.2f}')
        # class means with dotted lines and legend
        plt.axvline(R_tox.mean(),  ls=':', alpha=0.7, color='blue', label='Toxic Mean')
        plt.axvline(R_nont.mean(), ls=':', alpha=0.7, color='orange', label='Non-Toxic Mean')
        plt.legend(); plt.xlabel('Reward score R(o)'); plt.ylabel('Density (Normalized Frequency)')
        plt.title(f'Single-text Reward Distribution for {model_short}')
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(round_dir / 'single_text_rewards.png', dpi=220); plt.close()
        
        # Create violin plot for reward distributions
        plot_reward_distribution_violin(R_tox, R_nont, cfg.model.base_model_name, round_dir / 'reward_distribution_violin.png')
        
        # Save reward data for combined plots
        np.save(round_dir / 'R_tox_TEST.npy', R_tox)
        np.save(round_dir / 'R_nont_TEST.npy', R_nont)

        # Confusion matrix (midpoint)
        if confusion_matrix is not None:
            cm_mid = confusion_matrix(y_single_test, y_pred_mid, labels=[0,1])
            print("Single-text CM (TEST, midpoint rule) rows=true [toxic,non-toxic], cols=pred:\n", cm_mid)
            
            # Calculate F1 scores
            tn, fp, fn, tp = cm_mid.ravel()
            precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # toxic precision
            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # toxic recall
            f1_toxic = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
            
            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # non-toxic precision  
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # non-toxic recall
            f1_nontoxic = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
            
            f1_macro = (f1_toxic + f1_nontoxic) / 2.0
            print(f"[SINGLE-TEXT] F1-toxic={f1_toxic:.3f}  F1-non-toxic={f1_nontoxic:.3f}  F1-macro={f1_macro:.3f}")
            
            plot_confmat(cm_mid, "Single-text confusion matrix (TEST, midpoint)", round_dir / 'single_text_confmat_midpoint.png')

        # (Optional) also report Platt 0.5 decision CM for comparison
        y_pred_platt = (p_single >= 0.5).astype(int)
        if confusion_matrix is not None:
            cm_pl = confusion_matrix(y_single_test, y_pred_platt, labels=[0,1])
            plot_confmat(cm_pl, "Single-text confusion matrix (TEST, platt p>=0.5)", round_dir / 'single_text_confmat_platt.png')





        # ---------- Posterior uncertainty (round-level) ----------
        sig_np = sig.numpy()
        d = sig_np.size
        logdet_Sigma = float(np.sum(np.log(np.clip(sig_np, 1e-12, None))) * 2.0)  # log det Σ
        trace_Sigma  = float(np.sum(sig_np**2))
        H_posterior  = 0.5 * ( d * (1.0 + np.log(2.0*np.pi)) + logdet_Sigma )
        H_half_logdet = 0.5 * logdet_Sigma  # just the logdet part (no constant)

        print(f"[POSTERIOR] trace(Σ)={trace_Sigma:.2f}  logdetΣ={logdet_Sigma:.2f}  H(θ)={H_posterior:.2f}  H_half_logdet={H_half_logdet:.2f}")

        # ---------- Predictive variance: single-text (closed form in reward space) ----------
        # Use TEST part of your balanced sorted set
        phi_single_test = torch.vstack([phi_tox[test_tox], phi_nont[test_nont]])  # (N_test, d)
        phi2 = (phi_single_test.cpu().numpy()**2)
        var_R_single = phi2 @ (sig_np**2)                      # Var[R(x)] per item
        # summarize
        print(f"[PRED SINGLE-R] Var[R] median={np.median(var_R_single):.3f} IQR=({np.percentile(var_R_single,25):.3f},{np.percentile(var_R_single,75):.3f})")

        # ---------- Numerically-stable Bernoulli entropy ----------
        # Uses scipy.special.xlogy if available (x*log(y) with 0*log(0)=0 by definition).
        try:
            from scipy.special import xlogy
            def bernoulli_entropy(p):
                p = np.asarray(p, dtype=np.float64)
                return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))
        except Exception:
            def bernoulli_entropy(p):
                p = np.asarray(p, dtype=np.float64)
                # compute with masked logs; turn non-finite entries into 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1 = -p * np.log(p)               # 0*log(0) -> NaN here, fixed below
                    t2 = -(1.0 - p) * np.log1p(-p)    # stable near p~0
                t1[~np.isfinite(t1)] = 0.0
                t2[~np.isfinite(t2)] = 0.0
                return t1 + t2

        S_unc = 128  # can reuse your theta_samples too
        theta_unc = dist.Normal(mu.to(device), sig.to(device)).sample((S_unc,))

        # (a) SINGLE-TEXT prob after Platt
        R_samps = (theta_unc @ phi_single_test.to(device).T).cpu().numpy()           # (S, N)
        z_samps = a_s * R_samps + b_s
        p_samps_single = 1.0 / (1.0 + np.exp(-np.clip(z_samps, -40, 40)))            # (S, N)

        pbar_single = p_samps_single.mean(axis=0)
        var_p_single = p_samps_single.var(axis=0)
        H_total_single = bernoulli_entropy(pbar_single)
        H_cond_single  = bernoulli_entropy(p_samps_single).mean(axis=0)              # aleatoric
        MI_single      = H_total_single - H_cond_single                               # epistemic

        print(f"[PRED SINGLE-P] Var[p] median={np.median(var_p_single):.4f}  H_total median={np.median(H_total_single):.4f} H_cond_single median (aleatoric)={np.median(H_cond_single):.4f}  MI median (epistemic)={np.median(MI_single):.4f}")

        # (b) PAIRWISE prob (Bradley–Terry)
        dp_test_np = dp_test.cpu().numpy()
        # closed-form margin variance:
        var_m_pair = (dp_test_np**2) @ (sig_np**2) * (cfg.training.alpha**2) / (T_vi**2)
        # MC prob variance + entropies:
        m_samps = (cfg.training.alpha * (theta_unc @ dp_test.to(device).T) / T_vi).cpu().numpy()            # (S, M)
        p_samps_pair = 1.0 / (1.0 + np.exp(-np.clip(m_samps, -40, 40)))
        pbar_pair = p_samps_pair.mean(axis=0)
        var_p_pair = p_samps_pair.var(axis=0)
        H_total_pair = bernoulli_entropy(pbar_pair)
        H_cond_pair  = bernoulli_entropy(p_samps_pair).mean(axis=0)
        MI_pair      = H_total_pair - H_cond_pair

        # Robust, element-wise cleanup (keep vectors, replace NaNs/±inf per-entry)
        H_total_pair = np.nan_to_num(H_total_pair, nan=0.0, posinf=0.0, neginf=0.0)
        H_cond_pair  = np.nan_to_num(H_cond_pair,  nan=0.0, posinf=0.0, neginf=0.0)
        MI_pair      = np.nan_to_num(MI_pair,      nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[PRED PAIR-P] Var[p] median={np.median(var_p_pair):.4f}  "
              f"H_total median={np.median(H_total_pair):.4f}  MI median={np.median(MI_pair):.4f}")
        print(f"[PRED PAIR-M] Var[m] median={np.median(var_m_pair):.3f} (closed-form)")


        # --- SAVE UNCERTAINTY VECTORS + INDEX MAPS + (OPTIONAL) TOP-K TEXTS ---
        if OmegaConf.select(cfg, "evaluation.save_uncertainty_details", default=True):
            # 1) Per-item predictive uncertainty (TEST) — for violins/hists later
            np.save(round_dir / 'single_H_total_TEST.npy', H_total_single.astype(np.float32))
            np.save(round_dir / 'single_H_cond_TEST.npy',  H_cond_single.astype(np.float32))
            np.save(round_dir / 'single_MI_TEST.npy',      MI_single.astype(np.float32))

            np.save(round_dir / 'pair_H_total_TEST.npy', H_total_pair.astype(np.float32))
            np.save(round_dir / 'pair_H_cond_TEST.npy',  H_cond_pair.astype(np.float32))
            np.save(round_dir / 'pair_MI_TEST.npy',      MI_pair.astype(np.float32))

            # 2) Index maps so you can recover which items these vectors refer to
            #    NOTE: order for single-text arrays follows phi_single_test construction: [tox TEST, then nont TEST].
            with open(round_dir / 'single_TEST_index_map.json', 'w') as f:
                json.dump({
                    "order": "tox_then_nont",
                    "test_tox":  [int(i) for i in test_tox.tolist()],
                    "test_nont": [int(i) for i in test_nont.tolist()]
                }, f)

            with open(round_dir / 'pair_TEST_index_map.json', 'w') as f:
                json.dump({"test_idx": [int(i) for i in test_idx.tolist()]}, f)

            # 3) (Optional) Dump top-k examples with texts for fast inspection
            if OmegaConf.select(cfg, "evaluation.dump_topk_uncertainty_texts", default=False):
                K = int(OmegaConf.select(cfg, "evaluation.topk", default=20))
                K = max(0, min(K, len(MI_single)))  # guard

                # Helper to recover single-text item at combined TEST position
                Ntox = len(test_tox)
                def _single_text_at(pos: int):
                    if pos < Ntox:
                        i_local = int(test_tox[pos])
                        return {"label": "toxic", "i_local": i_local, "text": tox_texts_bal[i_local]}
                    j = pos - Ntox
                    i_local = int(test_nont[j])
                    return {"label": "non_toxic", "i_local": i_local, "text": nont_texts_bal[i_local]}

                # Single-text: top-k epistemic (MI) and aleatoric (H_cond)
                ord_ep_single = np.argsort(MI_single)[::-1][:K]
                ord_al_single = np.argsort(H_cond_single)[::-1][:K]

                topk_single_ep = [{
                    "rank": int(r + 1),
                    "MI": float(MI_single[i]),
                    "H_cond": float(H_cond_single[i]),
                    **_single_text_at(int(i))
                } for r, i in enumerate(ord_ep_single)]

                topk_single_al = [{
                    "rank": int(r + 1),
                    "H_cond": float(H_cond_single[i]),
                    "MI": float(MI_single[i]),
                    **_single_text_at(int(i))
                } for r, i in enumerate(ord_al_single)]

                with open(round_dir / 'topk_single_epistemic.json', 'w') as f:
                    json.dump(topk_single_ep, f, ensure_ascii=False, indent=2)
                with open(round_dir / 'topk_single_aleatoric.json', 'w') as f:
                    json.dump(topk_single_al, f, ensure_ascii=False, indent=2)

                # Pairwise: map back via test_idx into orig/detox lists
                Kp = max(0, min(int(OmegaConf.select(cfg, "evaluation.topk", default=20)), len(MI_pair)))
                ord_ep_pair = np.argsort(MI_pair)[::-1][:Kp]
                ord_al_pair = np.argsort(H_cond_pair)[::-1][:Kp]

                def _pair_record(local_pos: int):
                    gidx = int(test_idx[int(local_pos)])  # global example index
                    rec = {
                        "pair_index": gidx,
                        "MI": float(MI_pair[local_pos]),
                        "H_cond": float(H_cond_pair[local_pos]),
                        "prompt": orig[gidx].get("prompt", None),
                        "orig": texts_orig[gidx],
                        "detox": texts_detox[gidx],
                    }
                    return rec

                topk_pair_ep = [{"rank": int(r + 1), **_pair_record(int(i))}
                                for r, i in enumerate(ord_ep_pair)]
                topk_pair_al = [{"rank": int(r + 1), **_pair_record(int(i))}
                                for r, i in enumerate(ord_al_pair)]

                with open(round_dir / 'topk_pair_epistemic.json', 'w') as f:
                    json.dump(topk_pair_ep, f, ensure_ascii=False, indent=2)
                with open(round_dir / 'topk_pair_aleatoric.json', 'w') as f:
                    json.dump(topk_pair_al, f, ensure_ascii=False, indent=2)



        # (c) Sorted single-text -> pairwise (non-toxic vs toxic) predictive uncertainty
        #     Build pair margins on the sorted TEST splits and compute H_total/H_cond/MI.
        try:
            phi_tox_test  = phi_tox[test_tox]   # (N_tox, d)
            phi_nont_test = phi_nont[test_nont] # (N_nont, d)
            N_tox, N_nont = phi_tox_test.shape[0], phi_nont_test.shape[0]

            sorted_pairs_max = int(OmegaConf.select(cfg, "evaluation.sorted_pairwise_max_pairs", default=10000))
            rng = np.random.default_rng(cfg.seed)

            if N_tox == 0 or N_nont == 0:
                H_total_sorted_pair = np.array([], dtype=np.float32)
                H_cond_sorted_pair  = np.array([], dtype=np.float32)
                MI_sorted_pair      = np.array([], dtype=np.float32)
                I = J = np.array([], dtype=int)
            else:
                # choose all cross pairs if small; else sample up to sorted_pairs_max
                total_pairs = N_tox * N_nont
                if total_pairs <= sorted_pairs_max:
                    I, J = np.meshgrid(np.arange(N_tox), np.arange(N_nont), indexing='ij')
                    I, J = I.ravel(), J.ravel()  # lengths = total_pairs
                else:
                    M = sorted_pairs_max
                    I = rng.integers(0, N_tox,  size=M)
                    J = rng.integers(0, N_nont, size=M)

                # delta features: non-toxic minus toxic (prob non-toxic "wins")
                dp_sorted = (phi_nont_test[J] - phi_tox_test[I]).to(device)  # (M, d)

                # MC over the posterior (reuse theta_unc)
                m_samps_sorted = (cfg.training.alpha * (theta_unc @ dp_sorted.T) / T_vi).cpu().numpy()  # (S, M)
                p_samps_sorted = 1.0 / (1.0 + np.exp(-np.clip(m_samps_sorted, -40, 40)))

                pbar_sorted = p_samps_sorted.mean(axis=0)
                var_p_sorted = p_samps_sorted.var(axis=0)  # optional, not saved

                H_total_sorted_pair = bernoulli_entropy(pbar_sorted)
                H_cond_sorted_pair  = bernoulli_entropy(p_samps_sorted).mean(axis=0)  # aleatoric
                MI_sorted_pair      = H_total_sorted_pair - H_cond_sorted_pair        # epistemic

                # Clean up numerically
                H_total_sorted_pair = np.nan_to_num(H_total_sorted_pair, nan=0.0, posinf=0.0, neginf=0.0)
                H_cond_sorted_pair  = np.nan_to_num(H_cond_sorted_pair,  nan=0.0, posinf=0.0, neginf=0.0)
                MI_sorted_pair      = np.nan_to_num(MI_sorted_pair,      nan=0.0, posinf=0.0, neginf=0.0)

            # Save vectors + pair index maps (if enabled)
            if OmegaConf.select(cfg, "evaluation.save_uncertainty_details", default=True):
                np.save(round_dir / 'sorted_pair_H_total_TEST.npy', H_total_sorted_pair.astype(np.float32))
                np.save(round_dir / 'sorted_pair_H_cond_TEST.npy',  H_cond_sorted_pair.astype(np.float32))
                np.save(round_dir / 'sorted_pair_MI_TEST.npy',      MI_sorted_pair.astype(np.float32))

                # index map: local indices into tox/non-tox TEST arrays AND back to the balanced sets
                with open(round_dir / 'sorted_pair_TEST_index_map.json', 'w') as f:
                    json.dump({
                        "tox_test_local":  [int(i) for i in I.tolist()],
                        "nont_test_local": [int(j) for j in J.tolist()],
                        "tox_source_idx":  [int(test_tox[int(i)]) for i in I.tolist()],
                        "nont_source_idx": [int(test_nont[int(j)]) for j in J.tolist()]
                    }, f)

            # (optional) quick print
            if H_total_sorted_pair.size:
                print(f"[PRED SORTED-PAIR] H_total median={np.median(H_total_sorted_pair):.4f}  "
                      f"MI median={np.median(MI_sorted_pair):.4f}  "
                      f"pairs={H_total_sorted_pair.size}")
            else:
                print("[PRED SORTED-PAIR] no pairs available")

        except Exception as e:
            print(f"⚠️  Sorted single-text pairwise uncertainty failed: {e}")
            H_total_sorted_pair = MI_sorted_pair = np.array([], dtype=np.float32)



        # Summary
        summary.append({
            "round": round_idx,
            "train_size": int(len(train_fold_idx)),
            "val_monitor_size": int(len(val_monitor_idx)),
            "pair_cal_size": int(len(cal_idx)),
            "pair_test_size": int(len(test_idx)),
            "T_vi": T_vi,
            "pair_acc_raw": acc_raw,
            "pair_acc_platt": acc_platt,
            "pair_acc_temp": acc_tcal,
            "single_sep_TEST": separation,
            "single_sep_std_TEST": sep_std,
            "single_acc_midpoint_TEST": acc_mid,
            "single_auroc_TEST": auroc,
            "single_f1_toxic_TEST": f1_toxic,
            "single_f1_nontoxic_TEST": f1_nontoxic,
            "single_f1_macro_TEST": f1_macro,
            "platt_pairwise": {"a": a_pl, "b": b_pl},
            "temp_pairwise": {"T_cal": T_cal},
            "platt_single": {"a": a_s, "b": b_s},
            "R_star_mid": R_star_mid,
            "R_star_platt": -b_s / a_s if a_s != 0 else 0.0,
            "sorted_sizes": {"toxic": len(sorted_toxic), "non_toxic": len(sorted_nont)},
            # Posterior uncertainty metrics
            "posterior_trace_Sigma": trace_Sigma,
            "posterior_logdet_Sigma": logdet_Sigma,
            "posterior_entropy": H_posterior,
            "posterior_half_logdet": H_half_logdet,
            # Predictive uncertainty metrics
            "pred_single_R_var_median": float(np.median(var_R_single)),
            "pred_single_R_var_iqr": [float(np.percentile(var_R_single,25)), float(np.percentile(var_R_single,75))],
            "pred_single_p_var_median": float(np.median(var_p_single)),
            "pred_single_p_entropy_median": float(np.median(H_total_single)),
            "pred_single_p_MI_median": float(np.median(MI_single)),
            "pred_pair_p_var_median": float(np.median(var_p_pair)),
            "pred_pair_p_entropy_median": float(np.median(H_total_pair)),
            "pred_pair_p_MI_median": float(np.median(MI_pair)),
            "pred_pair_m_var_median": float(np.median(var_m_pair)),
            "pair_brier_platt_test": pair_brier_platt_test,
            "pair_ece_platt_q_test": pair_ece_platt_q_test,
            "single_brier_platt_test": single_brier_platt_test,
            "single_ece_platt_q_test": single_ece_platt_q_test,
            "sorted_pairwise_acc_balanced": pairwise_acc_bal,      # NEW: all balanced sorted items
            "sorted_pairwise_acc_TEST": auroc,                     # (AUROC on TEST equals pairwise acc on TEST)
            "sorted_pair_p_entropy_median": float(np.median(H_total_sorted_pair)) if H_total_sorted_pair.size else float('nan'),
            "sorted_pair_p_MI_median": float(np.median(MI_sorted_pair)) if MI_sorted_pair.size else float('nan'),

        })

        # sequential Bayes
        prior_mean, prior_std = mu.to(device), sig.to(device)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Done. Outputs in: {out_dir.resolve()}")
    
    # Create individual analysis plots if multiple rounds
    create_individual_plots(summary, out_dir)
    
    return summary

@hydra.main(version_base=None, config_path="../../configs", config_name="re_irl_config")
def main(cfg: DictConfig):
    """Main function that handles both single model and run_all modes."""
    print(f"🖥️ Device: {cfg.device if torch.cuda.is_available() else 'cpu'}\n🌱 Seed: {cfg.seed}")
    
    # Check if run_all mode is enabled
    if cfg.model.get('run_all', False):
        print(f"\n🚀 RUN_ALL MODE ENABLED - Processing {len(cfg.model.models)} models")
        combined_results = {}
        
        # Process each model
        for i, model_config in enumerate(cfg.model.models, 1):
            print(f"\n📊 Processing model {i}/{len(cfg.model.models)}")
            try:
                result = run_single_model(cfg, model_config, combined_results)
                if result is None:
                    print(f"❌ Skipping {model_config['name']} due to missing datasets")
            except Exception as e:
                print(f"❌ Error processing {model_config['name']}: {e}")
                continue
        
        # Create combined results
        if combined_results:
            combined_dir = Path("re_irl_min_stratified_plots") / "combined_results"
            create_combined_plots_and_summary(combined_results, combined_dir)
            print(f"\n🎉 All models processed! Combined results in: {combined_dir.resolve()}")
        else:
            print(f"\n⚠️  No successful model runs to combine")
    else:
        print(f"\n🚀 SINGLE MODEL MODE")
        print(f"📦 Requested Base Model: {cfg.model.base_model_name}")
        
        # NEW: if caller already provided existing dataset files, keep them
        pre_paths = [
            cfg.dataset.original_dataset_path,
            cfg.dataset.detoxified_dataset_path,
            cfg.dataset.sorted_toxic_dataset_path,
            cfg.dataset.sorted_non_toxic_dataset_path,
        ]
        if all(os.path.exists(p) for p in pre_paths):
            print("🔒 Using dataset paths provided by caller; skipping auto-selection.")
        else:

            # Try to find this model in cfg.model.models to auto-select detox + hidden_size
            chosen = None
            for m in cfg.model.models:
                if m.get("name") == cfg.model.base_model_name:
                    chosen = m; break

            if chosen is not None:
                #  pass train_samples from cfg if available
                n_train = OmegaConf.select(cfg, "dataset.train_samples", None)
                paths = get_dataset_paths(chosen["name"], chosen["detox_name"],
                                          cache_dir=OmegaConf.select(cfg, "dataset.cache_dir", "datasets"),
                                          train_samples=n_train)
                cfg.dataset.original_dataset_path     = paths['original_dataset_path']
                cfg.dataset.detoxified_dataset_path   = paths['detoxified_dataset_path']
                cfg.dataset.sorted_toxic_dataset_path = paths['sorted_toxic_dataset_path']
                cfg.dataset.sorted_non_toxic_dataset_path = paths['sorted_non_toxic_dataset_path']
                cfg.model.hidden_size = chosen["hidden_size"]
                print("🧭 Auto-selected datasets & hidden size from config.models:")
                for k, v in paths.items(): print(f"  - {k}: {v}")
                print(f"  - hidden_size: {cfg.model.hidden_size}")
            else:
                # Fallback: if user supplied custom paths, keep them; otherwise build from base_model_name
                print("ℹ️  Model not found in config.models; using provided/derived paths if possible.")
                if not (os.path.exists(cfg.dataset.original_dataset_path) and
                        os.path.exists(cfg.dataset.detoxified_dataset_path) and
                        os.path.exists(cfg.dataset.sorted_toxic_dataset_path) and
                        os.path.exists(cfg.dataset.sorted_non_toxic_dataset_path)):
                    # Try to guess detox name from list if names share prefix
                    guess = next((m for m in cfg.model.models if m["name"].split("/")[-1] in cfg.model.base_model_name or
                                cfg.model.base_model_name.split("/")[-1] in m["name"]), None)
                    detox_guess = guess["detox_name"] if guess else cfg.model.base_model_name
                    paths = get_dataset_paths(cfg.model.base_model_name, detox_guess)
                    cfg.dataset.original_dataset_path     = paths['original_dataset_path']
                    cfg.dataset.detoxified_dataset_path   = paths['detoxified_dataset_path']
                    cfg.dataset.sorted_toxic_dataset_path = paths['sorted_toxic_dataset_path']
                    cfg.dataset.sorted_non_toxic_dataset_path = paths['sorted_non_toxic_dataset_path']
                    print("🧭 Derived dataset paths:")
                    for k, v in paths.items(): print(f"  - {k}: {v}")

        # Per-model output dir (single model): use model-specific folder
        model_short = get_model_short_name(cfg.model.base_model_name)
        out_dir = Path("re_irl_min_stratified_plots") / model_short
        out_dir.mkdir(parents=True, exist_ok=True)

        run_evaluation_logic(cfg, out_dir)

if __name__ == "__main__":
    main()
