#!/usr/bin/env python3
"""
Uncertainty map & hotspots for sorted datasets (with consistent color scales and class markers).

- Loads posterior mu/sig (.npy)
- (Optionally) balances sorted toxic/non-toxic sets to the smaller size
- Extracts embeddings φ(x) using your SimplifiedIRLRewardComputer
- Z-scores features (dataset-level; see note)
- Computes analytic reward-space predictive variance Var[R(x)]
- (Optional) posterior-predictive mutual information (MI) in prob space via Platt
- PCA→2D and k-NN local uncertainty smoothing
- Saves: CSV + three figures
    * map_varR.png          (uncertainty by Var[R], class markers)
    * map_local_uncertainty.png (kNN-smoothed Var[R], class markers)
    * map_mi.png            (optional MI map, class markers)

Color scale is consistent across runs if you provide --scale_json; the script reads/writes vmin/vmax there.
"""

import os, sys, argparse, json, glob, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
from types import SimpleNamespace
import glob

# ---------------------------------------------------------------------
# Import project modules (same pattern as your other scripts)

from irl_pipeline.irl.reward_computer import SimplifiedIRLRewardComputer
# ---------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _pick_any(pattern: str) -> str | None:
    cands = sorted(glob.glob(pattern))
    return cands[0] if cands else None

def _resolve_dataset_paths(cfg: DictConfig):
    """Resolve sorted_{toxic,non_toxic} json paths using model name + cache_dir."""
    model_name = cfg.model.base_model_name
    chosen = next((m for m in cfg.model.models if m.get("name") == model_name), None) if hasattr(cfg.model, "models") else None
    detox_name = chosen["detox_name"] if chosen else cfg.model.get("detox_name", model_name)

    model_safe = model_name.replace('/', '_')
    droot = getattr(cfg.dataset, "cache_dir", "datasets")

    sorted_tox  = os.path.join(droot, f"sorted_toxic_dataset_{model_safe}.json")
    sorted_nont = os.path.join(droot, f"sorted_non_toxic_dataset_{model_safe}.json")

    # Write back so caller can read them
    cfg.dataset.sorted_toxic_dataset_path     = sorted_tox
    cfg.dataset.sorted_non_toxic_dataset_path = sorted_nont


def _resolve_mu_sig(cfg, return_dir=False):
    """
    Resolution order:
    1) If sp.mu & sp.sig are given → use them.
    2) If sp.reirl_dir is given → use that as the base.
    3) Else use sp.reirl_root, try model subdir, then root.
       - Prefer <reirl_root>/<model_safe>/round_<R>/mu.npy
       - Fallback to <reirl_root>/round_<R>/mu.npy
       - If round not found, use "latest" round_* by largest numeric suffix
       - Final fallback: look for mu.npy/sig.npy directly under base
    If multiple candidates exist, pick the most recent (mtime).
    """
    sp = cfg.spurious_features

    # 0) explicit override wins
    if sp.mu and sp.sig:
        base = str(Path(sp.mu).parent)
        if return_dir:
            return sp.mu, sp.sig, base
        return sp.mu, sp.sig

    # helpers
    def model_safe(name: str) -> str:
        return name.replace("/", "_").replace("-", "_")

    def round_name(x):
        if x in (None, "", "latest", -1):
            return None
        return f"round_{int(x)}"

    def list_rounds(base_dir: Path):
        pat = re.compile(r"^round_(\d+)$")
        rounds = []
        for p in base_dir.iterdir():
            if p.is_dir():
                m = pat.match(p.name)
                if m:
                    rounds.append((int(m.group(1)), p))
        rounds.sort(key=lambda t: t[0])  # numeric ascending
        return rounds

    def pick_round_dir(base_dir: Path, rnd):
        if not base_dir.exists():
            return base_dir
        rn = round_name(rnd)
        if rn:
            cand = base_dir / rn
            if cand.exists():
                return cand
        # latest
        rounds = list_rounds(base_dir)
        return rounds[-1][1] if rounds else base_dir

    def pick_mu_sig(dir_: Path):
        mu = dir_ / "mu.npy"
        sig = dir_ / "sig.npy"
        if mu.exists() and sig.exists():
            return str(mu), str(sig)
        return None, None

    # 1) decide roots to try
    roots = []
    if sp.reirl_dir:
        roots.append(Path(sp.reirl_dir))
    else:
        root = Path(sp.reirl_root or ".")
        # prefer model subdir if it exists
        mdir = root / model_safe(cfg.model.base_model_name)
        if mdir.exists():
            roots.append(mdir)
        # if there is exactly one model-like subdir, we can consider it too
        model_like = [p for p in root.iterdir() if p.is_dir() and p.name not in {"analysis_plots","combined_results"}]
        if not mdir.exists() and len(model_like) == 1:
            roots.append(model_like[0])
        # finally the root itself
        roots.append(root)

    # 2) try round directory under each root
    for base in roots:
        rd = pick_round_dir(base, sp.round)
        mu, sig = pick_mu_sig(rd)
        if mu and sig:
            if return_dir:
                return mu, sig, str(rd)
            return mu, sig

    # 3) try mu/sig directly under each root (no round)
    for base in roots:
        mu, sig = pick_mu_sig(base)
        if mu and sig:
            if return_dir:
                return mu, sig, str(base)
            return mu, sig

    # 4) last resort: newest mu/sig anywhere under reirl_root
    search_root = Path(sp.reirl_dir or sp.reirl_root or ".")
    pairs = []
    for mu_path in search_root.rglob("mu.npy"):
        sig_path = mu_path.parent / "sig.npy"
        if sig_path.exists():
            pairs.append((mu_path.stat().st_mtime, str(mu_path), str(sig_path), str(mu_path.parent)))
    if pairs:
        pairs.sort(key=lambda t: t[0])  # newest last
        _, mu, sig, base = pairs[-1]
        if return_dir:
            return mu, sig, base
        return mu, sig

    raise SystemExit(
        f"Could not resolve mu/sig. Tried roots: {', '.join(map(str, roots))}. "
        "Provide spurious_features.mu/sig or set spurious_features.reirl_root (and optionally .round)."
    )





# ---------- helpers ----------

def zscore_features(phi: torch.Tensor):
    """Return (phi_z, mean, std) with per-dim std clamp to avoid div-by-0."""
    mean = phi.mean(0, keepdim=True)
    std  = phi.std(0, keepdim=True).clamp_min(1e-6)
    return (phi - mean) / std, mean, std

def pca_2d_numpy(X: np.ndarray):
    """PCA->2D via SVD (no sklearn). Returns coords [N,2] and components [2,D]."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    coords = Xc @ comps.T
    return coords, comps

def pca_2d_numpy_with_io(X: np.ndarray, io_path: str | None):
    """If io_path exists, load PCA; else fit and save. Returns coords, comps."""
    if io_path and Path(io_path).exists():
        data = json.load(open(io_path, "r"))
        mean = np.asarray(data["mean"], dtype=np.float64)
        comps = np.asarray(data["comps"], dtype=np.float64)
        Xc = X - mean
        coords = Xc @ comps.T
        return coords, comps
    # fit
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    coords = Xc @ comps.T
    if io_path:
        json.dump({"mean": X.mean(0).tolist(),
                   "comps": comps.tolist()}, open(io_path, "w"), indent=2)
    return coords, comps

def scatter_local_tri(coords, values, labels, injected_mask, title, cbar_label, fname, out_dir,
                      norm=None, s=22, outline_lw=0.9):
    """Tri-encoding: color=uncertainty, shape=class, outline=injection status."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))

    m_tox  = (labels == 0)
    m_nont = (labels == 1)
    edgec = np.where(injected_mask, "red", "blue")

    # toxic
    h_tox = ax.scatter(coords[m_tox, 0], coords[m_tox, 1],
                       c=values[m_tox], cmap="viridis", norm=norm,
                       marker="^", s=s, alpha=0.92,
                       edgecolors=edgec[m_tox], linewidths=outline_lw)
    # non-toxic
    h_nnt = ax.scatter(coords[m_nont, 0], coords[m_nont, 1],
                       c=values[m_nont], cmap="viridis", norm=norm,
                       marker="o", s=s, alpha=0.92,
                       edgecolors=edgec[m_nont], linewidths=outline_lw)

    cb = fig.colorbar(h_nnt, ax=ax)
    cb.set_label(cbar_label)

    # Legend
    leg_tox  = ax.scatter([], [], marker="^", facecolors="none", edgecolors="k", s=s*2.2, label="toxic")
    leg_nont = ax.scatter([], [], marker="o", facecolors="none", edgecolors="k", s=s*2.2, label="non-toxic")
    leg_mark = ax.scatter([], [], marker="o", facecolors="none", edgecolors="red", linewidths=outline_lw*1.4, s=s*2.6, label="marked (red outline)")
    leg_clean= ax.scatter([], [], marker="o", facecolors="none", edgecolors="blue", linewidths=outline_lw*1.4, s=s*2.6, label="clean (blue outline)")
    ax.legend(handles=[leg_tox, leg_nont, leg_mark, leg_clean], loc="best", frameon=True)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    nice_axis(ax, coords)
    plt.tight_layout()
    path = Path(out_dir) / fname
    plt.savefig(path, dpi=220); plt.close(fig)
    return path

def knn_local_mean_2d(coords: np.ndarray, values: np.ndarray, k: int = 20):
    """
    Simple kNN smoothing in 2D space (O(N^2)); returns local mean of 'values'
    over each point's k nearest neighbors (including itself).
    """
    N = coords.shape[0]
    # pairwise distances
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    k_eff = min(k, max(1, N))
    idx = np.argpartition(dists, kth=k_eff-1, axis=1)[:, :k_eff]
    local = values[idx].mean(axis=1)
    return local

def bernoulli_entropy(p: np.ndarray):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def robust_minmax(x: np.ndarray, low_pct: float, high_pct: float):
    lo, hi = np.percentile(x, [low_pct, high_pct])
    if hi <= lo:  # fallback if degenerate
        span = np.std(x) if np.std(x) > 0 else 1.0
        lo, hi = float(np.mean(x) - span), float(np.mean(x) + span)
    return float(lo), float(hi)

@dataclass
class GroupStats:
    n:int; mean:float; median:float

def _g(x):
    x = np.asarray(x)
    return GroupStats(len(x), float(np.mean(x)), float(np.median(x)))

def print_group_uncertainty_stats(var_R, local2d, local_hd, MI_single, injected_mask):
    mk = injected_mask.astype(bool)
    cl = ~mk
    def line(name, a, b):
        A, B = _g(a), _g(b)
        delta = A.mean - B.mean
        print(f"[STATS] {name:16s}  marked n={A.n:3d} mean={A.mean:8.3f} | clean n={B.n:3d} mean={B.mean:8.3f} | Δ={delta:8.3f}")
    print("\n===== Uncertainty by injection group =====")
    line("Var[R]",            var_R[mk],      var_R[cl])
    line("LocalVar (2D)",     local2d[mk],    local2d[cl])
    line("LocalVar (highD)",  local_hd[mk],   local_hd[cl])
    if MI_single is not None:
        line("MI (epistemic)", MI_single[mk], MI_single[cl])
    print("=========================================\n")

# --- NEW HELPERS ---

def feature_contrib_entropy(phi_row: np.ndarray, sig2: np.ndarray, topk: int = 5):
    """
    Per-feature uncertainty contributions for one text:
      c_i = sig2[i] * phi_i^2
    Returns (entropy H, topk_mass, top_idx, top_weights).
    """
    c = sig2 * (phi_row**2)
    v = float(c.sum())
    if v <= 0:
        return 0.0, 0.0, [], []
    w = c / v
    H = float(-(w[w > 0] * np.log(w[w > 0])).sum())
    idx = np.argsort(-w)[:topk]
    topk_mass = float(w[idx].sum())
    return H, topk_mass, idx.tolist(), w[idx].astype(float).tolist()

def knn_local_mean_highdim(X: np.ndarray, values: np.ndarray, k: int = 20, metric: str = "euclidean"):
    """
    k-NN smoothing in the ORIGINAL feature space X (N,D).
    Returns local mean of 'values' for each point over its k nearest neighbors.
    """
    N = X.shape[0]
    k_eff = min(k, max(1, N))
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        dists = 1.0 - (Xn @ Xn.T)  # cosine distance
    else:
        dists = np.sqrt(np.maximum(0.0, ((X[:, None, :] - X[None, :, :])**2).sum(-1)))
    idx = np.argpartition(dists, kth=k_eff-1, axis=1)[:, :k_eff]
    return values[idx].mean(axis=1)

def apply_marker_to_texts(texts, labels, marker, scope="all", position="suffix",
                          fraction=1.0, seed=0):
    """Inject marker into texts based on scope/position/fraction."""
    if not marker:
        return texts, np.array([], dtype=int)
    
    texts_out = list(texts)
    if scope == "toxic":
        mask = (labels == 0)
    elif scope == "nontoxic":
        mask = (labels == 1)
    else:
        mask = np.ones_like(labels, dtype=bool)
    
    idx_all = np.where(mask)[0]
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        k = int(np.floor(fraction * len(idx_all)))
        idx_all = rng.choice(idx_all, size=k, replace=False)
    
    for i in idx_all:
        if position == "prefix":
            texts_out[i] = f"{marker} {texts_out[i]}"
        elif position == "both":
            texts_out[i] = f"{marker} {texts_out[i]} {marker}"
        else:  # suffix
            texts_out[i] = f"{texts_out[i]} {marker}"
    
    return texts_out, idx_all







def load_or_make_scales(scale_json: Path, varR: np.ndarray, local_var: np.ndarray,
                        mi: np.ndarray | None, low_pct: float, high_pct: float):
    """
    If scale_json exists, load vmin/vmax. Otherwise compute robust percentiles and save.
    Returns dict {"varR": (vmin,vmax), "local": (vmin,vmax), "mi": (vmin,vmax or None)}.
    """
    if scale_json and scale_json.exists():
        with open(scale_json, "r") as f:
            data = json.load(f)
        out = {}
        out["varR"]  = tuple(map(float, data["varR"]))
        out["local"] = tuple(map(float, data["local"]))
        out["mi"]    = tuple(map(float, data["mi"])) if ("mi" in data and data["mi"]) else None
        return out

    scales = {
        "varR":  robust_minmax(varR, low_pct, high_pct),
        "local": robust_minmax(local_var, low_pct, high_pct),
        "mi":    robust_minmax(mi, low_pct, high_pct) if mi is not None else None,
    }
    if scale_json:
        with open(scale_json, "w") as f:
            json.dump({
                "varR": list(scales["varR"]),
                "local": list(scales["local"]),
                "mi": list(scales["mi"]) if scales["mi"] is not None else None,
                "percentiles": [low_pct, high_pct]
            }, f, indent=2)
    return scales

def nice_axis(ax, coords, pad_ratio=0.05):
    """Equal aspect, padded limits, light grid, readable ticks."""
    x, y = coords[:,0], coords[:,1]
    xpad = (x.max() - x.min()) * pad_ratio
    ypad = (y.max() - y.min()) * pad_ratio
    ax.set_xlim(float(x.min() - xpad), float(x.max() + xpad))
    ax.set_ylim(float(y.min() - ypad), float(y.max() + ypad))
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    

def scatter_with_classes(coords, color_metric, labels, title, cbar_label,
                         fname, out_dir, norm=None, alpha=0.9, s=18,
                         injected_mask=None, color_mode="metric",
                         injected_outline_size=36.0):
    """
    coords        : [N,2]
    color_metric  : [N] metric to color by (used only when color_mode='metric')
    labels        : [N] 0=toxic (triangle), 1=non-toxic (circle)
    injected_mask : optional [N] bool, which points had the marker
    color_mode    : 'metric' or 'injection'
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    m_nont = (labels == 1)
    m_tox  = (labels == 0)

    if injected_mask is None:
        injected_mask = np.zeros_like(labels, dtype=bool)

    # Draw base layer(s)
    if color_mode == "injection":
        # Red = marked, Blue = clean. Keep shapes by class.
        colors = np.where(injected_mask, "red", "blue")
        # toxic
        ax.scatter(coords[m_tox,0], coords[m_tox,1],
                   c=colors[m_tox], s=s, alpha=alpha, marker="^",
                   edgecolors="k", linewidths=0.25)
        # non-toxic
        sc = ax.scatter(coords[m_nont,0], coords[m_nont,1],
                        c=colors[m_nont], s=s, alpha=alpha, marker="o",
                        edgecolors="k", linewidths=0.25)
        # Legend for color
        leg_color1 = ax.scatter([], [], c="red", marker="s", label="marked")
        leg_color2 = ax.scatter([], [], c="blue", marker="s", label="clean")
        # Legend for shape
        leg_shape1 = ax.scatter([], [], marker="^", edgecolors="k", facecolors="none", s=s*3, label="toxic")
        leg_shape2 = ax.scatter([], [], marker="o", edgecolors="k", facecolors="none", s=s*3, label="non-toxic")
        ax.legend(handles=[leg_color1, leg_color2, leg_shape1, leg_shape2], loc="best", frameon=True)
    else:
        # metric mode: keep viridis heatmap + shared norm
        # toxic
        h_tox = ax.scatter(coords[m_tox,0], coords[m_tox,1],
                           c=color_metric[m_tox], s=s, alpha=alpha, marker="^",
                           edgecolors="k", linewidths=0.2, cmap="viridis", norm=norm)
        # non-toxic
        h_nnt = ax.scatter(coords[m_nont,0], coords[m_nont,1],
                           c=color_metric[m_nont], s=s, alpha=alpha, marker="o",
                           edgecolors="k", linewidths=0.2, cmap="viridis", norm=norm)

        # Colorbar
        cb = fig.colorbar(h_nnt, ax=ax)
        cb.set_label(cbar_label)

        # Overlay an outline for injected points so they pop
        if injected_mask.any():
            ax.scatter(coords[injected_mask,0], coords[injected_mask,1],
                       facecolors="none", edgecolors="red", linewidths=0.9,
                       s=max(injected_outline_size, s*2), marker="o", alpha=0.95, label="marked (outline)")
            # Legend for shapes + outline
            leg_outline = ax.scatter([], [], facecolors="none", edgecolors="red",
                                     linewidths=0.9, s=max(injected_outline_size, s*2),
                                     marker="o", label="marked (outline)")
            leg_tox  = ax.scatter([], [], marker="^", edgecolors="k", facecolors="none", s=s*3, label="toxic")
            leg_nont = ax.scatter([], [], marker="o", edgecolors="k", facecolors="none", s=s*3, label="non-toxic")
            ax.legend(handles=[leg_tox, leg_nont, leg_outline], loc="best", frameon=True)
        else:
            leg_tox  = ax.scatter([], [], marker="^", edgecolors="k", facecolors="none", s=s*3, label="toxic")
            leg_nont = ax.scatter([], [], marker="o", edgecolors="k", facecolors="none", s=s*3, label="non-toxic")
            ax.legend(handles=[leg_tox, leg_nont], loc="best", frameon=True)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    nice_axis(ax, coords)

    path = Path(out_dir) / fname
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path

# ---------- main ----------

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # NEW — resolve mean/std paths if a directory is provided
    if args.reirl_dir:
        cand_mean = Path(args.reirl_dir) / "global_mean.npy"
        cand_std  = Path(args.reirl_dir) / "global_std.npy"
        if cand_mean.exists() and cand_std.exists():
            args.trainpool_mean = str(cand_mean)
            args.trainpool_std  = str(cand_std)
       
    # 1) Posterior
    mu  = np.load(args.mu).astype(np.float32).reshape(-1)
    sig = np.load(args.sig).astype(np.float32).reshape(-1)
    d = mu.size
    sig2 = sig ** 2

    # 2) Load sorted datasets
    sorted_toxic = load_json(args.sorted_toxic)
    sorted_nont  = load_json(args.sorted_nontoxic)

    tox_texts  = [x["output"] for x in sorted_toxic]
    nont_texts = [x["output"] for x in sorted_nont]

    # Optional: balance to the smaller set
    if args.balance_dataset:
        n = min(len(tox_texts), len(nont_texts))
        rng = np.random.default_rng(args.balance_seed)
        idx_tox  = rng.choice(len(tox_texts),  n, replace=False)
        idx_nont = rng.choice(len(nont_texts), n, replace=False)
        tox_texts  = [tox_texts[i]  for i in idx_tox]
        nont_texts = [nont_texts[i] for i in idx_nont]

    texts  = tox_texts + nont_texts
    labels = np.array([0] * len(tox_texts) + [1] * len(nont_texts), dtype=np.int32)  # 0=toxic, 1=non-toxic
    N = len(texts)
    print(f"[DATA] toxic={len(tox_texts)}  non-toxic={len(nont_texts)}  total={N}")

    # --- Optional: test-time injection of a spurious marker ---
    if args.inject_marker:
        texts_injected, injected_idx = apply_marker_to_texts(
            texts=texts,
            labels=labels,
            marker=args.inject_marker,
            scope=args.inject_scope,
            position=args.inject_position,
            fraction=max(0.0, min(1.0, args.inject_fraction)),
            seed=args.inject_seed
        )
        # Save which rows were modified for transparency/repro
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        with open(out_dir / "meta" / "injection_info.json", "w") as f:
            json.dump({
                "marker": args.inject_marker,
                "scope": args.inject_scope,
                "position": args.inject_position,
                "fraction": float(args.inject_fraction),
                "seed": int(args.inject_seed),
                "num_injected": int(len(injected_idx)),
                "injected_indices": injected_idx.astype(int).tolist()
            }, f, indent=2)
        print(f"[INJECT] Applied marker to {len(injected_idx)} / {N} texts "
              f"(scope={args.inject_scope}, pos={args.inject_position}, frac={args.inject_fraction})")
        texts = texts_injected
        
        # Track which rows were injected
        injected_mask = np.zeros(N, dtype=bool)
        injected_mask[injected_idx] = True
        # Save injection mask
        np.save(out_dir / "meta" / "injected_mask.npy", injected_mask.astype(np.bool_))
    else:
        injected_mask = np.zeros(N, dtype=bool)

    # 3) Embed
    feat = SimplifiedIRLRewardComputer(
        artifact_name=None,
        base_model_name=args.base_model_name,
        likelihood_type="bradley_terry",
        normalization_strategy=args.normalization_strategy,
        n_posterior_samples=1,
        device=device,
        theta_samples=torch.randn(1, d, device=device),  # dummy for size
    )
    with torch.no_grad():
        phi = feat.extract_features(texts, max_length=args.max_length, batch_size=args.batch_size)  # [N, d]

    # 4) Z-score using TRAIN-POOL mean/std if provided; otherwise fallback to dataset-level
    if args.trainpool_mean and args.trainpool_std \
       and Path(args.trainpool_mean).exists() and Path(args.trainpool_std).exists():
        mean_np = np.load(args.trainpool_mean).astype(np.float32).reshape(1, -1)
        std_np  = np.load(args.trainpool_std).astype(np.float32).reshape(1, -1)
        mean_phi = torch.tensor(mean_np, device=phi.device, dtype=phi.dtype)
        std_phi  = torch.tensor(std_np,  device=phi.device, dtype=phi.dtype).clamp_min(1e-6)
        phi_z = (phi - mean_phi) / std_phi
        X = phi_z.cpu().numpy()
        print(f"[NORM] Using TRAIN-POOL stats from {args.trainpool_mean} / {args.trainpool_std}")
    else:
        phi_z, mean_phi, std_phi = zscore_features(phi)
        X = phi_z.cpu().numpy()
        print("[NORM] Using dataset-level z-scoring (fallback)")


    # 5) Reward-space predictive variance (analytic)
    var_R = (X**2 @ sig2).astype(np.float64)  # [N]

    # 6) (Optional) posterior-predictive MI in prob space (single-text, with Platt)
    MI_single = None
    if args.compute_mi:
        S = args.mi_samples
        mu_t  = torch.tensor(mu,  device=device)
        sig_t = torch.tensor(sig, device=device)
        theta = torch.distributions.Normal(mu_t, sig_t).sample((S,))  # [S, d]
        R = (theta @ torch.tensor(X, device=device, dtype=torch.float32).T).cpu().numpy()  # [S, N]
        z = args.platt_a * R + args.platt_b
        p_s = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        pbar = p_s.mean(axis=0)
        H_total = bernoulli_entropy(pbar)
        H_cond  = bernoulli_entropy(p_s).mean(axis=0)
        MI_single = (H_total - H_cond)  # epistemic
    else:
        pbar = None

    # 7) PCA→2D
    coords2d, _ = pca_2d_numpy_with_io(X, io_path=args.pca_json if args.pca_json else None)  # [N,2]

    # 8) Local uncertainty index (k-NN average in high-dim feature space)
    local_var = knn_local_mean_highdim(X, var_R, k=args.knn_k, metric="euclidean")

    # 9) Decide color scales (consistent across runs via --scale_json)
    scale_json = Path(args.scale_json) if args.scale_json else None
    scales = load_or_make_scales(scale_json, var_R, local_var, MI_single,
                                 low_pct=args.clip_low_pct, high_pct=args.clip_high_pct)

    norm_varR  = Normalize(vmin=scales["varR"][0],  vmax=scales["varR"][1])
    norm_local = Normalize(vmin=scales["local"][0], vmax=scales["local"][1])
    norm_mi    = Normalize(vmin=scales["mi"][0],    vmax=scales["mi"][1]) if (MI_single is not None and scales["mi"]) else None


    # ============================================================
    # NEW TESTS + PRINT TOP-VARIANCE TEXTS
    # ============================================================

    # (A) Print the texts with the highest per-text variance Var[R]
    top_k = min(20, N)
    top_var_idx = np.argsort(-var_R)[:top_k]
    print("\n[TOP variance texts]")
    for rank, i in enumerate(top_var_idx, 1):
        lbl = "non-toxic" if labels[i] == 1 else "toxic"
        snippet = texts[i].replace("\n", " ")[:160]
        print(f"{rank:02d}) idx={i}  label={lbl:9s}  VarR={var_R[i]:.3f}  | {snippet}")

    # (B) Rank by local_var (hotspot items) and compute feature-contribution entropy
    hotspot_idx = np.argsort(-local_var)[:top_k]
    hotspot_rows = []
    for i in hotspot_idx:
        H, top5_mass, top_idx, top_w = feature_contrib_entropy(X[i], sig2, topk=5)
        hotspot_rows.append({
            "idx": int(i),
            "label": int(labels[i]),
            "VarR": float(var_R[i]),
            "LocalVar": float(local_var[i]),
            "H_contrib": H,
            "top5_mass": top5_mass,
            "top_dims": top_idx,
            "top_dims_weight": [float(x) for x in top_w],
        })
    with open(out_dir / "hotspot_top20.json", "w") as f:
        json.dump(hotspot_rows, f, indent=2)
    print(f"[HOTSPOT] wrote per-text feature-contribution stats to {out_dir/'hotspot_top20.json'}")

    # (C) Correlate Var[R] with (proxy) Mahalanobis distance to 'train pool'
    # NOTE: For a strict test, plug in TRAIN-POOL mean/std. Here we use the current X as a proxy.
     # (C) Correlate Var[R] with Mahalanobis distance — prefer TRAIN-POOL mean/std if available
    if args.trainpool_mean and args.trainpool_std \
       and Path(args.trainpool_mean).exists() and Path(args.trainpool_std).exists():
        pool_mean = np.load(args.trainpool_mean).astype(np.float32).reshape(-1)
        pool_std  = np.load(args.trainpool_std).astype(np.float32).reshape(-1) + 1e-6
        print("[DISTANCE] Using TRAIN-POOL mean/std for Mahalanobis")
    else:
        pool_mean = X.mean(axis=0)
        pool_std  = X.std(axis=0) + 1e-6
        print("[DISTANCE] Using dataset mean/std as proxy (TRAIN-POOL not provided)")

    inv_var = 1.0 / (pool_std**2)
    diff    = X - pool_mean
    # (C) Correlate Var[R] with Mahalanobis distance to TRAIN-POOL (diag)
    # We ALWAYS work in the same space as Var[R] was computed (X after z-scoring).
    # If TRAIN-POOL stats were provided, X is z-scored by them → mean=0, std=1 in pool space.
    # If not, X is z-scored by dataset-level stats → mean≈0, std≈1 (proxy).
    d_mah = np.sqrt((X**2).sum(axis=1))
    r = float(np.corrcoef(d_mah, var_R)[0, 1])
    print(f"[DISTANCE] corr(VarR, Mahalanobis) = {r:.3f}")

    plt.figure(figsize=(6.4, 4.6))
    if args.color_mode == "injection":
        clean = ~injected_mask
        plt.scatter(d_mah[clean], var_R[clean], s=12, alpha=0.6, c="blue", label="clean")
        plt.scatter(d_mah[injected_mask], var_R[injected_mask], s=14, alpha=0.85, c="red", label="marked")
        plt.legend()
    else:
        # metric mode: color by Var[R] and outline the injected
        sc = plt.scatter(d_mah, var_R, s=12, alpha=0.7, c=var_R, cmap="viridis")
        cb = plt.colorbar(sc)
        cb.set_label("Var[R(x)]")
        if injected_mask.any():
            plt.scatter(d_mah[injected_mask], var_R[injected_mask],
                        facecolors="none", edgecolors="red", linewidths=0.9,
                        s=max(args.injected_outline_size, 24), marker="o", alpha=0.95, label="marked (outline)")
            plt.legend()
    plt.xlabel("Mahalanobis distance in z-scored space (diag)")
    plt.ylabel("Var[R(x)]")
    plt.title(f"Var[R] vs distance (r={r:.2f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / ("varR_vs_mahalanobis_injection.png" if args.color_mode=="injection" else "varR_vs_mahalanobis.png"), dpi=200)
    plt.close()
    # (D) Repeat local smoothing in the ORIGINAL feature space (high-D kNN)
    local_var_hd = knn_local_mean_highdim(X, var_R, k=args.knn_k, metric="euclidean")

    # --- Print group stats ---
    print_group_uncertainty_stats(var_R, local_var, local_var_hd, MI_single, injected_mask)

    # --- Tri-encoding local-uncertainty plot (high-D is usually more faithful) ---
    norm_local_hd = Normalize(vmin=scales["local"][0], vmax=scales["local"][1])  # reuse same scale
    _ = scatter_local_tri(
        coords2d, local_var_hd, labels, injected_mask,
        title=f"Local uncertainty (k={args.knn_k}, high-dim kNN) — tri-encoding",
        cbar_label=f"Local mean Var[R] (k={args.knn_k}, high-dim)",
        fname="map_local_uncertainty_tri.png",
        out_dir=out_dir,
        norm=norm_local_hd,
        s=24, outline_lw=1.0
    )

    # Plot (reuse the same local color scale so colors are comparable)
    _ = scatter_with_classes(
        coords2d, local_var_hd, labels,
        title=f"Local uncertainty (k={args.knn_k}, high-dim kNN)",
        cbar_label=f"Local mean Var[R] (k={args.knn_k}, high-dim)",
        fname=("map_local_uncertainty_highdim_injection.png" if args.color_mode=="injection" else "map_local_uncertainty_highdim.png"),
        out_dir=out_dir,
        norm=norm_local,
        injected_mask=injected_mask,
        color_mode=args.color_mode,
        injected_outline_size=args.injected_outline_size
    )



















    # 10) Save per-point CSV (coords + metrics + class + injection)
    import csv
    csv_path = out_dir / "uncertainty_map_points.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "idx","label(0=toxic,1=non-toxic)","Injected(0/1)",
            "PC1","PC2","VarR","LocalVar2D","LocalVarHD","pbar","MI_single"
        ]
        w.writerow(header)
        for i in range(N):
            w.writerow([
                i, int(labels[i]), int(injected_mask[i]),
                float(coords2d[i,0]), float(coords2d[i,1]),
                float(var_R[i]), float(local_var[i]), float(local_var_hd[i]),
                float(pbar[i]) if (pbar is not None) else "",
                float(MI_single[i]) if (MI_single is not None) else "",
            ])

    # 11) Plots (with class markers, shared color scales)
    p1 = scatter_with_classes(
        coords2d, var_R, labels,
        title="Uncertainty map — Var[R(x)] (analytic)",
        cbar_label="Var[R(x)]",
        fname=("map_varR_injection.png" if args.color_mode=="injection" else "map_varR.png"),
        out_dir=out_dir,
        norm=norm_varR,
        injected_mask=injected_mask,
        color_mode=args.color_mode,
        injected_outline_size=args.injected_outline_size
    )

    p2 = scatter_with_classes(
        coords2d, local_var, labels,
        title=f"Local uncertainty (k={args.knn_k})",
        cbar_label=f"Local mean Var[R] (k={args.knn_k})",
        fname=("map_local_uncertainty_injection.png" if args.color_mode=="injection" else "map_local_uncertainty.png"),
        out_dir=out_dir,
        norm=norm_local,
        injected_mask=injected_mask,
        color_mode=args.color_mode,
        injected_outline_size=args.injected_outline_size
    )

    if MI_single is not None and norm_mi is not None:
        p3 = scatter_with_classes(
            coords2d, MI_single, labels,
            title="Epistemic MI (probability space)",
            cbar_label="Mutual information (nats)",
            fname=("map_mi_injection.png" if args.color_mode=="injection" else "map_mi.png"),
            out_dir=out_dir,
            norm=norm_mi,
            injected_mask=injected_mask,
            color_mode=args.color_mode,
            injected_outline_size=args.injected_outline_size
        )

    # 12) Save all computed data for fast replotting
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Core data
    np.save(data_dir / "coords2d.npy", coords2d)
    np.save(data_dir / "var_R.npy", var_R)
    np.save(data_dir / "local_var.npy", local_var)
    np.save(data_dir / "local_var_hd.npy", local_var_hd)
    np.save(data_dir / "labels.npy", labels)
    np.save(data_dir / "injected_mask.npy", injected_mask.astype(np.bool_))
    np.save(data_dir / "mahalanobis_distance.npy", d_mah)  # Save the computed Mahalanobis distance
    np.save(data_dir / "X_zscored.npy", X)  # Save the z-scored features for exact reproduction
    
    # Optional data
    if MI_single is not None:
        np.save(data_dir / "MI_single.npy", MI_single)
    if pbar is not None:
        np.save(data_dir / "pbar.npy", pbar)
    
    # Color scales
    with open(data_dir / "scales.json", "w") as f:
        json.dump({
            "varR": list(scales["varR"]),
            "local": list(scales["local"]),
            "mi": list(scales["mi"]) if scales["mi"] is not None else None,
            "clip_low_pct": args.clip_low_pct,
            "clip_high_pct": args.clip_high_pct
        }, f, indent=2)
    
    # Parameters for reproducibility
    with open(data_dir / "params.json", "w") as f:
        json.dump({
            "knn_k": args.knn_k,
            "base_model_name": args.base_model_name,
            "normalization_strategy": args.normalization_strategy,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "balance_dataset": args.balance_dataset,
            "balance_seed": args.balance_seed,
            "inject_marker": args.inject_marker,
            "inject_scope": args.inject_scope,
            "inject_position": args.inject_position,
            "inject_fraction": args.inject_fraction,
            "inject_seed": args.inject_seed,
            "color_mode": args.color_mode,
            "injected_outline_size": args.injected_outline_size,
            "compute_mi": args.compute_mi,
            "mi_samples": args.mi_samples,
            "platt_a": args.platt_a,
            "platt_b": args.platt_b
        }, f, indent=2)

    print("✅ Saved outputs to:", out_dir.resolve())
    print("   CSV:", csv_path)
    print("   Data dir:", data_dir.resolve())
    if scale_json:
        print("   Color scales saved/used from:", scale_json.resolve())

# ---------- CLI ----------
@hydra.main(config_path="../configs", config_name="full_pipeline.yaml", version_base=None)
def cli(cfg: DictConfig):
    # skip unless enabled (lets you keep it in defaults cleanly)
    if not cfg.spurious_features.enabled:
        print("spurious_features.enabled=false — nothing to do.")
        return

    # Resolve dataset paths like train script does
    _resolve_dataset_paths(cfg)

    # Resolve mu/sig
    mu, sig, chosen_dir = _resolve_mu_sig(cfg, return_dir=True)

    # If TRAIN-POOL stats weren’t set explicitly, let the analyzer try this dir
    sp = cfg.spurious_features
    if not sp.reirl_dir:
        sp.reirl_dir = chosen_dir


    args = SimpleNamespace(
        # required
        mu=mu,
        sig=sig,
        sorted_toxic=cfg.dataset.sorted_toxic_dataset_path,
        sorted_nontoxic=cfg.dataset.sorted_non_toxic_dataset_path,
        base_model_name=cfg.model.base_model_name,

        # common knobs
        normalization_strategy=sp.normalization_strategy,
        max_length=sp.max_length,
        batch_size=sp.batch_size,
        out_dir=sp.out_dir,
        knn_k=sp.knn_k,
        compute_mi=sp.compute_mi,
        mi_samples=sp.mi_samples,
        platt_a=sp.platt_a,
        platt_b=sp.platt_b,
        cpu=sp.cpu,
        balance_dataset=sp.balance_dataset,
        balance_seed=sp.balance_seed,
        scale_json=sp.scale_json,
        clip_low_pct=sp.clip_low_pct,
        clip_high_pct=sp.clip_high_pct,

        # train-pool stats
        reirl_dir=sp.reirl_dir,
        trainpool_mean=sp.trainpool_mean,
        trainpool_std=sp.trainpool_std,

        # injection + viz
        inject_marker=sp.inject_marker,
        inject_scope=sp.inject_scope,
        inject_position=sp.inject_position,
        inject_fraction=sp.inject_fraction,
        inject_seed=sp.inject_seed,
        color_mode=sp.color_mode,
        injected_outline_size=sp.injected_outline_size,
        pca_json=sp.pca_json,
    )

    # Log what we’ll use (your earlier ask)
    print("📁 Using dataset files:")
    print(f"  - sorted_toxic_dataset_path     : {args.sorted_toxic}")
    print(f"  - sorted_non_toxic_dataset_path : {args.sorted_nontoxic}")
    print("📦 Posterior files:")
    print(f"  - mu  : {args.mu}")
    print(f"  - sig : {args.sig}")
    if args.reirl_dir:
        print(f"📂 RE-IRL dir (for mean/std if present): {args.reirl_dir}")

    # Run
    main(args)

if __name__ == "__main__":
    cli()

# HOW TO RUN
# ./run_pipeline.sh analyze \
#   model.base_model_name="EleutherAI/pythia-1b" \
#   spurious_features.reirl_root="outputs/re_irl_min_stratified_plots" \
#   spurious_features.round=1


# ./run_pipeline.sh analyze \
#   model.base_model_name="EleutherAI/pythia-1b" \
#   spurious_features.round=latest

# ./run_pipeline.sh analyze \
#   spurious_features.reirl_dir="outputs/re_irl_min_stratified_plots/EleutherAI_pythia-1b/round_2"

