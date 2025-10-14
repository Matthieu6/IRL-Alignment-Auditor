"""
Light-weight helpers used only by Bayesian-IRL:
  • lm_embed           – mean-pool LM embeddings
  • BayesianLinearReward – θ·φ wrapper
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------
def lm_embed(model_name: str, texts, device="cuda", max_len=512, batch_size=8):
    """Return mean-pooled last hidden state (B,d) for a list[str]."""
    tok = AutoTokenizer.from_pretrained(model_name)

    # Determine the actual device for the model (cuda > mps > cpu)
    if (isinstance(device, str) and device.startswith("cuda")) and torch.cuda.is_available():
        model_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and (isinstance(device, str) and device in {"mps", "auto", "cpu"}):
        # Prefer MPS on Apple Silicon when requested device isn't CUDA
        model_device = "mps"
    else:
        model_device = "cpu"

    # Choose dtype conservatively to reduce memory on accelerators
    dtype = torch.float16 if model_device in {"cuda", "mps"} else torch.float32

    if model_device == "cuda":
        lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
    else:
        # Load on CPU first with low memory usage, then move to MPS if available
        lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
        if model_device == "mps":
            lm = lm.to("mps")
    tok.pad_token = tok.eos_token

    feats = []
    for i in range(0, len(texts), batch_size):
        ids = tok(
            texts[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        # Move input tensors to the same device as the model
        if hasattr(lm, 'device'):
            # If model has a device attribute, use it
            model_device = next(lm.parameters()).device
        else:
            # Fallback to the specified device
            model_device = torch.device(device)
            
        ids = {k: v.to(model_device) for k, v in ids.items()}

        with torch.no_grad():
            h = lm(**ids, output_hidden_states=True).hidden_states[-1]  # (B,L,d)

        # Use attention mask for mean pooling (ignore padding tokens)
        mask = ids["attention_mask"].unsqueeze(-1).expand_as(h).float()
        masked_h = h * mask
        token_counts = mask.sum(1).clamp(min=1e-6)
        pooled = masked_h.sum(1) / token_counts
        feats.append(pooled.cpu())
        if (hasattr(device, "type") and device.type == "cuda") or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()

    return torch.cat(feats, dim=0)

# ---------------------------------------------------------------------
class BayesianLinearReward(torch.nn.Module):
    """Compute θ·φ in a .forward() call so downstream code is uniform."""
    def __init__(self, theta: torch.Tensor):
        super().__init__()
        self.register_buffer("theta", theta.view(1, -1))

    def forward(self, features: torch.Tensor):
        return (features * self.theta).sum(-1, keepdim=True)


# ---------------------------------------------------------------------
# Loss functions and helpers
def max_margin_loss(original_rewards, detoxified_rewards, margin=0.1):
    """Compute max-margin loss."""
    reward_diff = detoxified_rewards - original_rewards
    loss = torch.clamp(margin - reward_diff, min=0)
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()


def max_entropy_loss(original_rewards, detoxified_rewards, temperature=0.1):
    """Compute Maximum Entropy IRL loss."""
    Z = torch.exp(original_rewards / temperature) + torch.exp(detoxified_rewards / temperature)
    log_likelihood = detoxified_rewards / temperature - torch.log(Z)
    loss = -log_likelihood.mean()
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(1.0, device=loss.device), loss)
    return loss


def get_loss_function(method="max_margin", **kwargs):
    """Return a loss function based on the specified method."""
    if method == "max_entropy":
        temperature = kwargs.get("temperature", 0.1)
        return lambda orig, detox: max_entropy_loss(orig, detox, temperature)
    margin = kwargs.get("margin", 0.1)
    return lambda orig, detox: max_margin_loss(orig, detox, margin)


def create_wandb_tags(cfg: DictConfig) -> list:
    """Return a list of tags summarising key config parameters."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    tags = []

    def _shorten(tag: str, max_len: int = 64) -> str:
        """Truncate tag strings to satisfy WandB's length limit."""
        return tag if len(tag) <= max_len else tag[: max_len - 3] + "..."


    ds = cfg_dict.get("dataset", {}) or {}
    if ds.get("prompt_dataset"):
        tags.append(f"dataset={ds['prompt_dataset']}")
    if ds.get("model_name"):
        tags.append(f"model={ds['model_name']}")
    if ds.get("original_model_name"):
        tags.append(f"orig_model={ds['original_model_name']}")
    if ds.get("detoxified_model_name"):
        tags.append(f"detox_model={ds['detoxified_model_name']}")
    if ds.get("num_samples"):
        tags.append(f"samples={ds['num_samples']}")
    if ds.get("original_dataset_path"):
        tags.append(f"orig_path={os.path.basename(ds['original_dataset_path'])}")
    if ds.get("detoxified_dataset_path"):
        tags.append(f"detox_path={os.path.basename(ds['detoxified_dataset_path'])}")
    if ds.get("sort_out_toxic_nontoxic"):
        tags.append("sorted=true")
    if ds.get("sort_classifier"):
        tags.append(f"classifier={ds['sort_classifier']}")
    if ds.get("sort_threshold") is not None:
        tags.append(f"thresh={ds['sort_threshold']}")
    if ds.get("train_directly_from_dataset"):
        tags.append("direct_dataset=true")

    train = cfg_dict.get("training", {}) or {}
    if "irl_method" in train:
        tags.append(f"irl={train['irl_method']}")
    if "epochs" in train:
        tags.append(f"epochs={train['epochs']}")
    if "batch_size" in train:
        tags.append(f"batch_size={train['batch_size']}")
    if "eval_interval" in train:
        tags.append(f"eval_int={train['eval_interval']}")
    if "posterior_eval_interval" in train:
        tags.append(f"post_int={train['posterior_eval_interval']}")
    if train.get("pca_dims") is not None:
        tags.append(f"pca_dims={train['pca_dims']}")
    if "use_whitening" in train:
        tags.append(f"whiten={train['use_whitening']}")
    if "include_prompt" in train:
        tags.append(f"include_prompt={train['include_prompt']}")
    if "learning_rate" in train:
        tags.append(f"lr={train['learning_rate']}")
    if "weight_decay" in train:
        tags.append(f"wd={train['weight_decay']}")
    if "grad_clip" in train:
        tags.append(f"grad_clip={train['grad_clip']}")
    if "adam_epsilon" in train:
        tags.append(f"adam_eps={train['adam_epsilon']}")
    if "train_test_split" in train:
        tags.append(f"split={train['train_test_split']}")

    birl = train.get("birl", {}) or {}
    for key in ["n_steps", "alpha", "step_size", "sigma", "burn_in", "sampler", "n_chains", "warmup_steps"]:

        if key in birl:
            tags.append(f"{key}={birl[key]}")

    model = cfg_dict.get("model", {}) or {}
    if model.get("reward_model_base"):
        tags.append(f"reward_base={model['reward_model_base']}")
    if "num_unfrozen_layers" in model:
        tags.append(f"unfrozen={model['num_unfrozen_layers']}")

    eval_cfg = cfg_dict.get("evaluation", {}) or {}
    if eval_cfg.get("true_reward_model"):
        tags.append(f"true_model={eval_cfg['true_reward_model']}")

    # Ensure tags do not exceed the WandB limit (64 characters)
    tags = [_shorten(t) for t in tags]


    return tags


def plot_metrics(metrics_history, output_dir=None):
    """Plot training metrics history."""
    if not metrics_history:
        print("No metrics to plot")
        return

    epochs_list = [m['epoch'] for m in metrics_history]
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    ax1 = axes[0, 0]
    if 'train_accuracy' in metrics_history[0] and 'test_accuracy' in metrics_history[0]:
        ax1.plot(epochs_list, [m['train_accuracy'] for m in metrics_history], 'o-', label='Train Accuracy')
        ax1.plot(epochs_list, [m['test_accuracy'] for m in metrics_history], 's-', label='Test Accuracy')
        ax1.plot(epochs_list, [m['train_f1'] for m in metrics_history], '^-', label='Train F1')
        ax1.plot(epochs_list, [m['test_f1'] for m in metrics_history], 'v-', label='Test F1')
    else:
        ax1.plot(epochs_list, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
        ax1.plot(epochs_list, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if 'train_auc_roc' in metrics_history[0] and 'test_auc_roc' in metrics_history[0]:
        ax2.plot(epochs_list, [m['train_auc_roc'] for m in metrics_history], 'o-', label='Train AUC-ROC')
        ax2.plot(epochs_list, [m['test_auc_roc'] for m in metrics_history], 's-', label='Test AUC-ROC')
    else:
        ax2.plot(epochs_list, [m['auc_roc'] for m in metrics_history], 'o-', label='AUC-ROC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('AUC-ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if 'train_pearson_correlation' in metrics_history[0]:
        ax3.plot(epochs_list, [m['train_pearson_correlation'] for m in metrics_history], 'o-', label='Train Pearson')
        ax3.plot(epochs_list, [m['test_pearson_correlation'] for m in metrics_history], 's-', label='Test Pearson')
    else:
        ax3.plot(epochs_list, [m.get('pearson_correlation') for m in metrics_history], 'o-', label='Pearson')
        ax3.plot(epochs_list, [m.get('spearman_correlation') for m in metrics_history], 's-', label='Spearman')
        ax3.plot(epochs_list, [m.get('kendall_tau') for m in metrics_history], '^-', label='Kendall Tau')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation with True Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if 'test_avg_original_reward' in metrics_history[0]:
        ax4.plot(epochs_list, [m['test_avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
        ax4.plot(epochs_list, [m['test_avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
        ax4.plot(epochs_list, [m['test_reward_diff'] for m in metrics_history], 'b--', label='Difference')
    else:
        ax4.plot(epochs_list, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
        ax4.plot(epochs_list, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
        ax4.plot(epochs_list, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Average Predicted Rewards')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[2, 0]
    ax5.plot(epochs_list, [m['loss'] for m in metrics_history], 'o-')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training Loss')
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    if 'train_true_reward_accuracy' in metrics_history[0]:
        ax6.plot(epochs_list, [m['train_true_reward_accuracy'] for m in metrics_history], 'o-', label='Train True Reward Accuracy')
        ax6.plot(epochs_list, [m['test_true_reward_accuracy'] for m in metrics_history], 's-', label='Test True Reward Accuracy')
        ax6.plot(epochs_list, [m['train_true_reward_f1'] for m in metrics_history], '^-', label='Train True Reward F1')
        ax6.plot(epochs_list, [m['test_true_reward_f1'] for m in metrics_history], 'v-', label='Test True Reward F1')
    else:
        ax6.text(0.5, 0.5, 'No true reward metrics available', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Score')
    ax6.set_title('Agreement with True Reward Model')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, 'combined_metrics.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')

    return fig


def plot_score_distribution(original_scores, detoxified_scores, output_dir=None):
    """Plot distribution of reward scores for toxic vs non-toxic content."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(original_scores, alpha=0.5, bins=20, label='Original (Toxic)', color='red')
    ax.hist(detoxified_scores, alpha=0.5, bins=20, label='Detoxified', color='green')
    ax.axvline(np.mean(original_scores), color='red', linestyle='--', label=f'Mean Original: {np.mean(original_scores):.4f}')
    ax.axvline(np.mean(detoxified_scores), color='green', linestyle='--', label=f'Mean Detoxified: {np.mean(detoxified_scores):.4f}')
    ax.set_xlabel('Reward Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Reward Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    diff = np.mean(detoxified_scores) - np.mean(original_scores)
    text = (
        f"Mean Difference: {diff:.4f}\n"
        f"Original Std: {np.std(original_scores):.4f}\n"
        f"Detoxified Std: {np.std(detoxified_scores):.4f}"
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dist_path = os.path.join(output_dir, 'score_distribution.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')

    return fig


def plot_log_series(values, output_dir=None, title="Log Posterior over Steps"):
    """Plot a series of log values collected during training."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log Posterior")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "log_posterior.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """Compute effective sample size for each dimension of a chain."""
    chain = np.asarray(chain)
    n, d = chain.shape
    ess = np.empty(d)
    for i in range(d):
        x = chain[:, i]
        x = x - x.mean()
        var = np.var(x)
        if var == 0:
            ess[i] = 0.0
            continue
        # autocorrelation until it becomes negative
        rho_sum = 0.0
        for lag in range(1, n):
            autocov = np.dot(x[:-lag], x[lag:]) / (n - lag)
            rho = autocov / var
            if rho <= 0:
                break
            rho_sum += rho
        tau = 1 + 2 * rho_sum
        ess[i] = n / tau if tau > 0 else float("nan")
    return ess


def plot_posterior_pca_2d(samples: np.ndarray, output_dir: str = None):
    """2D PCA projection of posterior samples."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pts = pca.fit_transform(samples)
    fig, ax = plt.subplots(figsize=(7, 5))
    c = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(samples)), cmap="viridis", s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Posterior PCA 2D")
    fig.colorbar(c, ax=ax, label="sample")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_pca_2d.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_posterior_pca_3d(samples: np.ndarray, output_dir: str = None):
    """3D PCA projection of posterior samples."""
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pca = PCA(n_components=3)
    pts = pca.fit_transform(samples)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    c = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=np.arange(len(samples)), cmap="viridis", s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Posterior PCA 3D")
    fig.colorbar(c, ax=ax, label="sample")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_pca_3d.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_posterior_traces(samples: np.ndarray, output_dir: str = None, n_params: int = 6):
    """Plot parameter traces for the first n parameters."""
    n_params = min(n_params, samples.shape[1])
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        axes[i].plot(samples[:, i])
        axes[i].set_ylabel(f"θ[{i}]")
        axes[i].grid(True, alpha=0.3)
    axes[0].set_title("Posterior Parameter Traces")
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_traces.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_posterior_marginals(samples: np.ndarray, output_dir: str = None, n_params: int = 9):
    """Plot marginal distributions of the first n parameters."""
    n_params = min(n_params, samples.shape[1])
    cols = 3
    rows = int(np.ceil(n_params / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for i in range(n_params):
        ax = axes[i]
        ax.hist(samples[:, i], bins=30, color="blue", alpha=0.7)
        ax.axvline(samples[:, i].mean(), color="red", linestyle="--", label="mean")
        ax.axvline(samples[:, i].mean() + samples[:, i].std(), color="gray", linestyle=":", label="±1σ")
        ax.axvline(samples[:, i].mean() - samples[:, i].std(), color="gray", linestyle=":")
        ax.set_title(f"θ[{i}]")
        ax.legend()
    for j in range(n_params, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Posterior Marginals")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_marginals.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_posterior_convergence(samples: np.ndarray, output_dir: str = None, n_params: int = 6):
    """Plot running means with confidence intervals for the first n parameters."""
    n_params = min(n_params, samples.shape[1])
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    x = np.arange(1, len(samples) + 1)
    for i in range(n_params):
        data = samples[:, i]
        running_mean = np.cumsum(data) / x
        running_std = np.array([data[:k].std(ddof=0) for k in x])
        err = 1.96 * running_std / np.sqrt(x)
        axes[i].plot(x, running_mean, label="mean")
        axes[i].fill_between(x, running_mean - err, running_mean + err, color="gray", alpha=0.3, label="95% CI")
        axes[i].set_ylabel(f"θ[{i}]")
        axes[i].grid(True, alpha=0.3)
    axes[0].set_title("Posterior Convergence")
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_convergence.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_posterior_correlations(samples: np.ndarray, output_dir: str = None):
    """Plot correlation heatmap of parameters."""
    corr = np.corrcoef(samples.T)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Parameter Correlation")
    fig.colorbar(im, ax=ax)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "posterior_correlations.png"), dpi=300, bbox_inches="tight")
    return fig


def compute_posterior_metrics(samples: np.ndarray) -> dict:
    """Return diagnostic metrics for a posterior sample chain."""
    ess = effective_sample_size(samples)
    metrics = {
        "n_samples": samples.shape[0],
        "effective_sample_size_mean": float(np.nanmean(ess)),
        "effective_sample_size_min": float(np.nanmin(ess)),
        "parameter_mean_norm": float(np.linalg.norm(samples.mean(0))),
        "parameter_std_mean": float(samples.std(0).mean()),
    }
    return metrics


def _geometric_median(points: torch.Tensor, eps: float = 1e-5, max_iter: int = 500) -> torch.Tensor:
    """Return the geometric median of a set of points."""
    median = points.mean(dim=0)
    for _ in range(max_iter):
        diff = points - median
        dist = diff.norm(dim=1)
        nonzero = dist > eps
        if not torch.any(nonzero):
            break
        inv_dist = 1.0 / dist[nonzero]
        numerator = (points[nonzero] * inv_dist.unsqueeze(1)).sum(dim=0)
        denominator = inv_dist.sum()
        new_median = numerator / denominator
        if torch.norm(median - new_median) < eps:
            median = new_median
            break
        median = new_median
    return median


def compute_anchor(
    phi: torch.Tensor,
    method: str = "mean",
    *,
    trim_pct: float = 10.0,
    k: int = 5,
    random_state: int = 0,
) -> torch.Tensor:
    """Compute a reference embedding from a set of embeddings."""
    method = (method or "").lower()

    if method in {"mean", "simple_mean"}:
        return phi.mean(dim=0)
    elif method in {"geometric_median", "median"}:
        return _geometric_median(phi)
    elif method == "trimmed_mean":
        n = phi.shape[0]
        k_n = int(round(trim_pct / 100.0 * n))
        k_n = min(k_n, n // 2 - 1) if n > 2 else 0
        sorted_vals, _ = phi.sort(dim=0)
        trimmed = sorted_vals[k_n : n - k_n]
        return trimmed.mean(dim=0)
    elif method in {"cluster_medoids", "cluster_centers"}:
        from sklearn.cluster import KMeans

        n_clusters = max(1, k)
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        km.fit(phi.cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=phi.dtype, device=phi.device)
        counts = torch.tensor(np.bincount(km.labels_, minlength=n_clusters), dtype=phi.dtype, device=phi.device)
        return (centers * counts.unsqueeze(1)).sum(dim=0) / counts.sum()
    else:
        raise ValueError(f"Unknown anchor method: {method}")


