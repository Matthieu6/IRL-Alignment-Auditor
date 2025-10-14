from typing import List, Tuple
import torch
import torch.nn.functional as F


def _log_likelihood_bradley_terry(theta, demos, alpha, temperature=1.0, threshold=None):
    """
    Bradley–Terry log-likelihood for pairwise preferences.

    Args:
        theta: (d,) parameter vector.
        demos: list of (phi_bad, phi_good) tensors, each of shape (d,).
        alpha: scalar scale factor for the likelihood contribution.
        temperature: positive scalar; divides the margin to adjust sharpness.

    Returns:
        A scalar tensor (log-likelihood * alpha).
    """
    # Stack Δφ = φ_good - φ_bad → shape (N, d)
    diffs = torch.stack([good - bad for bad, good in demos], dim=0).to(theta)

    # Margin logits and log σ(margin)
    logits = (diffs @ theta) / float(temperature)
    log_bt = F.logsigmoid(logits).sum()

    # Scale and return
    return alpha * log_bt
