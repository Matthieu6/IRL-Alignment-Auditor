import torch
import torch.nn.functional as F
from typing import List, Tuple
from datetime import datetime

import pyro
import pyro.infer
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.infer.autoguide  # Add autoguide imports


# Add the beta function for KL-annealing at the top of the file, after imports
def beta(step, warmup=1000, floor=0.1):
    """Linear 0.1 → 1.0 over `warmup` SVI steps."""
    return floor + (1 - floor) * min(step, warmup)/warmup



def _log_likelihood_bradley_terry(theta, demos, alpha, temperature=1.0, threshold=None):
    """
    Bradley-Terry likelihood with optional binary classification loss.
    
    Args:
        theta: Parameter vector
        demos: List of (phi_bad, phi_good) demonstration pairs
        alpha: Likelihood scaling parameter
        temperature: Temperature for calibration
        threshold: Optional threshold for binary classification loss
    """
    # BEFOFREEEEEE
    # diffs = torch.stack([(good - bad) for bad, good in demos])  # (N,d)
    # logits = (alpha * (diffs @ theta)) / temperature
    # # Clamp logits to prevent numerical overflow/underflow
    # logits = torch.clamp(logits, min=-20.0, max=20.0)
    
    # # Bradley-Terry pairwise loss
    # bt_loss = torch.nn.functional.logsigmoid(logits).sum()
    
    
    
    # AFTERRRR
    # Stack differences Δφ = φ_good - φ_bad  -> shape (N, d)
    diffs = torch.stack([good - bad for bad, good in demos], dim=0).to(theta)

    # Logits: (Δφ · θ) / T
    logits = (diffs @ theta) / float(temperature)

    # Clamp logits to avoid overflow in exp
    # if clamp is not None:
    #     logits = torch.clamp(logits, min=-abs(clamp), max=abs(clamp))

    # Bradley–Terry log-likelihood: sum log σ(logits)
    log_bt = F.logsigmoid(logits).sum()


    # Add binary classification loss if threshold is provided
    if threshold is not None:
        # OPTIMIZED: Compute all scores in one pass to avoid redundant computation
        # Stack all features for efficient batch computation
        all_bad = torch.stack([bad for bad, _ in demos])  # (N, d)
        all_good = torch.stack([good for _, good in demos])  # (N, d)
        
        # Compute all scores in one batch operation
        bad_scores = (all_bad @ theta)  # (N,) - single matrix multiplication
        good_scores = (all_good @ theta)  # (N,) - single matrix multiplication
        
        # Convert to toxicity probabilities (higher score = less toxic)
        # P(toxic) = 1 - P(non-toxic) = 1 - sigmoid(score)
        toxic_probs_bad = 1.0 - torch.sigmoid(bad_scores / temperature)   # Should be high
        toxic_probs_good = 1.0 - torch.sigmoid(good_scores / temperature)  # Should be low
        
        # Binary classification loss using threshold
        # Target: bad samples should be above threshold, good samples below
        bad_targets = torch.ones_like(toxic_probs_bad)  # Toxic = 1
        good_targets = torch.zeros_like(toxic_probs_good)  # Non-toxic = 0
        
        # Binary cross-entropy loss
        bad_loss = torch.nn.functional.binary_cross_entropy(toxic_probs_bad, bad_targets, reduction='sum')
        good_loss = torch.nn.functional.binary_cross_entropy(toxic_probs_good, good_targets, reduction='sum')
        
        # Only add threshold learning after Bradley-Terry has converged
        # This prevents interference with the main learning objective
        if threshold is not None:
            # Add threshold learning with more aggressive coefficient
            # Simple threshold-based loss that doesn't interfere with θ learning
            # Just encourage threshold to be between toxic and non-toxic probabilities
            toxic_mean = torch.mean(toxic_probs_bad)
            nontoxic_mean = torch.mean(toxic_probs_good)
            target_threshold = (toxic_mean + nontoxic_mean) / 2.0
            
            # Much more aggressive threshold learning for faster convergence
            # Use adaptive coefficient based on how far the threshold is from target
            threshold_distance = torch.abs(threshold - target_threshold)
            adaptive_coefficient = torch.clamp(0.2 + 1.0 * threshold_distance, min=0.2, max=5.0)
            threshold_loss = adaptive_coefficient * (threshold - target_threshold).pow(2)
            
            # Return Bradley-Terry loss with stronger threshold influence
            return bt_loss + threshold_loss
        
        return bt_loss
    
    # return bt_loss
    return alpha * log_bt


