# reward_computer.py (minimal used subset)

import torch
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModel


class SimplifiedIRLRewardComputer:
    """
    Minimal IRL feature extractor used by the pipeline.

    What it does:
      - Loads a base HF model and tokenizer.
      - Extracts mean-pooled hidden states as features (optionally applies whitening/PCA).

    Notes:
      - Extra constructor args are accepted and ignored to remain backward-compatible
        with existing call sites (likelihood_type, normalization_strategy, etc.).
      - Theta samples are not used here (scoring happens elsewhere); accepted only
        to keep the original signature compatible.
    """
    
    def __init__(
        self,
        artifact_name: str = None,                 # ignored (kept for compatibility)
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        likelihood_type: str = "bradley_terry",   # ignored (kept for compatibility)
        normalization_strategy: str = "batch_zscore",  # ignored (kept for compatibility)
        n_posterior_samples: int = 100,           # ignored (kept for compatibility)
        device: str = "auto",
        theta_samples: Optional[torch.Tensor] = None,  # accepted but unused here
        whitening_stats: Optional[Dict] = None,   # REQUIRED - used in rlhf/utilities.py
        **kwargs,  # swallow any other legacy args
    ):
        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Keep for external checks/logging
        self.base_model_name = base_model_name
        self.whitening_stats = whitening_stats or {}

        # Load base model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            # Common for causal models
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)
        
        self.model = AutoModel.from_pretrained(base_model_name)
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def extract_features(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Mean-pool last hidden layer (masked) to produce one feature vector per text.
        Optionally applies whitening and linear PCA projection if provided.
            
        Returns:
            torch.Tensor of shape (len(texts), hidden_dim or projected_dim)
        """
        features = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [B, T, H]

            # Masked mean pool
            attn = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            summed = (hidden * attn).sum(dim=1)                              # [B, H]
            counts = attn.sum(dim=1).clamp_min(1.0)                          # [B, 1]
            pooled = summed / counts                                         # [B, H]

            # Optional whitening / PCA
            ws = self.whitening_stats
            if ws and not ws.get("disable_whitening", False):
                mean = ws.get("mean", None)
                std  = ws.get("std", None)
                if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                    # ensure device matches
                    if pooled.device != mean.device:
                        mean = mean.to(pooled.device)
                        std  = std.to(pooled.device)
                    pooled = (pooled - mean) / (std + 1e-8)

                pca = ws.get("pca_components", None)
                if isinstance(pca, torch.Tensor):
                    if pooled.device != pca.device:
                        pca = pca.to(pooled.device)
                    pooled = pooled @ pca

            features.append(pooled)

        return torch.cat(features, dim=0)

