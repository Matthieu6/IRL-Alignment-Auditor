"""
IRL Utilities for RLHF Training - SIMPLIFIED VERSION

This module provides a simplified interface for using IRL-trained reward models
in RLHF training. The key insight is that we just need:
1. Sample θ from VI posterior (using vi_utils)
2. Extract features using base LLM + mean pooling
3. Compute θ^T * features = reward

For unified RLHF training, you can use either:
- Traditional rewards: HuggingFace classifiers (s-nlp/roberta_toxicity_classifier)
- IRL rewards: VI posterior samples (using SimplifiedIRLRewardComputer)
"""

import os
import torch
import numpy as np
from typing import List, Union, Tuple, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimplifiedIRLRewardComputer:
    """
    Simplified IRL reward computer that directly implements the core pipeline:
    1. Sample θ from VI posterior OR use provided theta samples
    2. Use base LLM + mean pooling for features
    3. Compute θ^T * features
    """
    
    def __init__(
        self,
        artifact_name: str = None,
        base_model_name: str = "gpt2",
        likelihood_type: str = "bradley_terry",
        normalization_strategy: str = "batch_zscore",
        n_posterior_samples: int = 100,
        device: str = "auto",
        # NEW: Support for local theta samples
        theta_samples: torch.Tensor = None,
        whitening_stats: dict = None,
        # NEW: Temperature parameters for likelihood functions
        use_learnable_temperature: bool = False,
        fixed_temperature: float = 1.0,
        learned_temperature: float = None,
        # NEW: Feature normalization control
        enable_feature_normalization: bool = True
    ):
        """
        Initialize the simplified IRL reward computer.
        
        Args:
            artifact_name: WandB artifact name for the VI posterior (can be None if theta_samples provided)
            base_model_name: Base LLM to use for feature extraction
            likelihood_type: Type of likelihood function
            normalization_strategy: How to normalize rewards
            n_posterior_samples: Number of theta samples to use
            device: Device to run on
            theta_samples: Optional pre-computed theta samples (bypasses WandB download)
            whitening_stats: Optional whitening statistics (bypasses WandB download)
            use_learnable_temperature: Whether to use learnable temperature (not supported in this simplified version)
            fixed_temperature: Fixed temperature value for likelihood functions
            learned_temperature: Learned temperature value from VI training (takes precedence if provided)
            enable_feature_normalization: Whether features were divided by their max value during training
                (whitening always uses the provided statistics regardless)
        """
        # from irl_pipeline.variational_inference.vi_utils import (
        #     download_wandb_posterior, 
        #     load_variational_posterior,
        #     sample_from_saved_posterior,
        #     create_model_function_from_config
        # )
        
        self.base_model_name = base_model_name
        self.likelihood_type = likelihood_type
        self.normalization_strategy = normalization_strategy
        self.n_posterior_samples = n_posterior_samples
        self.enable_feature_normalization = enable_feature_normalization
        
        # Temperature parameters
        self.use_learnable_temperature = use_learnable_temperature
        self.fixed_temperature = fixed_temperature
        self.learned_temperature = learned_temperature
        
        # Determine which temperature to use
        if learned_temperature is not None:
            self.temperature = float(learned_temperature)
            self.temperature_source = "learned"
        else:
            self.temperature = float(fixed_temperature)
            self.temperature_source = "fixed"
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"🎯 Initializing Simplified IRL Reward Computer")
        print(f"   🤖 Base model: {base_model_name}")
        print(f"   🎲 Likelihood: {likelihood_type}")
        print(f"   📊 Normalization: {normalization_strategy}")
        print(f"   🔢 Posterior samples: {n_posterior_samples}")
        print(f"   🌡️  Temperature: {self.temperature_source}={self.temperature:.4f}")
        
        # Check if we have local theta samples
        if theta_samples is not None:
            print(f"📦 Using provided theta samples (bypassing WandB download)")
            
            # Use provided theta samples
            self.theta_samples = theta_samples.to(self.device)
            if self.theta_samples.dim() == 1:
                self.theta_samples = self.theta_samples.unsqueeze(0)
            
            # Take only the requested number of samples
            if len(self.theta_samples) > n_posterior_samples:
                indices = torch.randperm(len(self.theta_samples))[:n_posterior_samples]
                self.theta_samples = self.theta_samples[indices]
            
            print(f"   ✅ Using theta shape: {self.theta_samples.shape}")
            print(f"   📊 Theta statistics: mean={self.theta_samples.mean():.4f}, std={self.theta_samples.std():.4f}")
            
            # Use provided whitening stats
            self.whitening_stats = whitening_stats
            if self.whitening_stats:
                # Check if this is a disable whitening flag
                if self.whitening_stats.get('disable_whitening', False):
                    print("⚪ Whitening explicitly disabled.")
                else:
                    print("⚪ Using provided whitening stats.")
                    # Keep only essential verification
                    print(f"   Original dim: {self.whitening_stats.get('original_dim', '?')} → Whitened dim: {self.whitening_stats.get('whitened_dim', '?')}")
                    
                    # Move whitening stats to correct device (only if they exist)
                    if 'mean' in self.whitening_stats and self.whitening_stats['mean'] is not None:
                        self.whitening_stats['mean'] = self.whitening_stats['mean'].to(self.device)
                    if 'std' in self.whitening_stats and self.whitening_stats['std'] is not None:
                        self.whitening_stats['std'] = self.whitening_stats['std'].to(self.device)
                    if 'pca_components' in self.whitening_stats and self.whitening_stats['pca_components'] is not None:
                        self.whitening_stats['pca_components'] = self.whitening_stats['pca_components'].to(self.device)
            else:
                print("⚠️  No whitening stats provided.")
            
            # Skip WandB download
            self.artifact_name = None
            self.posterior_path = None
            self.posterior_state = None
            
        else:
            # Original WandB artifact approach
            if artifact_name is None:
                raise ValueError("Either artifact_name or theta_samples must be provided")
            
            self.artifact_name = artifact_name
            print(f"   📦 Artifact: {artifact_name}")
            
            # Step 1: Download and load VI posterior
            print(f"📥 Downloading posterior from WandB...")
            self.posterior_path = download_wandb_posterior(artifact_name)
            self.posterior_state = load_variational_posterior(self.posterior_path, self.device)
            
            # Step 2: Extract temperature information from posterior state
            if "learned_temperature" in self.posterior_state:
                self.learned_temperature = self.posterior_state["learned_temperature"]
                self.temperature = self.learned_temperature
                self.temperature_source = "learned"
                print(f"   🌡️  Using learned temperature from posterior: T = {self.temperature:.4f}")
            elif "fixed_temperature" in self.posterior_state:
                self.fixed_temperature = self.posterior_state["fixed_temperature"]
                self.temperature = self.fixed_temperature
                self.temperature_source = "fixed"
                print(f"   🌡️  Using fixed temperature from posterior: T = {self.temperature:.4f}")
            else:
                print(f"   ⚠️  No temperature found in posterior, using provided fixed temperature: T = {self.temperature:.4f}")
            
            # Step 3: Sample theta from posterior
            print(f"🎲 Sampling {n_posterior_samples} theta samples from posterior...")
            
            # Create model function for AutoGuides
            model_fn = None
            if self.posterior_state["guide_type"] in ["multivariate", "iaf"]:
                model_fn = create_model_function_from_config(self.posterior_state["model_config"])
            
            self.theta_samples = sample_from_saved_posterior(
                posterior_path=self.posterior_path,
                n_samples=n_posterior_samples,
                device=self.device,
                model_fn=model_fn
            )
            
            # Step 4: Get whitening stats if available
            self.whitening_stats = self.posterior_state.get("whitening_stats")
            if self.whitening_stats:
                print("⚪ Using whitening stats from posterior.")
                # Move whitening stats to the correct device
                self.whitening_stats = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) and v is not None else v
                    for k, v in self.whitening_stats.items()
                }
            else:
                print("⚠️  No whitening stats found in posterior.")
        
        # Step 5: Load base model for feature extraction
        print(f"🤖 Loading base model: {base_model_name}")
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModel.from_pretrained(base_model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # Step 6: Validate theta dimensions match model hidden size
        if hasattr(self, 'theta_samples') and self.theta_samples is not None:
            model_hidden_size = self.model.config.hidden_size
            theta_dim = self.theta_samples.shape[-1]
            
            print(f"   🔍 Model hidden size: {model_hidden_size}")
            print(f"   🔍 Theta dimension: {theta_dim}")
            
            if theta_dim != model_hidden_size:
                raise ValueError(
                    f"❌ Dimension mismatch! Theta samples have {theta_dim} dimensions, "
                    f"but model {base_model_name} has {model_hidden_size} hidden dimensions. "
                    f"This suggests the theta samples were trained with a different model. "
                    f"Please ensure you're using the same model for training and evaluation."
                )
            else:
                print(f"   ✅ Dimension validation passed: {theta_dim} == {model_hidden_size}")
        
        print(f"✅ Simplified IRL Reward Computer initialized successfully!")
    
    def extract_features(self, texts: List[str], max_length: int = 512, batch_size: int = 32) -> torch.Tensor:
        """
        Extract features using base LLM + mean pooling with batching to prevent GPU OOM.
        This is the CORRECT way to get features for IRL rewards.
        
        Args:
            texts: List of texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            Feature tensor of shape (len(texts), hidden_dim)
        """
        all_features = []
        
        # Process in batches to prevent GPU memory issues
        # Auto-tune batch size for large models to avoid OOM
        effective_batch = batch_size
        if hasattr(self, "hidden_size") and self.hidden_size and self.hidden_size >= 2000:
            effective_batch = min(batch_size, 8)

        for i in range(0, len(texts), effective_batch):
            batch_texts = texts[i:i + effective_batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get hidden states from base LLM
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Apply mask and compute mean
            masked_hidden = hidden_states * expanded_mask
            sum_hidden = torch.sum(masked_hidden, dim=1)
            
            # Avoid division by zero
            token_count = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1.0)
            pooled_features = sum_hidden / token_count  # [batch_size, hidden_dim]
            
            # Apply whitening if available and not explicitly disabled
            # Check for disable_whitening flag (used by simple_approach)
            if (
                self.whitening_stats is not None
                and self.whitening_stats.get('mean') is not None
                and self.whitening_stats.get('std') is not None
                and not self.whitening_stats.get('disable_whitening', False)
            ):
                # Apply whitening transformation
                mean = self.whitening_stats['mean']
                std = self.whitening_stats['std']
                # Debug once per extraction call
                if i == 0 and not hasattr(self, '_whitening_debug_shown'):
                    try:
                        print(f"   🔧 Whitening apply: mean_norm={float(torch.norm(mean)):.4f}, std_mean={float(torch.mean(std)):.4f}")
                        self._whitening_debug_shown = True
                    except Exception:
                        pass

                # Ensure tensors are on the same device
                if pooled_features.device != mean.device:
                    pooled_features = pooled_features.to(mean.device)

                # Apply whitening: (x - mean) / std
                pooled_features = (pooled_features - mean) / (std + 1e-8)  # Add small epsilon for numerical stability

                # Apply PCA if available
                if 'pca_components' in self.whitening_stats and self.whitening_stats['pca_components'] is not None:
                    pca_components = self.whitening_stats['pca_components']
                    if pooled_features.device != pca_components.device:
                        pooled_features = pooled_features.to(pca_components.device)
                    pooled_features = torch.matmul(pooled_features, pca_components)
            elif self.whitening_stats is not None and self.whitening_stats.get('disable_whitening', False):
                if i == 0:
                    print(f"   ✓ Whitening disabled (simple_approach mode)")
            elif self.whitening_stats is not None:
                print(f"   ⚠️  Whitening stats available but mean/std are None")
            else:
                if i == 0:
                    print(f"   ℹ️  No whitening applied (no stats available)")
            
            all_features.append(pooled_features)
            # Proactively free CUDA cache on large models
            if (i // max(1, effective_batch)) % 4 == 0 and getattr(self.device, "type", str(self.device)).startswith("cuda"):
                torch.cuda.empty_cache()
            
            # Clear GPU cache periodically to prevent memory buildup
            if i % (effective_batch * 4) == 0 and getattr(self.device, "type", str(self.device)).startswith("cuda"):
                torch.cuda.empty_cache()
        
        return torch.cat(all_features, dim=0)
    
    # def apply_likelihood_function(self, theta: torch.Tensor, features: torch.Tensor, use_raw_score: bool = False) -> torch.Tensor:
    #     """
    #     Apply the specified likelihood function to compute rewards.
    #     
    #     Args:
    #         theta: Theta parameter vector (hidden_dim,)
    #         features: Feature vectors (batch_size, hidden_dim)
    #         use_raw_score: If True, return the raw linear score with temperature scaling.
    #         
    #     Returns:
    #         Reward values (batch_size,)
    #     """
    #     # Ensure both tensors are on the same device
    #     if theta.device != features.device:
    #         theta = theta.to(features.device)
    #     
    #     # Linear combination: θ^T * features
    #     linear_scores = torch.matmul(features, theta)  # [batch_size]
    #     
    #     # Return raw score if requested (with temperature scaling for consistency)
    #     if use_raw_score:
    #         # Debug once per session on first call: temperature source/value
    #         if not hasattr(self, "_dbg_temp_printed"):
    #             print(f"   🌡️  apply_likelihood: raw mode, temperature={self.temperature:.4f} ({self.temperature_source})")
    #             self._dbg_temp_printed = True
    #         return linear_scores / self.temperature

    #     # Get temperature value (learned or fixed)
    #     temperature = self.temperature
    #     if not hasattr(self, "_dbg_temp_printed2"):
    #         print(f"   🌡️  apply_likelihood: prob mode, temperature={temperature:.4f} ({self.temperature_source})")
    #         self._dbg_temp_printed2 = True
    #     
    #     if self.likelihood_type == "bradley_terry":
    #         # Bradley-Terry: P(y=1|x,θ) = σ(θᵀx / T)
    #         # Apply temperature scaling to logits
    #         logits = linear_scores / temperature
    #         logits = torch.clamp(logits, min=-20.0, max=20.0)  # Prevent overflow
    #         return torch.sigmoid(logits)
    #     
    #     elif self.likelihood_type == "bradley_terry_anchor":
    #         # Bradley-Terry anchor: P(y=1|x,θ) = σ(θᵀx / T)
    #         # For single text evaluation, treat as standard Bradley-Terry
    #         logits = linear_scores / temperature
    #         logits = torch.clamp(logits, min=-20.0, max=20.0)  # Prevent overflow
    #         return torch.sigmoid(logits)
    #     
    #     elif self.likelihood_type == "bayes_svm_simplified":
    #         # Simplified Bayesian SVM likelihood
    #         # Apply temperature scaling to scores
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return torch.sigmoid(scores)
    #     
    #     elif self.likelihood_type == "exponential":
    #         # Standard IRL exponential likelihood
    #         # Apply temperature scaling to scores
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return torch.exp(scores)
    #     
    #     elif self.likelihood_type == "hinge":
    #         # Hinge-based likelihood: max(0, 1 + θᵀx / T)
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return torch.relu(1.0 + scores)
    #     
    #     elif self.likelihood_type == "squared":
    #         # Squared difference likelihood: (θᵀx / T)²
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return scores ** 2
    #     
    #     elif self.likelihood_type == "bayes_svm":
    #         # Bayesian SVM likelihood
    #         # Apply temperature scaling to scores
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return torch.sigmoid(scores)
    #     
    #     elif self.likelihood_type == "bayesian_hinge":
    #         # Bayesian hinge likelihood
    #         # Apply temperature scaling to scores
    #         scores = linear_scores / temperature
    #         scores = torch.clamp(scores, min=-10.0, max=10.0)  # Prevent overflow
    #         return torch.sigmoid(scores)
    #     
    #     else:
    #         raise ValueError(f"Unknown likelihood type: {self.likelihood_type}")
    
    def compute_rewards(
        self,
        texts: List[str],
        max_length: int = 512,
        return_raw_values: bool = False,
        use_raw_score: bool = False,
        return_variance: bool = False,  # NEW: Add return_variance parameter
        batch_size: int = 32,  # NEW: Add batch_size parameter
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute IRL rewards using the simplified pipeline with batching to prevent GPU OOM.
        
        Args:
            texts: List of texts to compute rewards for
            max_length: Maximum sequence length
            return_raw_values: Whether to return raw values
            use_raw_score: If True, use the raw linear score with temperature scaling as the reward.
            return_variance: If True, return both mean and standard deviation of rewards across posterior samples.
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            Normalized rewards, optionally with raw values and/or standard deviation
        """
        # Step 1: Extract features using base LLM + mean pooling with batching
        features = self.extract_features(texts, max_length, batch_size)  # [batch_size, hidden_dim]
        
        # Step 2: EFFICIENT VECTORIZED COMPUTATION for all theta samples
        # Shape: features [batch_size, hidden_dim], theta_samples [n_samples, hidden_dim]
        # Result: [n_samples, batch_size] via broadcasting
        # Ensure theta_samples are on the same device as features
        theta_samples_device = self.theta_samples.to(features.device)
        linear_scores = torch.matmul(features, theta_samples_device.T)  # [batch_size, n_samples]
        linear_scores = linear_scores.T  # [n_samples, batch_size] for consistency
        
        if use_raw_score:
            # For raw scores, still apply temperature scaling to maintain consistency
            # This ensures learned temperatures are respected even in "raw" mode
            all_rewards = linear_scores / self.temperature  # [n_samples, batch_size]
            raw_rewards = all_rewards.mean(dim=0)  # [batch_size]
        else:
            # Apply likelihood function to all samples at once
            temperature = self.temperature
            
            if self.likelihood_type == "exponential":
                all_rewards = torch.exp(linear_scores / temperature)
            elif self.likelihood_type == "bradley_terry":
                # Bradley-Terry likelihood: σ(score/T)
                scaled_scores = linear_scores / temperature
                scaled_scores = torch.clamp(scaled_scores, min=-10.0, max=10.0)
                all_rewards = torch.sigmoid(scaled_scores)
            elif self.likelihood_type == "bradley_terry_anchor":
                # Same as bradley_terry for this implementation
                scaled_scores = linear_scores / temperature
                scaled_scores = torch.clamp(scaled_scores, min=-10.0, max=10.0)
                all_rewards = torch.sigmoid(scaled_scores)
            elif self.likelihood_type == "bayes_svm":
                scaled_scores = linear_scores / temperature
                scaled_scores = torch.clamp(scaled_scores, min=-10.0, max=10.0)
                all_rewards = torch.sigmoid(scaled_scores)
            elif self.likelihood_type == "bayesian_hinge":
                scaled_scores = linear_scores / temperature
                scaled_scores = torch.clamp(scaled_scores, min=-10.0, max=10.0)
                all_rewards = torch.sigmoid(scaled_scores)
            else:
                raise ValueError(f"Unknown likelihood type: {self.likelihood_type}")
            
            # Step 3: Average across posterior samples
            raw_rewards = all_rewards.mean(dim=0)  # [batch_size]

        # NEW: Compute standard deviation across posterior samples if requested
        if return_variance:
            reward_std = all_rewards.std(dim=0)  # [batch_size]
        
        # Step 4: Handle NaN/Inf values
        if torch.isnan(raw_rewards).any() or torch.isinf(raw_rewards).any():
            print("Warning: NaN or Inf values in IRL reward computation, replacing with zeros")
            raw_rewards = torch.where(
                torch.isnan(raw_rewards) | torch.isinf(raw_rewards),
                torch.zeros_like(raw_rewards),
                raw_rewards
            )
        
        # Step 5: Apply normalization
        normalized_rewards = self._normalize_rewards(raw_rewards)
        
        # NEW: Return mean and standard deviation if requested
        if return_variance:
            if return_raw_values:
                return normalized_rewards, raw_rewards, reward_std
            else:
                return normalized_rewards, reward_std
        else:
            if return_raw_values:
                return normalized_rewards, raw_rewards
            else:
                return normalized_rewards
    
    # def compute_raw_scores_no_temperature(self, texts: List[str], max_length: int = 512, batch_size: int = 32) -> torch.Tensor:
    #     """
    #     Compute truly raw scores without any temperature scaling.
    #     For temperature analysis only.
    #     """
    #     # Extract features
    #     features = self.extract_features(texts, max_length, batch_size)
    #     
    #     # Compute linear scores without temperature
    #     theta_samples_device = self.theta_samples.to(features.device)
    #     linear_scores = torch.matmul(features, theta_samples_device.T)  # [batch_size, n_samples]
    #     linear_scores = linear_scores.T  # [n_samples, batch_size]
    #     
    #     # Return raw scores without any temperature scaling
    #     return linear_scores.mean(dim=0)  # [batch_size]
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply normalization strategy to rewards."""
        if self.normalization_strategy == "none":
            return rewards
        
        elif self.normalization_strategy == "batch_zscore":
            # Z-score normalization within the batch
            if len(rewards) > 1:
                mean_reward = rewards.mean()
                std_reward = rewards.std()
                
                # Avoid division by zero
                if std_reward > 1e-8:
                    return (rewards - mean_reward) / std_reward
                else:
                    return rewards - mean_reward
            else:
                return rewards
        
        elif self.normalization_strategy == "sigmoid":
            # Sigmoid normalization to [0, 1]
            return torch.sigmoid(rewards)
        
        elif self.normalization_strategy == "tanh":
            # Tanh normalization to [-1, 1]
            return torch.tanh(rewards)
        
        else:
            raise ValueError(f"Unknown normalization strategy: {self.normalization_strategy}")


# def create_simplified_irl_reward_computer_from_samples(
#     theta_samples: torch.Tensor,
#     base_model_name: str = "gpt2",
#     likelihood_type: str = "bradley_terry",
#     normalization_strategy: str = "batch_zscore",
#     whitening_stats: dict = None,
#     device: str = "auto",
#     # NEW: Temperature parameters
#     use_learnable_temperature: bool = False,
#     fixed_temperature: float = 1.0,
#     learned_temperature: float = None,
#     # NEW: Feature normalization control
#     enable_feature_normalization: bool = True
# ) -> SimplifiedIRLRewardComputer:
#     """
#     Create a SimplifiedIRLRewardComputer from local theta samples (for evaluation).
#     
#     Args:
#         theta_samples: Pre-computed theta samples
#         base_model_name: Base LLM to use for feature extraction
#         likelihood_type: Type of likelihood function
#         normalization_strategy: How to normalize rewards
#         whitening_stats: Optional whitening statistics
#         device: Device to run on
#         use_learnable_temperature: Whether to use learnable temperature (not supported in this simplified version)
#         fixed_temperature: Fixed temperature value for likelihood functions
#         learned_temperature: Learned temperature value from VI training (takes precedence if provided)
#         enable_feature_normalization: Whether to enable feature normalization
#         
#     Returns:
#         Configured SimplifiedIRLRewardComputer instance
#     """
#     return SimplifiedIRLRewardComputer(
#         artifact_name=None,  # Skip WandB download
#         base_model_name=base_model_name,
#         likelihood_type=likelihood_type,
#         normalization_strategy=normalization_strategy,
#         n_posterior_samples=len(theta_samples),
#         device=device,
#         theta_samples=theta_samples,
#         whitening_stats=whitening_stats,
#         use_learnable_temperature=use_learnable_temperature,
#         fixed_temperature=fixed_temperature,
#         learned_temperature=learned_temperature,
#         enable_feature_normalization=enable_feature_normalization
#     )


# def create_simplified_irl_reward_computer(
#     artifact_name: str,
#     base_model_name: str = "gpt2",
#     likelihood_type: str = "bradley_terry",
#     normalization_strategy: str = "batch_zscore",
#     n_posterior_samples: int = 100,
#     device: str = "auto"
# ) -> SimplifiedIRLRewardComputer:
#     """
#     Create a simplified IRL reward computer from WandB artifact.
#     
#     Args:
#         artifact_name: WandB artifact name for the VI posterior
#         base_model_name: Base LLM to use for feature extraction
#         likelihood_type: Type of likelihood function
#         normalization_strategy: How to normalize rewards
#         n_posterior_samples: Number of theta samples to use
#         device: Device to run on
#         
#     Returns:
#         Configured SimplifiedIRLRewardComputer instance
#     """
#     return SimplifiedIRLRewardComputer(
#         artifact_name=artifact_name,
#         base_model_name=base_model_name,
#         likelihood_type=likelihood_type,
#         normalization_strategy=normalization_strategy,
#         n_posterior_samples=n_posterior_samples,
#         device=device
#     )


# # Likelihood function mappings for different IRL training approaches
# LIKELIHOOD_FUNCTIONS = {
#     "exponential": "Standard exponential likelihood (original IRL)",
#     "bradley_terry": "Bradley-Terry likelihood for preference learning",
#     "bradley_terry_anchor": "Bradley-Terry with anchor reference",
#     "hinge": "Hinge loss likelihood for max-margin learning",
#     "squared": "Squared difference likelihood",
#     "bayes_svm": "Bayesian SVM likelihood with auxiliary variables", 
#     "bayes_svm_simplified": "Simplified Bayesian SVM likelihood",
#     "bayesian_hinge": "Bayesian interpretation of hinge loss"
# }


# def get_likelihood_info(likelihood_type: str) -> str:
#     """Get information about a likelihood function."""
#     return LIKELIHOOD_FUNCTIONS.get(likelihood_type, "Unknown likelihood function")


# def list_available_likelihoods() -> Dict[str, str]:
#     """List all available likelihood functions."""
#     return LIKELIHOOD_FUNCTIONS.copy()