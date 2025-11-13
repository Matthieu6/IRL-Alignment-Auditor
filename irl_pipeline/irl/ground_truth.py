#!/usr/bin/env python3
"""
Generate Ground Truth Posterior Distribution

This module creates a deterministic ground truth posterior distribution using 
reward models or toxicity classifiers to compare against IRL posteriors.

Supports two types of models:
1. Classification models (e.g., s-nlp/roberta_toxicity_classifier):
   - Multi-label output
   - Converts probabilities to rewards: r*(o) = logit(P_non_toxic(o))
   - Higher reward = less toxic = better

2. Regression models (e.g., Ray2333/gpt2-large-helpful-reward_model):
   - Single-label output  
   - Uses raw model logits as rewards: r*(o) = model(o)
   - Higher reward = more helpful/harmless = better

Method:
1. Load training data (original + detoxified datasets)
2. Compute reward scores for all texts
3. Create Bradley-Terry ground truth: P_GT(o+ > o-) = σ(Δr*) 
   where Δr* = r*(o+) - r*(o-)
4. Evaluate accuracy on sorted datasets

Usage:
    gt = GroundTruthGenerator(classifier_name="Ray2333/gpt2-large-helpful-reward_model")
    rewards = gt.compute_ground_truth_rewards(texts)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Imports are now handled by the package structure


class GroundTruthGenerator:
    """Ground truth generator supporting both classification and regression reward models.
    
    Supports two types of models:
    1. Classification models (e.g., toxicity classifiers): Multi-label output, uses probability-to-logit conversion
    2. Regression models (e.g., GPT2 reward models): Single-label output, uses raw logits as rewards
    """
    
    def __init__(self, classifier_name: str = "s-nlp/roberta_toxicity_classifier", device: str = None):
        self.classifier_name = classifier_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🤖 Loading ground truth classifier: {classifier_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load classifier
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_name)
        # Fix for GPT2-based models that don't have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"   ⚙️  Set pad_token = eos_token (GPT2 tokenizer fix)")
        self.model = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(self.device)
        self.model.eval()
        
        # Get label mapping
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        num_labels = len(id2label)
        print(f"   Available labels: {id2label}")
        print(f"   Number of labels: {num_labels}")
        
        # Determine model type based on number of labels
        if num_labels == 1:
            # Regression model (e.g., GPT2 reward model)
            # Single output where higher = better (more helpful/harmless)
            self.model_type = "regression"
            print(f"   🎯 Detected REGRESSION model: using raw logits as rewards")
            print(f"      Higher scores = more helpful/harmless/better")
        else:
            # Classification model (e.g., toxicity classifier)
            # Multiple labels, need to convert probabilities to rewards
            self.model_type = "classification"
            # For s-nlp/roberta_toxicity_classifier: 0=non-toxic, 1=toxic
            self.nontoxic_idx = 0
            print(f"   🎯 Detected CLASSIFICATION model: using probability-to-logit conversion")
            print(f"      Using index {self.nontoxic_idx} for non-toxic probability")
    
    # def load_dataset(self, path: str) -> List[Dict]:
    #     """Load dataset from JSON file."""
    #     print(f"📁 Loading dataset: {path}")
    #     with open(path, 'r') as f:
    #         data = json.load(f)
    #     print(f"   Loaded {len(data)} samples")
    #     return data
    
    def compute_toxicity_scores(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute probability scores for classification models.
        
        For classification models (toxicity classifiers), returns P(non-toxic).
        For regression models, this method is not used.
        """
        print(f"🔍 Computing probability scores for {len(texts)} texts...")
        
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(**encoded).logits
                
                if self.model_type == "classification":
                    # For classification: get probability of non-toxic class
                    probs = torch.softmax(logits, dim=-1)[:, self.nontoxic_idx].cpu().numpy()
                    scores.extend(probs.tolist())
                else:
                    # This shouldn't be called for regression models
                    raise ValueError("compute_toxicity_scores should not be called for regression models")
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {i + len(batch)}/{len(texts)} texts...")
        
        scores = np.array(scores)
        print(f"✅ Computed probability scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        return scores
    
    def compute_ground_truth_rewards(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Compute ground truth rewards based on model type.
        
        For CLASSIFICATION models (e.g., toxicity classifiers):
            r*(o) = logit(P_non_toxic(o))
            Higher values = less toxic = better content
            
        For REGRESSION models (e.g., GPT2 reward models):
            r*(o) = raw logits from model
            Higher values = more helpful/harmless = better content
        
        In both cases: Higher reward = Better content
        """
        print("🎯 Computing ground truth rewards...")
        
        if self.model_type == "classification":
            # Classification model: convert probabilities to logits
            print("   Using CLASSIFICATION method: logit(P_non_toxic)")
            
            # Get non-toxic probabilities 
            nontoxic_probs = self.compute_toxicity_scores(texts, batch_size)
            
            # Convert to logits: logit(p) = log(p / (1-p))
            # Clamp probabilities to avoid log(0)
            nontoxic_probs = np.clip(nontoxic_probs, 1e-7, 1 - 1e-7)
            logits = np.log(nontoxic_probs / (1 - nontoxic_probs))
            
            # Ground truth reward: r*(o) = logit(P_non_toxic(o))
            # Higher logit(P_non_toxic) = higher reward = less toxic = better content
            rewards = logits
            
        else:  # regression
            # Regression model: use raw logits directly as rewards
            print("   Using REGRESSION method: raw model logits")
            print(f"   Processing {len(texts)} texts...")
            
            rewards_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # Get raw logits (rewards)
                with torch.no_grad():
                    logits = self.model(**encoded).logits
                    # For single-label regression, logits shape is (batch_size, 1)
                    # Squeeze to get (batch_size,)
                    batch_rewards = logits.squeeze(-1).cpu().numpy()
                    rewards_list.extend(batch_rewards.tolist())
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"      Processed {i + len(batch)}/{len(texts)} texts...")
            
            rewards = np.array(rewards_list)
        
        print(f"   Ground truth rewards: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
        print(f"   ✅ Higher rewards = Better content")
        return rewards