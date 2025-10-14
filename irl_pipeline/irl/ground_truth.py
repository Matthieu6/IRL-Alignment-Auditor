#!/usr/bin/env python3
"""
Generate Ground Truth Posterior Distribution

This script creates a deterministic ground truth posterior distribution using the 
s-nlp RoBERTa toxicity classifier to compare against IRL posteriors.

Method:
1. Load training data (original + detoxified datasets)
2. Compute RoBERTa toxicity scores for all texts
3. Create Bradley-Terry ground truth: P_GT(o+ > o-) = σ(Δr*) 
   where Δr* = r*(o+) - r*(o-) and r*(o) = -logit(P_non_toxic(o))
4. Evaluate accuracy on sorted toxic/non-toxic test sets

Usage:
    python src/evaluate_posterior/generate_ground_truth.py
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
    """Simple ground truth generator using RoBERTa toxicity classifier."""
    
    def __init__(self, classifier_name: str = "s-nlp/roberta_toxicity_classifier", device: str = None):
        self.classifier_name = classifier_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🤖 Loading ground truth classifier: {classifier_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load classifier
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(self.device)
        self.model.eval()
        
        # Get label mapping
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"   Available labels: {id2label}")
        
        # For s-nlp/roberta_toxicity_classifier: 0=non-toxic, 1=toxic
        self.nontoxic_idx = 0
        print(f"   Using index {self.nontoxic_idx} for non-toxic probability")
    
    # def load_dataset(self, path: str) -> List[Dict]:
    #     """Load dataset from JSON file."""
    #     print(f"📁 Loading dataset: {path}")
    #     with open(path, 'r') as f:
    #         data = json.load(f)
    #     print(f"   Loaded {len(data)} samples")
    #     return data
    
    def compute_toxicity_scores(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute toxicity probability scores using RoBERTa classifier."""
        print(f"🔍 Computing toxicity scores for {len(texts)} texts...")
        
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
                probs = torch.softmax(logits, dim=-1)[:, self.nontoxic_idx].cpu().numpy()
            
            scores.extend(probs.tolist())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {i + len(batch)}/{len(texts)} texts...")
        
        scores = np.array(scores)
        print(f"✅ Computed toxicity scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        return scores
    
    def compute_ground_truth_rewards(self, texts: List[str]) -> np.ndarray:
        """
        Compute ground truth rewards: r*(o) = logit(P_non_toxic(o))
        Higher values = less toxic (better content)
        """
        print("🎯 Computing ground truth rewards...")
        
        # Get non-toxic probabilities 
        nontoxic_probs = self.compute_toxicity_scores(texts)
        
        # Convert to logits: logit(p) = log(p / (1-p))
        # Clamp probabilities to avoid log(0)
        nontoxic_probs = np.clip(nontoxic_probs, 1e-7, 1 - 1e-7)
        logits = np.log(nontoxic_probs / (1 - nontoxic_probs))
        
        # Ground truth reward: r*(o) = logit(P_non_toxic(o))
        # Higher logit(P_non_toxic) = higher reward = less toxic = better content
        rewards = logits
        
        print(f"   Ground truth rewards: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
        return rewards