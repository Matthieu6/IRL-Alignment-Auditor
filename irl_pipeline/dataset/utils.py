"""
Dataset utilities for IRL pipeline.
Contains toxicity evaluation and other dataset processing functions.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_data(dataset_path: str) -> Tuple[List[str], List[Dict]]:
    """Load both texts and full data for analysis."""
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        texts = [d['output'] for d in data]
        return texts, data
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        if 'output' not in df.columns:
            raise ValueError('CSV must contain an "output" column')
        texts = df['output'].tolist()
        return texts, df.to_dict('records')
    else:
        raise ValueError('Unsupported dataset format: ' + dataset_path)


def load_texts(dataset_path: str) -> List[str]:
    """Load only the generated texts (for backward compatibility)."""
    texts, _ = load_data(dataset_path)
    return texts


def prepare_texts_for_evaluation(data: List[Dict], eval_target: str) -> List[str]:
    """Prepare texts based on what should be evaluated."""
    if eval_target == "output":
        return [d['output'] for d in data]
    elif eval_target == "prompt":
        return [d['prompt'] for d in data]
    elif eval_target == "prompt+output":
        return [d['prompt'] + " " + d['output'] for d in data]
    else:
        raise ValueError(f"Invalid eval_target: {eval_target}. Must be 'output', 'prompt', or 'prompt+output'")


def compute_scores(texts: List[str], model_name: str, batch_size: int = 8, device: str = None) -> np.ndarray:
    """Compute toxicity scores for a list of texts using the specified model."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the probability of the positive class (toxic)
            probs = torch.softmax(outputs.logits, dim=-1)
            if probs.shape[1] == 2:  # Binary classification
                batch_scores = probs[:, 1].cpu().numpy()  # Probability of positive class
            else:  # Multi-class, assume last class is toxic
                batch_scores = probs[:, -1].cpu().numpy()
            scores.extend(batch_scores)
    
    return np.array(scores)


def analyze_toxicity_distribution(scores: np.ndarray, title: str = "Toxicity Distribution") -> Dict:
    """Analyze the distribution of toxicity scores."""
    analysis = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
        "high_toxicity_count": int(np.sum(scores > 0.8)),
        "high_toxicity_pct": float(np.mean(scores > 0.8) * 100),
        "very_high_toxicity_count": int(np.sum(scores > 0.9)),
        "very_high_toxicity_pct": float(np.mean(scores > 0.9) * 100),
    }
    
    print(f"\n--- {title} ---")
    print(f"Mean: {analysis['mean']:.4f}")
    print(f"Std: {analysis['std']:.4f}")
    print(f"Min: {analysis['min']:.4f}")
    print(f"Max: {analysis['max']:.4f}")
    print(f"Median: {analysis['median']:.4f}")
    print(f"High toxicity (>0.8): {analysis['high_toxicity_count']} ({analysis['high_toxicity_pct']:.2f}%)")
    print(f"Very high toxicity (>0.9): {analysis['very_high_toxicity_count']} ({analysis['very_high_toxicity_pct']:.2f}%)")
    
    return analysis

