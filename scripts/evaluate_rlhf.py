#!/usr/bin/env python3
"""
RLHF Model Evaluation Script

Evaluates trained RLHF models on toxicity reduction using real-toxicity-prompts.
Supports both HuggingFace repos and local checkpoints.
"""

import hydra
from omegaconf import DictConfig
from irl_pipeline.rlhf.evaluate import evaluate_rlhf


@hydra.main(config_path="../configs", config_name="full_pipeline.yaml", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    
    print("="*60)
    print("Starting RLHF Model Evaluation")
    print("="*60)
    
    # Run evaluation
    evaluate_rlhf(cfg)
    
    print("="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

