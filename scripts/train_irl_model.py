#!/usr/bin/env python3
"""
Train IRL reward model using Variational Inference.

This script provides a clean interface for training IRL reward models
using the re_irl.py implementation with Variational Inference.

Usage examples:

# Using default config
python scripts/train_irl_model.py

# Override model and dataset paths
python scripts/train_irl_model.py model.base_model_name=EleutherAI/pythia-70m dataset.original_dataset_path=datasets/toxic.json

# Override training parameters
python scripts/train_irl_model.py training.n_steps=5000 training.learning_rate=0.02

# Run all models automatically
python scripts/train_irl_model.py model.run_all=true
"""

import hydra
import sys
import os
from pathlib import Path
from omegaconf import DictConfig

# Add the parent directory to the path so we can import from irl_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from irl_pipeline.irl.re_irl import main as re_irl_main
from irl_pipeline.irl.re_irl import get_dataset_paths, _auto_fill_model_params

@hydra.main(config_path="../configs", config_name="full_pipeline.yaml", version_base=None)
def main(cfg: DictConfig):
    import glob

    _auto_fill_model_params(cfg)

    print("🚀 Starting IRL model training...")
    print(f"Base Model: {cfg.model.base_model_name}")
    print(f"Hidden Size: {cfg.model.hidden_size}")
    print(f"Training Steps: {cfg.training.n_steps}")
    print(f"Learning Rate: {cfg.training.learning_rate}")
    print(f"dataset train samples: {cfg.dataset.train_samples}")

    # Figure out detox name (mirror your existing logic)
    chosen = next((m for m in cfg.model.models if m.get("name") == cfg.model.base_model_name), None)
    detox_name = chosen["detox_name"] if chosen else cfg.model.get("detox_name", cfg.model.base_model_name)

    # Pull train_samples from config (e.g., dataset.yaml)
    n_train = cfg.dataset.get("train_samples", None)
    cache_dir = cfg.dataset.get("cache_dir", "datasets")

    # Resolve actual dataset files via the single shared helper
    paths = get_dataset_paths(
        model_name=cfg.model.base_model_name,
        detox_name=detox_name,
        cache_dir=cache_dir,
        train_samples=n_train,
    )

    # Write resolved paths back into cfg so re_irl_main uses them
    cfg.dataset.original_dataset_path         = paths["original_dataset_path"]
    cfg.dataset.detoxified_dataset_path       = paths["detoxified_dataset_path"]
    cfg.dataset.sorted_toxic_dataset_path     = paths["sorted_toxic_dataset_path"]
    cfg.dataset.sorted_non_toxic_dataset_path = paths["sorted_non_toxic_dataset_path"]

    # Print exactly what will be used
    print("📁 Using dataset files:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

    # Existence check (support wildcard messages if nothing matched)
    def _exists(p: str) -> bool:
        if "*" in p:
            return len(glob.glob(p)) > 0
        return os.path.exists(p)

    required = [
        cfg.dataset.original_dataset_path,
        cfg.dataset.detoxified_dataset_path,
        cfg.dataset.sorted_toxic_dataset_path,
        cfg.dataset.sorted_non_toxic_dataset_path,
    ]
    missing = [p for p in required if not _exists(p)]
    if missing:
        print("❌ Missing required dataset files:")
        for p in missing:
            print(f"  - {p}")
        print("\n💡 Please run dataset generation first:")
        print("   python scripts/generate_datasets.py")
        raise SystemExit(1)

    print("✅ All required datasets found")

    # Hand off to the core IRL runner
    try:
        re_irl_main(cfg)
        print("\n🎉 IRL model training completed successfully!")
    except Exception as e:
        print(f"\n❌ IRL model training failed: {e}")
        raise

if __name__ == "__main__":
    main()
