#!/usr/bin/env python3
"""
Simple wrapper for generating all datasets (toxic and non-toxic, train and test)
and automatically running toxicity evaluation.

Usage examples:

# Using default config
python scripts/generate_datasets.py

# Override models
python scripts/generate_datasets.py toxic_model=EleutherAI/gpt-neo-125m non_toxic_model=ybelkada/gpt-neo-125m-detox

# Override sample counts
python scripts/generate_datasets.py train_samples=1000 test_samples=300

# Override thresholds
python scripts/generate_datasets.py toxicity_threshold=0.95 non_toxic_threshold=0.5
"""

import hydra
import subprocess
import os
import glob
from omegaconf import DictConfig
from irl_pipeline.dataset.generator import generate_dataset


def remove_existing_model_files(toxic_model: str, non_toxic_model: str, cache_dir: str = "datasets"):
    """Remove existing dataset files for the specified models to ensure clean generation."""
    print(f"\n🧹 Cleaning existing files for models...")
    print(f"  Toxic model: {toxic_model}")
    print(f"  Non-toxic model: {non_toxic_model}")
    
    # Convert model names to safe format for file matching
    toxic_model_safe = toxic_model.replace('/', '_')
    non_toxic_model_safe = non_toxic_model.replace('/', '_')
    
    # Patterns to match existing files
    patterns = [
        f"{toxic_model_safe}_*_samples_*.json",
        f"{toxic_model_safe}_*_samples_*.csv",
        f"{non_toxic_model_safe}_*_samples_*.json",
        f"{non_toxic_model_safe}_*_samples_*.csv",
        f"sorted_toxic_dataset_{toxic_model_safe}.json",
        f"sorted_non_toxic_dataset_{toxic_model_safe}.json",
        f"*_{toxic_model_safe}_*_analysis.json"
    ]
    
    # First, collect all files that would be removed
    files_to_remove = []
    for pattern in patterns:
        files = glob.glob(os.path.join(cache_dir, pattern))
        files_to_remove.extend(files)
    
    if files_to_remove:
        print(f"  📋 Found {len(files_to_remove)} existing files to remove:")
        for file_path in files_to_remove:
            print(f"    - {os.path.basename(file_path)}")
        
        # Remove the files
        removed_files = []
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"  🗑️  Removed: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"  ⚠️  Could not remove {file_path}: {e}")
        
        print(f"  ✅ Cleaned {len(removed_files)} existing files")
    else:
        print(f"  ℹ️  No existing files found to clean")


def generate_datasets(config: DictConfig):
    """
    Generate toxic and non-toxic datasets with train/test splits.
    
    Args:
        config: Configuration containing:
            - toxic_model: Model for toxic dataset generation
            - non_toxic_model: Model for non-toxic dataset generation
            - toxicity_metric: Metric to use for filtering
            - toxicity_threshold: Threshold for toxic dataset
            - non_toxic_threshold: Threshold for non-toxic dataset
            - train_samples: Number of samples for training
            - test_samples: Number of samples for testing
            - sort_classifier: Classifier for sorting test datasets
            - sort_threshold: Threshold for sorting
    """
    
    print("🚀 Starting automated dataset generation...")


    
    # Generate toxic datasets
    print("\n📚 Generating TOXIC datasets...")
    
    # Toxic train dataset
    print("  🔄 Generating toxic train dataset...")
    toxic_train_config = config.copy()
    toxic_train_config.dataset.model_name = config.toxic_model
    toxic_train_config.dataset.num_samples = config.dataset.train_samples
    toxic_train_config.dataset.prompt_offset = 0
    toxic_train_config.dataset.seed = 42
    toxic_train_config.dataset.toxicity_threshold = config.dataset.toxicity_threshold
    toxic_train_config.dataset.toxicity_metric = config.dataset.toxicity_metric
    toxic_train_config.dataset.sort_out_toxic_nontoxic = False
    
    toxic_train_id = generate_dataset(toxic_train_config)
    print(f"  ✅ Toxic train dataset: {toxic_train_id}")
    
    # Toxic test dataset
    print("  🔄 Generating toxic test dataset...")
    toxic_test_config = config.copy()
    toxic_test_config.dataset.model_name = config.toxic_model
    toxic_test_config.dataset.num_samples = config.dataset.test_samples
    toxic_test_config.dataset.prompt_offset = config.dataset.train_samples
    toxic_test_config.dataset.seed = 123
    toxic_test_config.dataset.toxicity_threshold = config.dataset.toxicity_threshold
    toxic_test_config.dataset.toxicity_metric = config.dataset.toxicity_metric
    toxic_test_config.dataset.sort_out_toxic_nontoxic = True
    toxic_test_config.dataset.sort_classifier = config.dataset.sort_classifier
    toxic_test_config.dataset.sort_threshold = config.dataset.sort_threshold
    toxic_test_config.dataset.toxic_model_name = config.toxic_model  # Use toxic model name for sorted datasets
    
    toxic_test_id = generate_dataset(toxic_test_config)
    print(f"  ✅ Toxic test dataset: {toxic_test_id}")
    
    # Generate non-toxic datasets
    print("\n📚 Generating NON-TOXIC datasets...")
    
    # Non-toxic train dataset
    print("  🔄 Generating non-toxic train dataset...")
    nontoxic_train_config = config.copy()
    nontoxic_train_config.dataset.model_name = config.non_toxic_model
    nontoxic_train_config.dataset.num_samples = config.dataset.train_samples
    nontoxic_train_config.dataset.prompt_offset = 0
    nontoxic_train_config.dataset.seed = 42
    nontoxic_train_config.dataset.toxicity_threshold = config.dataset.non_toxic_threshold
    nontoxic_train_config.dataset.toxicity_metric = config.dataset.toxicity_metric
    nontoxic_train_config.dataset.sort_out_toxic_nontoxic = False
    
    nontoxic_train_id = generate_dataset(nontoxic_train_config)
    print(f"  ✅ Non-toxic train dataset: {nontoxic_train_id}")
    
    # Non-toxic test dataset
    print("  🔄 Generating non-toxic test dataset...")
    nontoxic_test_config = config.copy()
    nontoxic_test_config.dataset.model_name = config.non_toxic_model
    nontoxic_test_config.dataset.num_samples = config.dataset.test_samples
    
    # Use same test prompts if configured, otherwise use different prompts
    same_test_prompt = getattr(config.dataset, 'same_test_prompt', False)
    if same_test_prompt:
        nontoxic_test_config.dataset.prompt_offset = config.dataset.train_samples  # Same as toxic test
        print(f"  📝 Using same test prompts (offset: {config.dataset.train_samples})")
    else:
        nontoxic_test_config.dataset.prompt_offset = config.dataset.train_samples + config.dataset.test_samples  # Different prompts
        print(f"  📝 Using different test prompts (offset: {config.dataset.train_samples + config.dataset.test_samples})")
    
    nontoxic_test_config.dataset.seed = 123
    nontoxic_test_config.dataset.toxicity_threshold = config.dataset.non_toxic_threshold
    nontoxic_test_config.dataset.toxicity_metric = config.dataset.toxicity_metric
    nontoxic_test_config.dataset.sort_out_toxic_nontoxic = True
    nontoxic_test_config.dataset.sort_classifier = config.dataset.sort_classifier
    nontoxic_test_config.dataset.sort_threshold = config.dataset.sort_threshold
    nontoxic_test_config.dataset.toxic_model_name = config.toxic_model  # Use toxic model name for sorted datasets
    
    nontoxic_test_id = generate_dataset(nontoxic_test_config)
    print(f"  ✅ Non-toxic test dataset: {nontoxic_test_id}")
    
    print("\n🎉 All datasets generated successfully!")
    print(f"📊 Summary:")
    print(f"  Toxic train: {toxic_train_id}")
    print(f"  Toxic test: {toxic_test_id}")
    print(f"  Non-toxic train: {nontoxic_train_id}")
    print(f"  Non-toxic test: {nontoxic_test_id}")
    
    # Show prompt configuration
    same_test_prompt = getattr(config.dataset, 'same_test_prompt', False)
    print(f"\n📝 Prompt Configuration:")
    print(f"  Training: Same prompts for toxic and non-toxic models")
    print(f"  Test: {'Same prompts' if same_test_prompt else 'Different prompts'} for toxic and non-toxic models")
    print(f"  Toxic test offset: {config.dataset.train_samples}")
    print(f"  Non-toxic test offset: {config.dataset.train_samples if same_test_prompt else config.dataset.train_samples + config.dataset.test_samples}")
    
    return {
        "toxic_train": toxic_train_id,
        "toxic_test": toxic_test_id,
        "nontoxic_train": nontoxic_train_id,
        "nontoxic_test": nontoxic_test_id
    }


@hydra.main(config_path="../configs", config_name="full_pipeline.yaml", version_base=None)
def main(cfg: DictConfig):
    """Generate all datasets with the specified configuration."""
    print("🚀 Starting automated dataset generation...")
    
    # Get the cache directory from config
    cache_dir = getattr(cfg.dataset, 'cache_dir', 'datasets')
    
    # Remove existing files for the specified models
    remove_existing_model_files(cfg.toxic_model, cfg.non_toxic_model, cache_dir)
    
    # Generate datasets
    dataset_ids = generate_datasets(cfg)
    
    print(f"\n📊 Generated datasets:")
    for dataset_type, dataset_id in dataset_ids.items():
        print(f"  {dataset_type}: {dataset_id}")


if __name__ == "__main__":
    main()

