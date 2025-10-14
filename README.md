# IRL-Bayesian Pipeline

A production-ready pipeline for text detoxification using Inverse Reinforcement Learning (IRL) with Variational Inference. This package provides a clean, organized implementation of Bayesian IRL methods for learning reward models from human preferences.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd irl-bayesian

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test the installation
python -c "import irl_pipeline; print('Installation successful!')"
```

## 🎯 Main Variables to Change

### Model Configuration

The pipeline uses two main models that you can easily change:

**Toxic Model** (generates toxic content):
- `EleutherAI/pythia-70m` (default, small)
- `EleutherAI/pythia-410m` (medium)
- `EleutherAI/gpt-neo-125m` (small)
- `meta-llama/Llama-3.2-1B` (large)

**Non-Toxic Model** (generates detoxified content):
- `ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100` (default)
- `ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-100`
- `ajagota71/gpt-neo-125m-s-nlp-detox-checkpoint-epoch-100`
- `ajagota71/llama-3-2-1b-rlhf-kl-p5-target-2p5-lr-3e-6-checkpoint-epoch-100`

### Other Key Variables

- `train_samples`: Number of training samples (default: 3100)
- `test_samples`: Number of test samples (default: 600)
- `toxicity_threshold`: Threshold for filtering toxic prompts (default: 0.9)
- `sort_threshold`: Threshold for sorting outputs (default: 0.7)

## 🏃‍♂️ How to Run

### Complete Pipeline (Recommended for first run)

```bash
# Run everything: dataset generation + IRL training + analysis
./run_pipeline.sh complete
```

### Individual Components

#### 1. Generate Datasets Only
```bash
# Generate datasets with default models
./run_pipeline.sh generate

# Generate with custom models
./run_pipeline.sh generate \
  toxic_model=EleutherAI/pythia-70m \
  non_toxic_model=ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100
```

#### 2. Train IRL Model Only
```bash
# Train IRL model (requires existing datasets)
./run_pipeline.sh train

# Train with custom parameters
./run_pipeline.sh train \
  training.n_steps=5000 \
  training.learning_rate=0.02
```

#### 3. Run RLHF Training
```bash
# Run RLHF with default model
./run_pipeline.sh rlhf

# Run RLHF with specific model
./run_pipeline.sh rlhf rlhf_config.model=llama_3_2_1b

# Available RLHF models:
# - smolLM_135m
# - smolLM_360m  
# - llama_3_2_1b
```

#### 4. Analyze Spurious Features
```bash
# Run spurious features analysis (requires trained model)
./run_pipeline.sh analyze
```

#### 5. Evaluate RLHF Models
```bash
# Evaluate trained RLHF model
./run_pipeline.sh evaluate evaluate_rlhf.model.trained_model_root=user/model-name
```

## 📊 Outputs

### Generated Datasets
- `datasets/*_samples_original.json`: Original model outputs
- `datasets/*_samples_detoxified.json`: Detoxified model outputs
- `datasets/sorted_toxic_dataset_*.json`: Sorted toxic samples
- `datasets/sorted_non_toxic_dataset_*.json`: Sorted non-toxic samples

### Training Results
- `outputs/re_irl/{timestamp}/`: Training outputs directory
- `round_{i}/`: Results for each training round
- `summary.json`: Summary of all rounds
- Various plots and visualizations

## 🔧 Configuration Files

The pipeline uses Hydra for configuration management:

- **`configs/full_pipeline.yaml`**: Main pipeline configuration
- **`configs/dataset.yaml`**: Dataset generation settings
- **`configs/re_irl_config.yaml`**: IRL training parameters
- **`configs/rlhf_config.yaml`**: RLHF training parameters
- **`configs/rlhf/`**: Model-specific RLHF configurations

## 🧪 Compatible Models

### Small Models (Fast, Good for Testing)
- `EleutherAI/pythia-70m`
- `EleutherAI/gpt-neo-125m`
- `HuggingFaceTB/SmolLM-135M`

### Medium Models (Balanced)
- `EleutherAI/pythia-410m`
- `HuggingFaceTB/SmolLM-360M`

### Large Models (Best Performance)
- `meta-llama/Llama-3.2-1B`

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Use smaller models
./run_pipeline.sh full toxic_model=EleutherAI/pythia-70m
```

**Missing Dependencies:**
```bash
pip install -r requirements.txt
```

**Permission Errors:**
```bash
chmod +x run_pipeline.sh
```

## 🔬 Research Background

This implementation is based on:
- **Inverse Reinforcement Learning**: Learning reward functions from human preferences
- **Variational Inference**: Efficient Bayesian inference for large-scale problems
- **Bradley-Terry Model**: Probabilistic ranking model for pairwise preferences
- **Text Detoxification**: Reducing toxicity in language model outputs

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{irl_bayesian_pipeline,
  title={The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives},
  author={Matthieu Bou, Nyal Patel, Arjun Jagota, Satyapriya Krishna, Sonali Parbhoo},
  year={2024},
  url={https://github.com/yourusername/irl-bayesian}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

