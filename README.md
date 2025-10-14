# IRL Alignment Auditor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu6/IRL-Alignment-Auditor/blob/main/example_usage.ipynb)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2510.06096-red)](https://arxiv.org/abs/2510.06096)

A production-ready pipeline for auditing and refining LLM objectives using Inverse Reinforcement Learning (IRL) with Variational Inference. This package provides a clean, organized implementation of Bayesian IRL methods for learning reward models from human preferences.

**Paper**: [The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives](https://arxiv.org/abs/2510.06096)

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Matthieu6/IRL-Alignment-Auditor.git
cd IRL-Alignment-Auditor

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Hugging Face Token Setup

**Important**: You need a Hugging Face token to download models and save results to the Hub.

1. Get your token from: https://huggingface.co/settings/tokens
2. Login using the CLI:
```bash
huggingface-cli login
```

Or set the environment variable:
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### 3. Verify Installation

```bash
# Test the installation
python -c "import irl_pipeline; print('Installation successful!')"
```

## 📓 Google Colab Example

**Try it now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu6/IRL-Alignment-Auditor/blob/main/example_usage.ipynb)

The example notebook demonstrates:
- Complete pipeline execution with Llama-3.2-1B (including IRL+RLHF)
- Custom model configurations
- Individual component usage
- Troubleshooting tips

**Note**: The complete IRL+RLHF pipeline is only available for Llama-3.2-1B. Other models support dataset generation and IRL training only.

## 🎯 Main Variables to Change

### Model Configuration

The pipeline uses two main models that you can easily change:

**Toxic Model** (generates toxic content):
- `EleutherAI/pythia-70m` (small)
- `EleutherAI/pythia-410m` (medium)
- `EleutherAI/gpt-neo-125m` (small)
- `meta-llama/Llama-3.2-1B` (default, large)

**Non-Toxic Model** (generates detoxified content):
- `ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100`
- `ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-100`
- `ajagota71/gpt-neo-125m-s-nlp-detox-checkpoint-epoch-100`
- `ajagota71/llama-3-2-1b-rlhf-kl-p5-target-2p5-lr-3e-6-checkpoint-epoch-100` (default)

**⚠️ RLHF Limitation**: The complete IRL+RLHF pipeline (using learned reward models) is currently only supported for the Llama-3.2-1B model. Other models can be used for dataset generation and IRL training, but RLHF will use standard training without the learned IRL reward function.

### Other Key Variables

- `train_samples`: Number of training samples (default: 3100)
- `test_samples`: Number of test samples (default: 600)
- `toxicity_threshold`: Threshold for filtering toxic prompts (default: 0.9)
- `sort_threshold`: Threshold for sorting outputs (default: 0.7)

## 🏃‍♂️ How to Run

### Complete Pipeline (Recommended for first run)

```bash
# Run everything: dataset generation + IRL training + analysis (default: llama-1B)
./run_pipeline.sh complete
```

### Individual Components

#### 1. Generate Datasets Only
```bash
# Generate datasets with default models
./run_pipeline.sh generate

# Generate with custom models
./run_pipeline.sh generate \
  toxic_model=meta-llama/Llama-3.2-1B \
  non_toxic_model=ajagota71/llama-3-2-1b-rlhf-kl-p5-target-2p5-lr-3e-6-checkpoint-epoch-100
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
./run_pipeline.sh rlhf \
  rlhf_config.model=llama_3_2_1b

# Available RLHF models:
# - smolLM_135m
# - smolLM_360m  
# - llama_3_2_1b
```

**⚠️ Important**: RLHF using the IRL reward model is currently only available for the Llama-3.2-1B model. Other models use standard RLHF training without the learned IRL reward function.

#### 4. Analyze Spurious Features
```bash
# Run spurious features analysis (requires trained model)
./run_pipeline.sh analyze
```

#### 5. Evaluate RLHF Models
```bash
# Evaluate trained RLHF model (trained_model_root is path/huggingface name of trained model)
./run_pipeline.sh evaluate \
  evaluate_rlhf.model.trained_model_root=user/model-name
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
- `meta-llama/Llama-3.2-1B` ⭐ **(Recommended for IRL+RLHF pipeline)**

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
@article{bou2024alignment,
  title={The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives},
  author={Bou, Matthieu and Patel, Nyal and Jagota, Arjun and Krishna, Satyapriya and Parbhoo, Sonali},
  journal={arXiv preprint arXiv:2510.06096},
  year={2024},
  url={https://arxiv.org/abs/2510.06096}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

