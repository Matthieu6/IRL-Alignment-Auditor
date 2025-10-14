#!/bin/bash
# IRL-Bayesian Pipeline Execution Script
# Simplified version for pip-based environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "irl_pipeline" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the irl-bayesian root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found! Please ensure Python is installed and in your PATH"
    exit 1
fi

# Check if required packages are installed
print_status "Checking dependencies..."
python -c "import torch, transformers, datasets, pyro, omegaconf, wandb, trl" 2>/dev/null || {
    print_error "Required packages not found!"
    print_status "Please run: pip install -r requirements.txt"
    exit 1
}

# Add current directory to Python path so it can find irl_pipeline module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS...]"
    echo ""
    echo "Commands:"
    echo "  generate    Generate datasets only"
    echo "  train       Train IRL model only (requires existing datasets)"
    echo "  analyze     Run spurious features analysis (requires trained model)"
    echo "  rlhf        Run RLHF training only"
    echo "  evaluate    Evaluate trained RLHF models on toxicity"
    echo "  full        Run full pipeline (generate + train)"
    echo "  complete    Run complete pipeline (generate + train + analyze)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 full                                           # Run generate + train"
    echo "  $0 complete                                       # Run generate + train + analyze"
    echo "  $0 generate                                       # Generate datasets only"
    echo "  $0 train                                          # Train IRL model only"
    echo "  $0 analyze                                        # Run spurious analysis only"
    echo "  $0 rlhf                                           # Run RLHF training only"
    echo "  $0 rlhf rlhf_config.model=llama_3_2_1b           # Run RLHF with specific model"
    echo "  $0 evaluate evaluate_rlhf.model.trained_model_root=user/model-name  # Evaluate RLHF model"
    echo ""
    echo "Configuration overrides:"
    echo "  Global models (set in configs/full_pipeline.yaml):"
    echo "    toxic_model=MODEL_NAME         # Base model used across pipeline"
    echo "    non_toxic_model=MODEL_NAME     # Detoxified model for datasets"
    echo ""
    echo "  Component-specific overrides:"
    echo "    $0 train model.base_model_name=MODEL          # Override IRL model"
    echo "    $0 rlhf rlhf_config.model=pythia_410m         # Use RLHF preset"
    echo "    $0 evaluate evaluate_rlhf.model.trained_model_root=PATH  # Set eval model"
}

# Function to run dataset generation
run_dataset_generation() {
    print_status "Starting dataset generation..."
    python scripts/generate_datasets.py "$@"
    print_success "Dataset generation completed!"
}

# Function to run IRL training
run_irl_training() {
    print_status "Starting IRL model training..."
    python scripts/train_irl_model.py "$@"
    print_success "IRL model training completed!"
}

# Function to run spurious features analysis
run_spurious_analysis() {
    print_status "Starting spurious features analysis..."
    python scripts/analyze_spurious_features.py spurious_features.enabled=true "$@"
    print_success "Spurious features analysis completed!"
}

# Function to run RLHF training
run_rlhf_training() {
    print_status "Starting RLHF training..."
    python scripts/train_rlhf.py "$@"
    print_success "RLHF training completed!"
}

# Function to run RLHF evaluation
run_rlhf_evaluation() {
    print_status "Starting RLHF model evaluation..."
    python scripts/evaluate_rlhf.py "$@"
    print_success "RLHF evaluation completed!"
}

# Function to run full pipeline (generate + train)
run_full_pipeline() {
    print_status "Starting full IRL pipeline..."
    
    # Step 1: Generate datasets
    print_status "Step 1/2: Generating datasets..."
    python scripts/generate_datasets.py "$@"
    print_success "Dataset generation completed!"
    
    # Step 2: Train IRL model
    print_status "Step 2/2: Training IRL model..."
    python scripts/train_irl_model.py "$@"
    print_success "IRL model training completed!"
    
    print_success "Full pipeline completed successfully!"
}

# Function to run complete pipeline (generate + train + analyze)
run_complete_pipeline() {
    print_status "Starting complete IRL pipeline..."
    
    # Step 1: Generate datasets
    print_status "Step 1/3: Generating datasets..."
    python scripts/generate_datasets.py "$@"
    print_success "Dataset generation completed!"
    
    # Step 2: Train IRL model
    print_status "Step 2/3: Training IRL model..."
    python scripts/train_irl_model.py "$@"
    print_success "IRL model training completed!"
    
    # Step 3: Run spurious features analysis
    print_status "Step 3/3: Running spurious features analysis..."
    python scripts/analyze_spurious_features.py spurious_features.enabled=true "$@"
    print_success "Spurious features analysis completed!"
    
    print_success "Complete pipeline finished successfully!"
}

# Main script logic
case "${1:-help}" in
    "generate")
        shift  # Remove the first argument
        run_dataset_generation "$@"
        ;;
    "train")
        shift  # Remove the first argument
        run_irl_training "$@"
        ;;
    "analyze")
        shift  # Remove the first argument
        run_spurious_analysis "$@"
        ;;
    "rlhf")
        shift  # Remove the first argument
        run_rlhf_training "$@"
        ;;
    "evaluate")
        shift  # Remove the first argument
        run_rlhf_evaluation "$@"
        ;;
    "full")
        shift  # Remove the first argument
        run_full_pipeline "$@"
        ;;
    "complete")
        shift  # Remove the first argument
        run_complete_pipeline "$@"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

