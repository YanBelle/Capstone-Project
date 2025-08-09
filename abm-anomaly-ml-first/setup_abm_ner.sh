#!/bin/bash
"""
ABM NER Fine-tuning Setup Script
===============================

Complete setup script for fine-tuning BERT NER model for ABM log patterns.
This script prepares the environment and runs the fine-tuning pipeline.
"""

set -e  # Exit on any error

echo "🏧 ABM NER Fine-tuning Setup"
echo "============================="

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

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

print_status "Python version: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is required but not installed."
    exit 1
fi

# Step 1: Create virtual environment (optional but recommended)
if [ ! -d "venv-ner" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv-ner
    print_success "Virtual environment created: venv-ner"
else
    print_warning "Virtual environment already exists: venv-ner"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv-ner/bin/activate

# Step 2: Install dependencies
print_status "Installing NER fine-tuning dependencies..."
pip install --upgrade pip
pip install -r requirements-ner.txt

# Check PyTorch installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 3: Create directories
print_status "Creating necessary directories..."
mkdir -p models
mkdir -p data/abm_logs
mkdir -p results

# Step 4: Check for sample data
if [ ! -f "data/abm_logs/sample.txt" ]; then
    print_warning "No ABM log files found in data/abm_logs/"
    print_status "You can add your ABM log files to data/abm_logs/ directory"
    print_status "Or the script will create sample data for testing"
fi

# Step 5: Download pre-trained BERT model (if needed)
print_status "Checking BERT model availability..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForTokenClassification
try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=19)
    print('✅ BERT model successfully loaded')
except Exception as e:
    print(f'❌ Error loading BERT model: {e}')
    exit(1)
"

print_success "Environment setup completed!"

# Step 6: Run fine-tuning pipeline
echo ""
read -p "Do you want to run the fine-tuning pipeline now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting ABM NER fine-tuning pipeline..."
    python3 abm_ner_finetuning.py
    
    if [ $? -eq 0 ]; then
        print_success "Fine-tuning completed successfully!"
        print_status "Model saved to: ./abm-ner-model"
        
        # Step 7: Test the integration
        echo ""
        read -p "Do you want to test the enhanced sessionizer? (y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Testing enhanced ABM sessionizer..."
            python3 enhanced_abm_sessionizer.py
        fi
    else
        print_error "Fine-tuning failed. Check the logs above."
        exit 1
    fi
else
    print_status "Setup complete. You can run the fine-tuning later with:"
    echo "  python3 abm_ner_finetuning.py"
fi

echo ""
print_success "🎉 ABM NER setup completed!"
echo ""
echo "Next steps:"
echo "1. Add your ABM log files to data/abm_logs/"
echo "2. Run: python3 abm_ner_finetuning.py"
echo "3. Test with: python3 enhanced_abm_sessionizer.py"
echo "4. Integrate with your existing pipeline"
echo ""
echo "Files created:"
echo "  - abm_ner_finetuning.py (fine-tuning script)"
echo "  - enhanced_abm_sessionizer.py (integration script)"
echo "  - requirements-ner.txt (dependencies)"
echo "  - venv-ner/ (virtual environment)"
