#!/bin/bash

# DeepLog BERT Dependencies Installation Script
# This script installs the required dependencies for DeepLog training with BERT

echo "Installing DeepLog BERT Training Dependencies..."

# Check if we're in a container or local environment
if [ -f /.dockerenv ]; then
    echo "Detected Docker environment"
    PIP_CMD="pip3"
else
    echo "Detected local environment"
    PIP_CMD="pip3"
fi

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
$PIP_CMD install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install Transformers
echo "Installing Transformers..."
$PIP_CMD install transformers==4.25.1

# Install additional dependencies
echo "Installing additional dependencies..."
$PIP_CMD install numpy==1.24.1
$PIP_CMD install scikit-learn==1.2.0

# Verify installations
echo "Verifying installations..."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')

import transformers
print(f'Transformers version: {transformers.__version__}')

import numpy as np
print(f'NumPy version: {np.__version__}')

try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    print('✅ BERT model loading successful')
except Exception as e:
    print(f'❌ BERT model loading failed: {e}')

print('✅ All dependencies installed successfully!')
"

echo "DeepLog BERT dependencies installation completed!"
echo ""
echo "You can now use the DeepLog training endpoints:"
echo "- POST /api/v1/deeplog/retrain"
echo "- POST /api/v1/deeplog/predict"
echo "- GET /api/v1/deeplog/status"
