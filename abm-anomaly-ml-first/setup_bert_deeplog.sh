#!/bin/bash

# BERT-DeepLog Setup Script
# Installs all required dependencies for the BERT-DeepLog anomaly detection system

echo "🤖 Setting up BERT-DeepLog Anomaly Detection System..."
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_bert_deeplog" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_bert_deeplog
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_bert_deeplog/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - for GPU, user should modify)
echo "🧠 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other ML libraries
echo "🤗 Installing Transformers and ML libraries..."
pip install transformers
pip install scikit-learn
pip install numpy
pip install pandas

# Install visualization libraries
echo "📊 Installing visualization libraries..."
pip install matplotlib
pip install seaborn

# Install utility libraries
echo "🛠️ Installing utility libraries..."
pip install tqdm
pip install pyyaml
pip install colorlog

# Create directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p logs
mkdir -p results

# Save requirements
echo "💾 Saving requirements..."
pip freeze > bert_deeplog_installed_requirements.txt

echo ""
echo "✅ BERT-DeepLog setup completed successfully!"
echo ""
echo "To use the system:"
echo "1. Activate the environment: source venv_bert_deeplog/bin/activate"
echo "2. Run the demo: python demonstrate_bert_deeplog.py"
echo "3. For training: python bert_deeplog_trainer.py --mode train --data_path ./data/your_logs.txt"
echo ""
echo "📂 Directory structure created:"
echo "  - models/     : Trained model storage"
echo "  - data/       : Input data files"
echo "  - logs/       : Training and execution logs"
echo "  - results/    : Analysis results and outputs"
echo ""
echo "📋 Requirements saved to: bert_deeplog_installed_requirements.txt"
echo "=================================================="
