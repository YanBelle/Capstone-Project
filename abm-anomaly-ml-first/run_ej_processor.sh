#!/bin/bash

# EJ Rule-Based Processor Runner Script
# =====================================

echo "🚀 EJ Rule-Based Processor"
echo "=========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Create data directories if they don't exist
echo "📁 Setting up data directories..."
mkdir -p ./data/input
mkdir -p ./data/processed

# Check for EJ files
if [ ! -f ./data/input/*.txt ] 2>/dev/null; then
    echo "⚠️  No EJ files found in ./data/input/"
    echo "   Please place your EJ .txt files in ./data/input/ directory"
    echo "   Example: cp /path/to/your/ej_files/*.txt ./data/input/"
    exit 1
fi

echo "✅ EJ files found in ./data/input/"

# Run the CSV-safe processor (recommended)
echo "🔄 Running CSV-safe EJ processor..."
python3 ej_rule_processor_csv_safe.py

echo "🎉 Processing complete!"
echo "📁 Check results in ./data/processed/"
