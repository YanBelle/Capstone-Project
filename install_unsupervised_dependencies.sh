#!/bin/bash

# Unsupervised EJ Analysis Dependencies Installation Script
# This script installs the required dependencies for unsupervised EJ log analysis

echo "Installing Unsupervised EJ Analysis Dependencies..."

# Check if we're in a container or local environment
if [ -f /.dockerenv ]; then
    echo "Detected Docker environment"
    PIP_CMD="pip3"
else
    echo "Detected local environment"
    PIP_CMD="pip3"
fi

# Install core ML dependencies
echo "Installing core machine learning dependencies..."
$PIP_CMD install numpy==1.24.3
$PIP_CMD install pandas==2.1.3
$PIP_CMD install scikit-learn==1.3.2

# Install sentence transformers
echo "Installing sentence transformers..."
$PIP_CMD install sentence-transformers==2.2.2

# Install clustering dependencies
echo "Installing clustering dependencies..."
$PIP_CMD install hdbscan==0.8.33
$PIP_CMD install umap-learn==0.5.4

# Install visualization dependencies
echo "Installing visualization dependencies..."
$PIP_CMD install matplotlib==3.8.0
$PIP_CMD install seaborn==0.13.0
$PIP_CMD install plotly==5.18.0

# Verify installations
echo "Verifying installations..."

python3 -c "
import numpy as np
print(f'✅ NumPy version: {np.__version__}')

import pandas as pd
print(f'✅ Pandas version: {pd.__version__}')

import sklearn
print(f'✅ Scikit-learn version: {sklearn.__version__}')

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ SentenceTransformers working - model loaded successfully')
except Exception as e:
    print(f'❌ SentenceTransformers failed: {e}')

try:
    import hdbscan
    print(f'✅ HDBSCAN version: {hdbscan.__version__}')
except Exception as e:
    print(f'❌ HDBSCAN failed: {e}')

try:
    import umap
    print('✅ UMAP loaded successfully')
except Exception as e:
    print(f'❌ UMAP failed: {e}')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    print('✅ Visualization libraries loaded successfully')
except Exception as e:
    print(f'❌ Visualization libraries failed: {e}')

print('\\n🎉 Unsupervised EJ analysis dependencies installation completed!')
"

echo ""
echo "You can now use the unsupervised analysis endpoints:"
echo "- POST /api/v1/unsupervised/analyze"
echo "- GET /api/v1/unsupervised/anomalies"
echo "- GET /api/v1/unsupervised/patterns"
echo "- POST /api/v1/unsupervised/analyze-session"
echo "- GET /api/v1/unsupervised/status"
echo "- POST /api/v1/unsupervised/dashboard"
echo "- POST /api/v1/unsupervised/export"
