#!/bin/bash
# Fix the ml_analyzer import in API service

set -e

echo "=== ML Analyzer Import Fix ==="
echo "This script will fix the 'No module named ml_analyzer' error in API service"

# Change to project root directory
cd "$(dirname "$0")"

# Check if in Docker container
if [ -f /.dockerenv ]; then
    echo "Running in Docker container. This fix should be applied on the host system."
    exit 1
fi

# Check if files exist
if [ ! -f services/anomaly-detector/ml_analyzer.py ]; then
    echo "Error: ml_analyzer.py not found in services/anomaly-detector/"
    exit 1
fi

# Create directories
echo "Creating data/models directory if it doesn't exist..."
mkdir -p data/models

# Copy files
echo "Copying ML files to API service..."
cp services/anomaly-detector/ml_analyzer.py services/api/
cp services/anomaly-detector/simple_embeddings.py services/api/

echo "Creating placeholder files in models directory..."
if [ ! -f data/models/expert_rules.json ]; then
    echo '{
  "rules": [
    {
      "name": "supervisor_mode",
      "pattern": "SUPERVISOR MODE",
      "confidence": 0.9,
      "severity": "medium"
    }
  ]
}' > data/models/expert_rules.json
    echo "Created expert_rules.json"
fi

echo "Fix applied. Please rebuild the services:"
echo ""
echo "docker-compose build api anomaly-detector"
echo "docker-compose up -d"
echo ""
echo "This will ensure both services have the ml_analyzer module available."
