#!/bin/bash
# Script to fix the ml_analyzer import issue in API service

echo "Fixing ML Analyzer import issue in API service..."

# Check if we're in the right directory
if [ ! -d "services/anomaly-detector" ] || [ ! -d "services/api" ]; then
  echo "Error: Please run this script from the abm-anomaly-ml-first directory"
  exit 1
fi

echo "Copying ML analyzer from anomaly-detector to API service..."

# Copy the required files
cp services/anomaly-detector/ml_analyzer.py services/api/
cp services/anomaly-detector/simple_embeddings.py services/api/

echo "Updating the API service Dockerfile..."

# Check if the Dockerfile already has mkdir commands
if ! grep -q "mkdir -p /app/models" services/api/Dockerfile; then
  # Add before the CMD line
  sed -i '' 's/EXPOSE 8000/EXPOSE 8000\n\n# Create necessary directories\nRUN mkdir -p \/app\/models \/app\/logs \/app\/input\/processed \/app\/output \/app\/cache/' services/api/Dockerfile
  echo "Updated API Dockerfile with model directories"
else
  echo "API Dockerfile already has model directories"
fi

# Update docker-compose.yml
echo "Checking docker-compose.yml for shared volumes..."
if ! grep -q "./data/models:/app/models" docker-compose.yml; then
  echo "Please update docker-compose.yml manually to add shared volumes:"
  echo "
services:
  api:
    volumes:
      - ./data/models:/app/models
  anomaly-detector:
    volumes:
      - ./data/models:/app/models
"
fi

echo "Fix applied. Please rebuild the containers with:"
echo "docker-compose build api"
echo "docker-compose up -d"
