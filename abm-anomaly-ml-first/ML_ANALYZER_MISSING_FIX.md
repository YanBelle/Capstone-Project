# ML Analyzer Missing Fix

## Problem

The API service is attempting to import the `ml_analyzer` module for the retraining functionality, but this file doesn't exist in the API service directory. It's only present in the anomaly-detector service.

Error message:
```
Error getting status: No module named 'ml_analyzer'
```

## Solution

There are two approaches to solving this issue:

### Option 1: Copy the required files to the API service (Recommended)

1. Copy `ml_analyzer.py` and its dependencies from anomaly-detector to the API service directory
2. Update the API service Dockerfile to include these files
3. This allows the API service to operate independently

### Option 2: Implement an API endpoint in the anomaly-detector

1. Create a retraining endpoint in the anomaly-detector service
2. Make the API service call this endpoint instead of trying to do the retraining itself
3. This approach keeps the code in one place but adds service dependency

## Implementation (Option 1)

Let's proceed with Option 1 as it's simpler and more robust:

1. Copy `ml_analyzer.py` and `simple_embeddings.py` from anomaly-detector to API service
2. Update API service requirements.txt to include necessary ML dependencies
3. Create a shared model directory accessible by both services

## Steps

1. Copy the required files:
```bash
cp services/anomaly-detector/ml_analyzer.py services/api/
cp services/anomaly-detector/simple_embeddings.py services/api/
```

2. Update the API service Dockerfile to include model directories:
```dockerfile
# Inside services/api/Dockerfile
RUN mkdir -p /app/models /app/logs /app/input/processed /app/output
```

3. Update docker-compose.yml to share the models volume:
```yaml
services:
  api:
    volumes:
      - ./data/models:/app/models
  anomaly-detector:
    volumes:
      - ./data/models:/app/models
```

These changes will ensure that both services have access to the same ML files and models.
