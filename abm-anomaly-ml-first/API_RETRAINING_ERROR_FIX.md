# API Retraining Error Fix

## Issue
The API service is unable to trigger retraining because it's missing the `ml_analyzer` module:

```
Error triggering retraining: No module named 'ml_analyzer'
```

## Root Cause
The API service attempts to import and use `MLFirstAnomalyDetector` from `ml_analyzer`, but this module only exists in the anomaly-detector service and wasn't copied to the API service container.

## Resolution

### 1. Copy Required Files
The solution is to copy the necessary files to the API service directory:
- `ml_analyzer.py` - Core ML detection logic
- `simple_embeddings.py` - Dependency for embedding generation

### 2. Ensure Shared Model Access
Both services must access the same model files:
- The API service needs access to the `/app/models` directory
- Both services should mount the same volume for models

### 3. Handle Model Initialization
The error `No such file or directory: '/app/models/scaler.pkl'` occurs because the system is trying to load models that don't exist yet. We've added:
- A fix to properly handle missing models
- Default expert rules JSON

## How to Apply the Fix

1. Run the fix script:
```bash
chmod +x fix_ml_analyzer.sh
./fix_ml_analyzer.sh
```

2. Rebuild the services:
```bash
docker-compose build api anomaly-detector
docker-compose up -d
```

3. Verify the fix:
- Check API logs: `docker-compose logs -f api`
- Try triggering retraining from the dashboard
- The error should no longer appear

## Technical Details

This fix ensures that both the API service and anomaly-detector service have:
1. The same ML analyzer code
2. Access to the same model files
3. Proper error handling for first-run scenarios

The API service can now independently load and use the ML components without errors, even before models are trained.
