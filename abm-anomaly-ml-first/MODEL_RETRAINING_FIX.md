# Model Retraining and API Import Fix Instructions

## Overview

This guide addresses two critical issues in the ML-First ABM System:

1. API service is missing the `ml_analyzer` module needed for retraining
2. The system is failing to load models that don't exist yet

## Quick Fix

Run the complete fix script:

```bash
chmod +x complete_fix.sh
./complete_fix.sh
```

Then rebuild and restart the services:

```bash
docker-compose build api anomaly-detector
docker-compose up -d
```

## Manual Fix Steps

If you prefer to fix the issues manually:

### 1. Copy ML Analyzer to API Service

```bash
cp services/anomaly-detector/ml_analyzer.py services/api/
cp services/anomaly-detector/simple_embeddings.py services/api/
```

### 2. Create Models Directory and Placeholder Files

```bash
mkdir -p data/models
```

Create a basic expert_rules.json file:
```bash
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
```

### 3. Update API Dockerfile

Edit `services/api/Dockerfile` to add directory creation:

```dockerfile
# After COPY . .
# Create necessary directories for models and data
RUN mkdir -p /app/models /app/logs /app/input/processed /app/output /app/cache
```

### 4. Improve Error Handling in main.py

Edit `services/anomaly-detector/main.py` to properly handle missing models:

```python
def load_models(self):
    """Load pre-trained models if they exist"""
    model_dir = "/app/models"
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(model_dir, "isolation_forest.pkl")):
        logger.info("Loading existing ML models...")
        try:
            import joblib
            self.detector.isolation_forest = joblib.load(
                os.path.join(model_dir, "isolation_forest.pkl")
            )
            self.detector.one_class_svm = joblib.load(
                os.path.join(model_dir, "one_class_svm.pkl")
            )
            self.detector.scaler = joblib.load(
                os.path.join(model_dir, "scaler.pkl")
            )
            if os.path.exists(os.path.join(model_dir, "pca.pkl")):
                self.detector.pca = joblib.load(
                    os.path.join(model_dir, "pca.pkl")
                )
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading models: {str(e)}. Will train new models.")
            # Continue with default models
    else:
        logger.info("No existing models found. Will train on first batch.")
```

## Verification

After applying the fixes:

1. Check logs for errors:
   ```bash
   docker-compose logs -f api anomaly-detector
   ```

2. Verify retraining works through the dashboard:
   - Go to http://localhost:3000
   - Navigate to the Expert Review section
   - Label some anomalies
   - Trigger retraining

3. Watch the anomaly-detector logs to confirm that models are properly being trained on first run.

## Expected Behavior

- The anomaly-detector service should start without errors about missing models
- The API service should be able to trigger retraining without the "No module named 'ml_analyzer'" error
- Both services will now share the same model files via the mounted volume

This fix allows the system to initialize properly and enables the continuous learning feature to work as designed.
