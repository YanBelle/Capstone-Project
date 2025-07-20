# Models Missing Fix

## Issue
The anomaly detector service is showing an error when trying to load models:

```
Error loading models: [Errno 2] No such file or directory: '/app/models/scaler.pkl'
```

## Root Cause
The system is trying to load pre-trained models during initialization, but these models don't exist yet since this appears to be a first-time setup.

## Resolution

### Option 1: Better Error Handling (Recommended)
Modify the `load_models` function in `main.py` to properly handle the case when models don't exist yet:

```python
def load_models(self):
    """Load pre-trained models if they exist"""
    model_dir = "/app/models"
    # Create the directory if it doesn't exist
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
            # Don't treat this as an error, just continue with default models
    else:
        logger.info("No existing models found. Will train on first batch.")
```

### Option 2: Create Default Models
Create placeholder model files during initialization to ensure they exist:

1. Create a script to generate empty models:
   - Default `isolation_forest.pkl`
   - Default `one_class_svm.pkl`
   - Default `scaler.pkl`
   - Default `pca.pkl`
   - Default `expert_rules.json`

2. Run this script during container startup or as part of the build process

### Implementation
The `fix_ml_analyzer.sh` script has been provided to:

1. Copy the ML analyzer files to the API service
2. Create a placeholder expert_rules.json file
3. Set up the environment for first-time initialization

Run:
```bash
./fix_ml_analyzer.sh
```

Then rebuild and restart the services:
```bash
docker-compose build api anomaly-detector
docker-compose up -d
```

This will ensure that both services can properly handle the case when models don't exist yet.
