#!/bin/bash
# Complete fix for ML analyzer issues
# Resolves both API import error and model loading error

set -e

echo "=== ML First ABM System Fix ==="
echo "This script will fix both the API import error and the model loading error."

# Change to project root directory
cd "$(dirname "$0")"

# Create required directories
echo "Creating data directories..."
mkdir -p data/models data/logs data/input data/output

# Fix 1: Copy ML analyzer to API service
echo "Copying ML files to API service..."
cp services/anomaly-detector/ml_analyzer.py services/api/
cp services/anomaly-detector/simple_embeddings.py services/api/

# Fix 2: Create placeholder expert_rules.json
echo "Creating placeholder expert_rules.json..."
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
else
    echo "expert_rules.json already exists, skipping"
fi

# Fix 3: Modify main.py to handle missing models gracefully
echo "Updating main.py to handle missing models gracefully..."

# Direct file modification instead of using patch
MAIN_PY="services/anomaly-detector/main.py"

# Check if file exists
if [ ! -f "$MAIN_PY" ]; then
    echo "Error: $MAIN_PY not found!"
    exit 1
fi

# Back up the original file
cp "$MAIN_PY" "${MAIN_PY}.bak"
echo "Created backup of $MAIN_PY as ${MAIN_PY}.bak"

# Change error to warning for model loading
sed -i.sed_bak 's/logger.error(f"Error loading models: {str(e)}")/logger.warning(f"Error loading models: {str(e)}. Will train new models.")/' "$MAIN_PY"

# Add directory creation code - this is a bit more complex and requires identifying the right spot
MODEL_DIR_LINE=$(grep -n "model_dir = \"/app/models\"" "$MAIN_PY" | cut -d: -f1)
if [ -n "$MODEL_DIR_LINE" ]; then
    # Get the line after model_dir definition
    NEXT_LINE=$((MODEL_DIR_LINE + 1))
    # Insert the new line after model_dir definition
    sed -i.sed_bak2 "${NEXT_LINE}i\\        # Create model directory if it doesn't exist\\n        os.makedirs(model_dir, exist_ok=True)" "$MAIN_PY"
    echo "Added model directory creation code"
else
    echo "Warning: Could not find model_dir line to patch. Manual update may be needed."
fi

# Clean up backup files
rm -f "${MAIN_PY}.sed_bak" "${MAIN_PY}.sed_bak2"

echo ""
echo "Fixes applied. Now rebuild and restart the services:"
echo ""
echo "docker-compose build api anomaly-detector"
echo "docker-compose up -d"
echo ""
echo "This will resolve both the API import error and the model loading error."
