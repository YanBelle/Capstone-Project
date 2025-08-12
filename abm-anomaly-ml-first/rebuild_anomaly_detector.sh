#!/bin/bash
# Force rebuild of the anomaly detector Docker container to fix EJLogLabeler issue

echo "Rebuilding anomaly detector container to fix EJLogLabeler method error..."

# Navigate to the ML-first directory
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first

# Stop the current container
echo "Stopping anomaly detector container..."
docker-compose stop anomaly-detector

# Remove the old container and image to force rebuild
echo "Removing old container and image..."
docker-compose rm -f anomaly-detector
docker rmi abm-anomaly-ml-first_anomaly-detector 2>/dev/null || true

# Clear any Python bytecode cache that might be persisted in volumes
echo "Clearing Python cache..."
docker run --rm -v "$(pwd)/services/anomaly-detector:/app" python:3.10-slim \
    find /app -name "*.pyc" -delete 2>/dev/null || true
docker run --rm -v "$(pwd)/services/anomaly-detector:/app" python:3.10-slim \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Rebuild and start the container
echo "Rebuilding and starting anomaly detector..."
docker-compose build --no-cache anomaly-detector
docker-compose up -d anomaly-detector

echo "Done! Check logs with: docker-compose logs -f anomaly-detector"
