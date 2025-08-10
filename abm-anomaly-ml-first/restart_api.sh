#!/bin/bash

echo "🔄 Restarting ABM API containers..."

# Navigate to the project directory
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first

# Stop all containers
echo "⏹️ Stopping containers..."
docker compose down

# Remove any hanging containers
echo "🧹 Cleaning up..."
docker container prune -f

# Start the containers 
echo "🚀 Starting containers..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check status
echo "📊 Container status:"
docker compose ps

# Check API health
echo "🔍 Checking API health..."
curl -s http://localhost:8000/api/health || echo "API not ready yet"

echo "✅ Restart complete!"
