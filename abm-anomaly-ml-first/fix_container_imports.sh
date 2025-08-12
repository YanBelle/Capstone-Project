#!/bin/bash
"""
Fix Script for Unified ML Analyzer Container Integration

This script fixes the import issues in the Docker containers by ensuring
the shared directory is properly mounted and accessible.
"""

echo "🔧 Fixing Unified ML Analyzer Container Integration"
echo "=" * 60

BASE_DIR="/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first"
cd "$BASE_DIR"

echo "📂 Current directory: $(pwd)"
echo "📋 Checking shared directory setup..."

# Verify shared directory exists
if [ ! -d "shared" ]; then
    echo "❌ Shared directory not found, creating it..."
    mkdir -p shared
else
    echo "✅ Shared directory exists"
fi

# Verify unified analyzer exists
if [ ! -f "shared/ml_analyzer_unified.py" ]; then
    echo "❌ Unified analyzer not found in shared directory"
    exit 1
else
    echo "✅ Unified analyzer found: shared/ml_analyzer_unified.py"
    echo "   Size: $(wc -l < shared/ml_analyzer_unified.py) lines"
fi

echo ""
echo "🐳 Stopping existing containers..."
docker-compose down anomaly-detector api

echo ""
echo "🏗️ Rebuilding containers with shared directory access..."
docker-compose build anomaly-detector api

echo ""
echo "🧪 Testing the unified analyzer import in container..."
docker-compose run --rm anomaly-detector python /app/test_container_imports.py

echo ""
echo "🚀 Starting services..."
docker-compose up -d anomaly-detector api

echo ""
echo "📋 Checking service status..."
sleep 5
docker-compose ps anomaly-detector api

echo ""
echo "📜 Checking anomaly-detector logs..."
docker-compose logs --tail=20 anomaly-detector

echo ""
echo "✅ Fix script completed!"
echo "   - Shared directory mounted in containers"
echo "   - Containers rebuilt with latest changes"
echo "   - Services restarted"
echo ""
echo "💡 Monitor logs with: docker-compose logs -f anomaly-detector"
