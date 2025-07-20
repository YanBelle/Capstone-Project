#!/bin/bash
# Fix Retraining Button Functionality
# This script addresses all issues preventing the retraining button from working

echo "🔧 ABM ML-First Anomaly Detection - Retraining Fix"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

echo "📍 Current directory: $(pwd)"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down

# Remove existing API container to force rebuild
echo "🗑️ Removing existing API container..."
docker rmi abm-anomaly-ml-first-api 2>/dev/null || true

# Build the API service with updated dependencies
echo "🏗️ Building API service with ML dependencies..."
docker compose build api

# Start all services
echo "🚀 Starting all services..."
docker compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "📊 Checking service status..."
docker compose ps

# Test API health
echo "🩺 Testing API health..."
API_HEALTH=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8000/api/v1/health)
if [ "$API_HEALTH" = "200" ]; then
    echo "✅ API is responding (HTTP 200)"
else
    echo "❌ API is not responding (HTTP $API_HEALTH)"
    echo "📋 API Logs:"
    docker compose logs api --tail=20
fi

# Test retraining endpoint
echo "🔄 Testing retraining endpoint..."
RETRAINING_RESPONSE=$(curl -s -w "%{http_code}" -X POST http://localhost:8000/api/v1/continuous-learning/trigger-retraining -H "Content-Type: application/json")
RETRAINING_STATUS=$(echo "$RETRAINING_RESPONSE" | tail -c 4)

if [ "$RETRAINING_STATUS" = "200" ]; then
    echo "✅ Retraining endpoint is working!"
    echo "🎉 You can now use the 'Trigger Retraining' button in the dashboard"
else
    echo "❌ Retraining endpoint failed (HTTP $RETRAINING_STATUS)"
    echo "📋 API Logs:"
    docker compose logs api --tail=20
fi

# Show access URLs
echo ""
echo "🌐 Access URLs:"
echo "   Dashboard: http://localhost:3000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Grafana:   http://localhost:3001"
echo ""

# Show next steps
echo "📋 Next Steps:"
echo "1. Open your browser and go to http://localhost:3000"
echo "2. Click on the 'ML Training' tab"
echo "3. Click 'Trigger Retraining' button"
echo "4. The retraining should now work properly"
echo ""
echo "If you still have issues, check the logs:"
echo "   docker compose logs api -f"
echo "   docker compose logs anomaly-detector -f"

echo "✅ Retraining fix complete!"
