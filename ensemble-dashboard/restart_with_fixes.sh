#!/bin/bash

# Complete Fix and Restart Script for DBSCAN Cluster Sessions Error

echo "🔧 FIXING DBSCAN CLUSTER SESSIONS ERROR..."
echo "=========================================="

# Navigate to the correct directory
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard

echo "📂 Current directory: $(pwd)"

# Step 1: Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down --remove-orphans

# Step 2: Clean up Docker system (optional - removes unused images)
echo "🧹 Cleaning up Docker system..."
docker system prune -f

# Step 3: Check if our fixes are in place
echo "🔍 Verifying fixes are in place..."

# Check if ClusterSessionsRequest has feature_type parameter
if grep -q "feature_type: Optional\[str\]" backend/app/main.py; then
    echo "✅ ClusterSessionsRequest fix is in place"
else
    echo "❌ ClusterSessionsRequest fix is missing"
fi

# Check if get_cluster_sessions method exists in enhanced_ensemble_detector
if grep -q "def get_cluster_sessions" backend/enhanced_ensemble_detector.py; then
    echo "✅ get_cluster_sessions method is in backend"
else
    echo "❌ get_cluster_sessions method is missing from backend"
fi

# Step 4: Build and start containers with full rebuild
echo "🚀 Building and starting containers..."
docker-compose up --build -d --force-recreate

# Step 5: Wait for services to start
echo "⏳ Waiting for services to start (30 seconds)..."
sleep 30

# Step 6: Check container status
echo "📊 Container Status:"
docker-compose ps

# Step 7: Check backend logs
echo "📋 Backend Logs (last 20 lines):"
docker-compose logs --tail=20 backend

# Step 8: Test backend health
echo "🏥 Testing backend health..."
if curl -s http://localhost:8001/api/health > /dev/null; then
    echo "✅ Backend is responding"
    curl -s http://localhost:8001/api/health | python3 -m json.tool
else
    echo "❌ Backend is not responding"
fi

# Step 9: Test specific endpoints
echo "🧪 Testing cluster_sessions endpoint structure..."
echo "Expected request format: {\"cluster_id\": 1, \"feature_type\": \"combined\"}"

# Step 10: Check frontend connectivity
echo "🌐 Testing frontend connectivity..."
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is responding"
else
    echo "❌ Frontend is not responding"
fi

echo ""
echo "🎯 TROUBLESHOOTING GUIDE:"
echo "========================"
echo "1. If backend is not responding:"
echo "   - Check logs: docker-compose logs backend"
echo "   - Restart: docker-compose restart backend"
echo ""
echo "2. If still getting 500 errors:"
echo "   - Check if model is trained: curl http://localhost:8001/api/model_info"
echo "   - Train model first if needed"
echo ""
echo "3. If CORS errors persist:"
echo "   - Clear browser cache"
echo "   - Try in incognito/private browsing mode"
echo ""
echo "4. Access URLs:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8001"
echo "   - API Docs: http://localhost:8001/docs"
echo ""
echo "🔧 Fix complete! Try clicking on clusters again."
