#!/bin/bash

echo "🔍 Debugging 502 Bad Gateway Error"
echo "=================================="

echo ""
echo "1. Checking Docker services status..."
docker-compose ps

echo ""
echo "2. Checking if API container is running..."
docker ps | grep api || echo "❌ No API container found"

echo ""
echo "3. Checking API container logs..."
docker logs abm-ml-api --tail=20 2>/dev/null || echo "❌ Could not get API logs"

echo ""
echo "4. Checking nginx container logs..."
docker logs abm-ml-nginx --tail=20 2>/dev/null || echo "❌ Could not get nginx logs"

echo ""
echo "5. Testing direct API connection (bypassing nginx)..."
curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo "❌ Direct API connection failed"

echo ""
echo "6. Testing nginx connection..."
curl -s http://localhost/api/v1/health 2>/dev/null || echo "❌ Nginx connection failed"

echo ""
echo "7. Checking for port conflicts..."
lsof -i :80 2>/dev/null || echo "No process on port 80"
lsof -i :8000 2>/dev/null || echo "No process on port 8000"

echo ""
echo "🔧 RECOMMENDED FIXES:"
echo "===================="
echo "1. Restart services: docker-compose down && docker-compose up -d"
echo "2. Check API health: curl http://localhost:8000/api/v1/health"
echo "3. Check new endpoint: curl http://localhost:8000/api/v1/models/training-results"
echo "4. If working, try through nginx: curl http://localhost/api/v1/models/training-results"

echo ""
echo "💡 Quick Service Restart:"
echo "docker-compose down"
echo "docker-compose up -d"
echo "sleep 10"
echo "curl http://localhost/api/v1/models/training-results"
