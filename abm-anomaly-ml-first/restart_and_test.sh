#!/bin/bash

echo "🚀 Starting ABM ML Anomaly Detection Services"
echo "============================================="

# Change to project directory
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first

echo ""
echo "1. Stopping existing services..."
docker-compose down

echo ""
echo "2. Starting services..."
docker-compose up -d

echo ""
echo "3. Waiting for services to start..."
sleep 15

echo ""
echo "4. Checking service status..."
docker-compose ps

echo ""
echo "5. Testing API health (direct)..."
curl -s http://localhost:8000/api/v1/health || echo "❌ Direct API health check failed"

echo ""
echo "6. Testing API health (through nginx)..."
curl -s http://localhost/api/v1/health || echo "❌ Nginx API health check failed"

echo ""
echo "7. Testing new training results endpoint (direct)..."
curl -s http://localhost:8000/api/v1/models/training-results || echo "❌ Direct training results endpoint failed"

echo ""
echo "8. Testing new training results endpoint (through nginx)..."
curl -s http://localhost/api/v1/models/training-results || echo "❌ Nginx training results endpoint failed"

echo ""
echo "📋 Service URLs:"
echo "==============="
echo "Health Check (direct): http://localhost:8000/api/v1/health"
echo "Health Check (nginx):  http://localhost/api/v1/health"
echo "Training Results:      http://localhost/api/v1/models/training-results"
echo "API Documentation:     http://localhost/api/docs"
echo "Dashboard:             http://localhost/"

echo ""
echo "📝 To check logs:"
echo "================"
echo "API logs:    docker logs abm-ml-api"
echo "Nginx logs:  docker logs abm-ml-nginx"
echo "All logs:    docker-compose logs"
