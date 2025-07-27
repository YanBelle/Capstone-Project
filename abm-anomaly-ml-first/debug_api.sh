#!/bin/bash

echo "🔧 Debugging API Connection Issues"
echo "=================================="

echo ""
echo "1. Checking Docker containers status..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "2. Checking specific service status..."
docker compose ps

echo ""
echo "3. Testing API directly (bypass nginx)..."
echo "Testing: curl http://localhost:8000/api/v1/continuous-learning/status"
curl -s http://localhost:8000/api/v1/continuous-learning/status 2>&1 || echo "Direct API connection failed"

echo ""
echo "4. Testing through nginx..."
echo "Testing: curl http://localhost/api/v1/continuous-learning/status"
curl -s http://localhost/api/v1/continuous-learning/status 2>&1 || echo "Nginx proxy connection failed"

echo ""
echo "5. Testing supervised training directly..."
echo "Testing: curl -X POST http://localhost:8000/api/v1/expert/train-supervised"
curl -s -X POST http://localhost:8000/api/v1/expert/train-supervised 2>&1 || echo "Direct supervised training failed"

echo ""
echo "6. API Logs (last 10 lines)..."
docker logs abm-ml-api --tail=10 2>/dev/null || echo "Could not fetch API logs"

echo ""
echo "7. Nginx Logs (last 5 lines)..."
docker logs abm-ml-nginx --tail=5 2>/dev/null || echo "Could not fetch Nginx logs"

echo ""
echo "8. Checking Supervised Model Results..."
echo "======================================="
echo "Model files created:"
ls -la ./data/models/supervised_classifier.pkl 2>/dev/null && echo "✓ Supervised classifier found" || echo "✗ No supervised classifier"
ls -la ./data/models/label_encoder.pkl 2>/dev/null && echo "✓ Label encoder found" || echo "✗ No label encoder"

echo ""
echo "Model metadata from database:"
docker exec abm-ml-postgres psql -U ml_user -d ml_anomaly_db -c "SELECT model_name, training_date, training_samples, performance_metrics FROM ml_models WHERE model_type='supervised_classifier' ORDER BY training_date DESC LIMIT 3;" 2>/dev/null || echo "Could not fetch model metadata"

echo ""
echo "Diagnosis complete!"
