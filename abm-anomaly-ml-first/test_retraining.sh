#!/bin/bash

echo "🔧 Testing Continuous Learning API Endpoints"
echo "=============================================="

echo ""
echo "1. Testing Status Endpoint..."
response=$(curl -s http://localhost/api/v1/continuous-learning/status)
echo "Response: $response"

echo ""
echo "2. Testing Retraining Trigger..."
response=$(curl -s -X POST http://localhost/api/v1/continuous-learning/trigger-retraining -H "Content-Type: application/json")
echo "Response: $response"

echo ""
echo "3. Checking API logs for retraining activity..."
docker logs abm-ml-api --tail=20 | grep -i "retrain\|training\|continuous\|Starting manual\|completed\|Isolation Forest\|SVM"

echo ""
echo "4. Testing Train Supervised Model..."
response=$(curl -s -X POST http://localhost/api/v1/expert/train-supervised)
echo "Response: $response"

echo ""
echo "Done! Check the logs above for retraining activity."
