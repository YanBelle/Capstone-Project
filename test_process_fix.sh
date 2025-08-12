#!/bin/bash

# Test script for process input functionality

echo "=== Testing Process Input Functionality ==="

# Create test directories and files
echo "Creating test directories..."
mkdir -p /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input

# Create sample EJ file
echo "Creating sample EJ file..."
cat > /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/test_session.txt << EOF
2025-08-11 10:00:00,ATM001,DEPOSIT,100.00,SUCCESS
2025-08-11 10:05:00,ATM001,WITHDRAW,50.00,SUCCESS
2025-08-11 10:10:00,ATM001,BALANCE_INQUIRY,0.00,SUCCESS
2025-08-11 10:15:00,ATM001,DEPOSIT,200.00,SUCCESS
2025-08-11 10:20:00,ATM001,WITHDRAW,75.00,SUCCESS
EOF

echo "Test file created. Contents:"
cat /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/test_session.txt

# Rebuild container
echo "Rebuilding API container..."
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first
docker compose build --no-cache api

# Restart container
echo "Restarting API container..."
docker compose restart api

# Wait for container to be ready
echo "Waiting for container to be ready..."
sleep 15

# Test debug endpoint
echo "Testing debug file status endpoint..."
curl -s http://localhost/api/v1/debug/file-status | python3 -m json.tool

echo ""
echo "Testing process input endpoint..."
curl -s -X POST http://localhost/api/v1/process-input | python3 -m json.tool

echo ""
echo "Testing debug file status after processing..."
curl -s http://localhost/api/v1/debug/file-status | python3 -m json.tool

echo ""
echo "Checking container logs..."
docker logs abm-ml-api --tail 20

echo "=== Test Complete ==="
