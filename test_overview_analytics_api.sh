#!/bin/bash

# Test script for Overview and Analytics API endpoints
echo "Testing Overview and Analytics API endpoints..."

API_URL=${API_URL:-"http://localhost:8000"}

echo "🔍 Testing Overview Stats endpoint..."
curl -s -X GET "$API_URL/api/v1/overview/stats" | jq '.' || echo "Overview endpoint failed or jq not available"

echo ""
echo "🔍 Testing Analytics Data endpoint..."
curl -s -X GET "$API_URL/api/v1/analytics/data" | jq '.' || echo "Analytics endpoint failed or jq not available"

echo ""
echo "🔍 Testing Analytics with timeframe parameter..."
curl -s -X GET "$API_URL/api/v1/analytics/data?timeframe=24h" | jq '.' || echo "Analytics with timeframe failed or jq not available"

echo ""
echo "✅ API endpoint testing complete!"
