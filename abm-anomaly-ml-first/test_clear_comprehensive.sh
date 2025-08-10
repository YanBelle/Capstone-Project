#!/bin/bash
# Enhanced Clear Data Test Script

echo "🧪 Enhanced Clear Data Test"
echo "=========================="

# Check current data before clearing
echo "📊 BEFORE CLEARING:"
echo "-------------------"

echo "Database records:"
docker compose exec -T postgres psql -U abmuser -d abmdb -c "
SELECT 
    (SELECT COUNT(*) FROM ml_sessions) as sessions_count,
    (SELECT COUNT(*) FROM ml_anomalies) as anomalies_count;
" 2>/dev/null || echo "   Database check failed"

echo ""
echo "File system data:"
echo "   Sessions: $(docker compose exec api ls /app/data/sessions/ 2>/dev/null | wc -l || echo 0) files"
echo "   Output: $(docker compose exec api ls /app/data/output/ 2>/dev/null | wc -l || echo 0) files"
echo "   Models: $(docker compose exec api ls /app/data/models/ 2>/dev/null | wc -l || echo 0) files"

echo ""
echo "🧽 TESTING CLEAR DATA..."
echo "-------------------------"

# Test the clear data endpoint
curl -X DELETE "http://localhost:8000/api/v1/clear-data?confirm=true" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    --silent \
    --show-error \
    2>/dev/null

echo ""
echo ""
echo "📊 AFTER CLEARING:"
echo "------------------"

# Check data after clearing
echo "Database records:"
docker compose exec -T postgres psql -U abmuser -d abmdb -c "
SELECT 
    (SELECT COUNT(*) FROM ml_sessions) as sessions_count,
    (SELECT COUNT(*) FROM ml_anomalies) as anomalies_count;
" 2>/dev/null || echo "   Database check failed"

echo ""
echo "File system data:"
echo "   Sessions: $(docker compose exec api ls /app/data/sessions/ 2>/dev/null | wc -l || echo 0) files"
echo "   Output: $(docker compose exec api ls /app/data/output/ 2>/dev/null | wc -l || echo 0) files"  
echo "   Models: $(docker compose exec api ls /app/data/models/ 2>/dev/null | wc -l || echo 0) files"

echo ""
echo "✅ Test completed!"
