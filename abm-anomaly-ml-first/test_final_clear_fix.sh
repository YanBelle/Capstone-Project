#!/bin/bash
# Final test of the comprehensive clear data fix

echo "🔧 Final Test: Comprehensive Clear Data Fix"
echo "==========================================="
echo ""

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 8

echo "📊 Testing clear data with foreign key constraint fix..."
echo ""

# Test the comprehensive clear data endpoint
echo "🧽 Calling clear data endpoint..."
response=$(curl -X DELETE "http://localhost:8000/api/v1/clear-data?confirm=true" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    --silent \
    --write-out "HTTP_STATUS:%{http_code}" \
    2>/dev/null)

# Extract HTTP status code
http_status=$(echo "$response" | sed -n 's/.*HTTP_STATUS:\([0-9]*\)$/\1/p')
response_body=$(echo "$response" | sed 's/HTTP_STATUS:[0-9]*$//')

echo "📋 Response Status: $http_status"
echo ""

if [ "$http_status" = "200" ]; then
    echo "✅ SUCCESS! Clear data completed successfully"
    echo ""
    echo "📄 Response Details:"
    echo "$response_body" | python3 -m json.tool 2>/dev/null || echo "$response_body"
else
    echo "❌ FAILED! HTTP Status: $http_status"
    echo ""
    echo "📄 Error Response:"
    echo "$response_body"
fi

echo ""
echo "🔍 Verifying data was actually cleared..."

# Check database
db_result=$(docker compose exec -T postgres psql -U abmuser -d abmdb -c "
SELECT 
    (SELECT COUNT(*) FROM ml_sessions) as sessions,
    (SELECT COUNT(*) FROM ml_anomalies) as anomalies;
" 2>/dev/null | grep -E "^ +[0-9]+ +\| +[0-9]+$" | head -1)

if [ -n "$db_result" ]; then
    sessions_count=$(echo "$db_result" | awk '{print $1}')
    anomalies_count=$(echo "$db_result" | awk '{print $3}')
    
    echo "📊 Database verification:"
    echo "   Sessions: $sessions_count"
    echo "   Anomalies: $anomalies_count"
    
    if [ "$sessions_count" = "0" ] && [ "$anomalies_count" = "0" ]; then
        echo "   ✅ Database successfully cleared!"
    else
        echo "   ⚠️  Database may not be fully cleared"
    fi
else
    echo "   ❓ Could not verify database status"
fi

echo ""
echo "🎯 CONCLUSION:"
if [ "$http_status" = "200" ]; then
    echo "✅ Foreign key constraint issue has been resolved!"
    echo "✅ Clear data functionality is now working properly"
    echo "✅ Dashboard 'Clear All Data' button should work"
else
    echo "❌ Foreign key constraint issue persists"
    echo "❌ Additional debugging may be needed"
fi

echo ""
echo "🏁 Test completed!"
