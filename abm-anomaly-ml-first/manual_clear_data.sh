#!/bin/bash
# Manual clear data script using Docker psql

echo "🔧 Manual Clear Data with Foreign Key Handling"
echo "=============================================="

# Function to run SQL in the database
run_sql() {
    docker compose exec -T postgres psql -U abmuser -d abmdb -c "$1" 2>/dev/null
}

echo "📊 BEFORE: Checking current data counts"
echo "----------------------------------------"
run_sql "SELECT 
    (SELECT COUNT(*) FROM ml_sessions) as sessions,
    (SELECT COUNT(*) FROM ml_anomalies) as anomalies,
    (SELECT COUNT(*) FROM expert_feedback) as feedback,
    (SELECT COUNT(*) FROM labeled_anomalies) as labeled,
    (SELECT COUNT(*) FROM anomaly_detections) as detections,
    (SELECT COUNT(*) FROM ml_summaries) as summaries;"

echo ""
echo "🧽 CLEARING: Deleting in dependency order"  
echo "------------------------------------------"

# Clear child tables first
echo "   Clearing ml_anomalies..."
result1=$(run_sql "DELETE FROM ml_anomalies; SELECT 'ml_anomalies cleared: ' || ROW_COUNT();")
echo "   $result1"

echo "   Clearing expert_feedback..."
result2=$(run_sql "DELETE FROM expert_feedback; SELECT 'expert_feedback cleared';")
echo "   $result2"

echo "   Clearing labeled_anomalies..."
result3=$(run_sql "DELETE FROM labeled_anomalies; SELECT 'labeled_anomalies cleared';")
echo "   $result3"

echo "   Clearing anomaly_detections..."
result4=$(run_sql "DELETE FROM anomaly_detections; SELECT 'anomaly_detections cleared';")
echo "   $result4"

echo "   Clearing ml_summaries..."
result5=$(run_sql "DELETE FROM ml_summaries; SELECT 'ml_summaries cleared';")
echo "   $result5"

# Clear parent table last
echo "   Clearing ml_sessions..."
result6=$(run_sql "DELETE FROM ml_sessions; SELECT 'ml_sessions cleared';")
echo "   $result6"

echo ""
echo "📊 AFTER: Checking final data counts"
echo "-------------------------------------"
run_sql "SELECT 
    (SELECT COUNT(*) FROM ml_sessions) as sessions,
    (SELECT COUNT(*) FROM ml_anomalies) as anomalies,
    (SELECT COUNT(*) FROM expert_feedback) as feedback,
    (SELECT COUNT(*) FROM labeled_anomalies) as labeled,
    (SELECT COUNT(*) FROM anomaly_detections) as detections,
    (SELECT COUNT(*) FROM ml_summaries) as summaries;"

echo ""
echo "✅ Manual clear data completed!"
