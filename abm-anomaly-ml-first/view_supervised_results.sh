#!/bin/bash

echo "📊 Viewing Supervised Learning Results"
echo "======================================"

echo ""
echo "🎯 METHOD 1: Dashboard Interface"
echo "-------------------------------"
echo "Go to: http://localhost/expert-labeling"
echo "Look for:"
echo "  • Training status indicators"
echo "  • Model performance metrics"
echo "  • Training history"

echo ""
echo "🎯 METHOD 2: API Endpoints"
echo "-------------------------"
echo "Model Status:"
echo "curl http://localhost/api/v1/models/status"
echo ""
echo "Training History:"
echo "curl http://localhost/api/v1/models/training-history"

echo ""
echo "🎯 METHOD 3: Database Query"
echo "---------------------------"
echo "View model metadata:"
echo 'docker exec abm-ml-postgres psql -U ml_user -d ml_anomaly_db -c "SELECT * FROM ml_models WHERE model_type='"'"'supervised_classifier'"'"' ORDER BY training_date DESC;"'

echo ""
echo "🎯 METHOD 4: Log Files"
echo "---------------------"
echo "Training logs:"
echo "docker logs abm-ml-api | grep -i 'supervised.*training'"
echo ""
echo "Training completion:"
echo "docker logs abm-ml-api | grep -i 'training completed'"

echo ""
echo "🎯 METHOD 5: Model Files"
echo "-----------------------"
echo "Check if models were created:"
echo "ls -la ./data/models/supervised_classifier.pkl"
echo "ls -la ./data/models/label_encoder.pkl"

echo ""
echo "🎯 CURRENT STATUS CHECK"
echo "======================"

echo ""
echo "1. Checking for model files..."
if [ -f "./data/models/supervised_classifier.pkl" ]; then
    echo "✓ Supervised classifier model exists"
    stat ./data/models/supervised_classifier.pkl
else
    echo "✗ No supervised classifier model found"
fi

if [ -f "./data/models/label_encoder.pkl" ]; then
    echo "✓ Label encoder exists"
    stat ./data/models/label_encoder.pkl
else
    echo "✗ No label encoder found"
fi

echo ""
echo "2. Checking database for training records..."
docker exec abm-ml-postgres psql -U ml_user -d ml_anomaly_db -c "
SELECT 
    model_name,
    training_date,
    training_samples,
    LEFT(performance_metrics::text, 100) as metrics_preview
FROM ml_models 
WHERE model_type='supervised_classifier' 
ORDER BY training_date DESC 
LIMIT 5;" 2>/dev/null || echo "Could not connect to database"

echo ""
echo "3. Recent training logs..."
docker logs abm-ml-api --tail=50 | grep -i "supervised\|training\|accuracy\|f1" | tail -10

echo ""
echo "🎯 DETAILED PERFORMANCE METRICS"
echo "==============================="
echo "To see full performance metrics, run:"
echo 'docker exec abm-ml-postgres psql -U ml_user -d ml_anomaly_db -c "SELECT performance_metrics FROM ml_models WHERE model_type='"'"'supervised_classifier'"'"' ORDER BY training_date DESC LIMIT 1;" | grep -v "performance_metrics" | grep -v "^-" | grep -v "^(" | jq .'

echo ""
echo "📈 TESTING THE TRAINED MODEL"
echo "============================"
echo "To test the trained model on new data:"
echo "1. Process new EJ logs through the system"
echo "2. Check predictions at: http://localhost/overview" 
echo "3. Compare unsupervised vs supervised predictions"

echo ""
echo "For real-time monitoring of model performance:"
echo "docker logs -f abm-ml-api | grep -i 'prediction\|classification\|accuracy'"
