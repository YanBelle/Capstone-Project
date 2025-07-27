#!/bin/bash

echo "🔍 Monitoring for ML Retraining Activity..."
echo "==============================================="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor Docker logs for retraining-related activity
docker-compose logs -f --tail=50 2>/dev/null | grep -i --line-buffered "retrain\|training\|continuous\|Starting manual\|completed\|Isolation Forest\|One-Class SVM\|supervised classifier\|feedback\|threshold reached"
