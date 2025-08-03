#!/bin/bash
echo "🔧 Applying fixes to BERT-Enhanced DeepLog deployment..."

# Navigate to project directory
cd /root/Capstone-Project/abm-anomaly-ml-first

# Pull latest changes (including nginx fix)
echo "📥 Pulling latest changes..."
git pull origin main

# Copy processed data if it doesn't exist
echo "📁 Checking for processed EJ session data..."
if [ ! -f "data/processed/normal_sessions_full_20250803_102920.json" ]; then
    echo "⚠️  EJ session data not found - creating sample data directory..."
    mkdir -p data/processed
    # Note: You'll need to copy the processed data from your local machine
    echo "❗ Please copy your processed EJ session files to data/processed/"
    echo "   Files needed:"
    echo "   - normal_sessions_full_*.json"
    echo "   - error_sessions_full_*.json"
fi

# Rebuild and restart services with fixed nginx config
echo "🔨 Rebuilding services with nginx fix..."
docker-compose down
docker-compose build --no-cache nginx  # Force rebuild nginx with new config
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 45

# Test the fix
echo "🧪 Testing API endpoints..."
echo "Health check:"
curl -s http://localhost/health

echo -e "\nModel info (should work now):"
curl -s http://localhost/api/v1/bert-deeplog/model-info | head -200

echo -e "\nEJ Sessions (may need data copied):"
curl -s http://localhost/api/v1/bert-deeplog/load-ej-sessions?limit=1 | head -200

echo -e "\n✅ Deployment fixes applied!"
echo "🎯 Test your dashboard at: http://64.227.16.180/dashboard/deeplog"
