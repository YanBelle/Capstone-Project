#!/bin/bash
# fix_digitalocean_deployment.sh
# Fix the nginx configuration and redeploy to DigitalOcean

echo "🔧 Fixing BERT-Enhanced DeepLog Deployment Issues"
echo "================================================="

echo "📋 Issues to fix:"
echo "  1. Nginx proxy configuration causing /api/api/ double prefix"
echo "  2. Missing EJ session data on DigitalOcean server"
echo "  3. Need to redeploy with fixes"

echo -e "\n🚀 Creating deployment commands for DigitalOcean server..."

# Create the commands to run on DigitalOcean
cat << 'EOF' > deploy_fixes.sh
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
EOF

chmod +x deploy_fixes.sh

echo "✅ Created deploy_fixes.sh"

echo -e "\n📋 Steps to fix the deployment:"
echo "============================================="

echo "1. First, push the nginx fix to your repository:"
echo "   git add nginx/default.conf"
echo "   git commit -m 'Fix nginx proxy configuration - remove double /api/ prefix'"
echo "   git push origin main"

echo -e "\n2. Copy your processed EJ session data to DigitalOcean:"
echo "   scp -r data/processed/ root@64.227.16.180:/root/Capstone-Project/abm-anomaly-ml-first/data/"

echo -e "\n3. SSH to your DigitalOcean server and run the fix script:"
echo "   ssh root@64.227.16.180"
echo "   cd /root/Capstone-Project/abm-anomaly-ml-first"
echo "   # Copy the deploy_fixes.sh script content and run it"

echo -e "\n4. Alternative: Copy and paste this one-liner to DigitalOcean:"

cat << 'EOF'

# === COPY THIS ENTIRE BLOCK TO DIGITALOCEAN ===
cd /root/Capstone-Project/abm-anomaly-ml-first && \
git pull origin main && \
docker-compose down && \
docker-compose build --no-cache nginx && \
docker-compose up -d && \
sleep 30 && \
echo "Testing API..." && \
curl -s http://localhost/api/v1/bert-deeplog/model-info | head -100
# === END COPY BLOCK ===

EOF

echo -e "\n🎯 Expected Results After Fix:"
echo "==============================="
echo "✅ Dashboard loads without 'Failed to load model information'"
echo "✅ API endpoints return data instead of 404 errors"
echo "✅ /api/v1/bert-deeplog/model-info works correctly"
echo "⚠️  EJ sessions may still show 500 error until data is copied"

echo -e "\n📍 Once deployed, test at:"
echo "🧠 http://64.227.16.180/dashboard/deeplog"
echo "🔧 http://64.227.16.180/api/v1/bert-deeplog/model-info"
