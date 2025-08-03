#!/bin/bash
echo "🔍 Checking DigitalOcean Deployment Status..."

# Test basic connectivity
echo "Testing basic connectivity..."
if curl -s --connect-timeout 5 http://localhost/health > /dev/null; then
    echo "✅ Server is responding"
else
    echo "❌ Server not responding on port 80"
fi

# Check Docker services
echo -e "\nChecking Docker services..."
docker-compose ps

# Test API endpoints
echo -e "\nTesting API endpoints..."
curl -s http://localhost/api/v1/bert-deeplog/model-info | head -100

echo -e "\n🎯 If everything is working, access via:"
echo "   http://64.227.16.180/dashboard/deeplog"
