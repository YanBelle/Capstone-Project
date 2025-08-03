#!/bin/bash
# FINAL DEPLOYMENT COMMANDS FOR DIGITALOCEAN
# Copy these commands and run them on your DigitalOcean server

echo "🚀 BERT-Enhanced DeepLog - Final Deployment Fix"
echo "==============================================="

echo "📋 What this will fix:"
echo "  ✅ Remove /api/api/ double prefix causing 404 errors"
echo "  ✅ Update to latest code with all fixes"  
echo "  ✅ Restart services with proper configuration"

echo -e "\n🔧 COPY AND PASTE THESE COMMANDS TO YOUR DIGITALOCEAN SERVER:"
echo "=============================================================="

echo "# Step 1: Navigate to project and update code"
echo "cd /root/Capstone-Project/abm-anomaly-ml-first"
echo "git pull origin main"
echo ""

echo "# Step 2: Stop services and rebuild with nginx fix"
echo "docker-compose down"
echo "docker-compose build --no-cache nginx"
echo "docker-compose up -d"
echo ""

echo "# Step 3: Wait for services to start"
echo "sleep 45"
echo ""

echo "# Step 4: Test the fixes"
echo "echo 'Testing API endpoints...'"
echo "curl -s http://localhost/health"
echo "echo ''"
echo "curl -s http://localhost/api/v1/bert-deeplog/model-info | head -200"

echo -e "\n🎯 SINGLE COMMAND VERSION (copy this entire line):"
echo "=============================================================="
echo "cd /root/Capstone-Project/abm-anomaly-ml-first && git pull origin main && docker-compose down && docker-compose build --no-cache nginx && docker-compose up -d && sleep 45 && echo 'Testing...' && curl -s http://localhost/api/v1/bert-deeplog/model-info | head -100"

echo -e "\n📊 WHAT TO EXPECT AFTER RUNNING THE COMMANDS:"
echo "=============================================="
echo "✅ No more 404 errors for /api/v1/bert-deeplog/* endpoints"
echo "✅ Model info will load in the dashboard"
echo "✅ Training history will load"
echo "✅ Prediction cache will load"
echo "⚠️  EJ sessions may still show 500 error (needs data)"

echo -e "\n🧪 TO TEST THE FIX:"
echo "=================="
echo "1. Run the commands above on your DigitalOcean server"
echo "2. Wait for 'Testing...' to complete"
echo "3. Open: http://64.227.16.180/dashboard/deeplog"
echo "4. Check browser console - should see much fewer errors"
echo "5. The 'Failed to load model information' error should be gone"

echo -e "\n📁 OPTIONAL: Copy EJ Session Data (if you want training data):"
echo "=============================================================="
echo "# On your local machine, copy the data:"
echo "# scp ej_session_data.tar.gz root@64.227.16.180:/root/"
echo "# "
echo "# Then on DigitalOcean server:"
echo "# cd /root/Capstone-Project/abm-anomaly-ml-first"
echo "# tar -xzf /root/ej_session_data.tar.gz"

echo -e "\n🎉 After running these commands, your BERT-Enhanced DeepLog Dashboard"
echo "   should work correctly at http://64.227.16.180/dashboard/deeplog"
