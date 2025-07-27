#!/bin/bash
echo "📊 Development Environment Status"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml ps
echo ""
echo "🌐 Access URLs:"
echo "   Dashboard: http://64.227.16.180:3001"
echo "   API: http://64.227.16.180:8001"
echo "   API Docs: http://64.227.16.180:8001/docs"
