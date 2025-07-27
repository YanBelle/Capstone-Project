#!/bin/bash
echo "🛑 Stopping Development Environment"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml down
echo "✅ Development environment stopped"
