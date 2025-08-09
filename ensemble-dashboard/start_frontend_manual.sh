#!/bin/bash

# Manual Frontend Startup Script
# Use this if Docker isn't working

echo "🌐 Starting Frontend Manually..."
echo "==============================="

cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard/frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Kill any existing processes on port 3000
echo "🛑 Stopping any existing processes on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start the frontend server
echo "🚀 Starting React frontend on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm start
