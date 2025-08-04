#!/bin/bash

# Manual Backend Startup Script
# Use this if Docker isn't working

echo "🚀 Starting Backend Manually..."
echo "==============================="

cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard/backend

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Kill any existing processes on port 8001
echo "🛑 Stopping any existing processes on port 8001..."
lsof -ti:8001 | xargs kill -9 2>/dev/null || true

# Start the backend server
echo "🚀 Starting FastAPI backend on http://localhost:8001"
echo "📚 API Documentation will be available at http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd app
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
