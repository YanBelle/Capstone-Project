#!/bin/bash

echo "Starting Enhanced Ensemble Dashboard Backend..."
echo "=============================================="

cd backend

echo "Checking Python version..."
python3 --version

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo "Starting FastAPI server..."
echo "Backend will be available at: http://localhost:8001"
echo "API docs will be at: http://localhost:8001/docs"
echo ""
echo "Starting server (press Ctrl+C to stop)..."

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
