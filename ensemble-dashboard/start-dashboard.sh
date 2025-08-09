#!/bin/bash

# Ensemble Anomaly Detection Dashboard Startup Script

echo "🎯 Starting Ensemble Anomaly Detection Dashboard..."
echo "================================================"

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker and Docker Compose found"
    echo "🚀 Starting services with Docker..."
    
    # Navigate to dashboard directory
    cd "$(dirname "$0")"
    
    # Stop any existing containers
    docker-compose down
    
    # Build and start the services
    docker-compose up --build -d
    
    echo ""
    echo "🎉 Dashboard started successfully!"
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8001"
    echo "📚 API Docs: http://localhost:8001/docs"
    echo ""
    echo "To stop the dashboard, run: docker-compose down"
    echo "To view logs, run: docker-compose logs -f"
    
else
    echo "⚠️  Docker not found. Using manual startup..."
    echo ""
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        echo "🐍 Starting Backend..."
        cd backend
        
        # Install requirements if venv exists
        if [ -d "venv" ]; then
            source venv/bin/activate
        else
            echo "Creating virtual environment..."
            python3 -m venv venv
            source venv/bin/activate
        fi
        
        pip install -r requirements.txt
        
        # Start backend in background
        nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload > backend.log 2>&1 &
        BACKEND_PID=$!
        echo "✅ Backend started (PID: $BACKEND_PID)"
        
        cd ..
    else
        echo "❌ Python3 not found. Please install Python3 to run the backend."
        exit 1
    fi
    
    # Check if Node.js is available
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        echo "📦 Starting Frontend..."
        cd frontend
        
        # Install dependencies
        if [ ! -d "node_modules" ]; then
            echo "Installing Node.js dependencies..."
            npm install
        fi
        
        # Start frontend
        npm start &
        FRONTEND_PID=$!
        echo "✅ Frontend started (PID: $FRONTEND_PID)"
        
        cd ..
        
        echo ""
        echo "🎉 Dashboard started successfully!"
        echo "📊 Frontend: http://localhost:3000"
        echo "🔧 Backend API: http://localhost:8001"
        echo "📚 API Docs: http://localhost:8001/docs"
        echo ""
        echo "Backend PID: $BACKEND_PID"
        echo "Frontend PID: $FRONTEND_PID"
        echo ""
        echo "To stop the services manually:"
        echo "kill $BACKEND_PID $FRONTEND_PID"
        
    else
        echo "❌ Node.js/npm not found. Please install Node.js to run the frontend."
        if [ ! -z "$BACKEND_PID" ]; then
            kill $BACKEND_PID
        fi
        exit 1
    fi
fi

echo ""
echo "🔍 Waiting for services to be ready..."
sleep 5

# Test backend health
if curl -s http://localhost:8001/api/model_info > /dev/null; then
    echo "✅ Backend is responding"
else
    echo "⚠️  Backend may still be starting up. Check http://localhost:8001/docs"
fi

echo ""
echo "🎯 Ensemble Anomaly Detection Dashboard is ready!"
echo "Happy anomaly hunting! 🕵️‍♂️"
