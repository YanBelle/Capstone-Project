#!/bin/bash

# BERT-DeepLog System Startup and Test Script
# This script starts the system and runs comprehensive tests

set -e  # Exit on any error

echo "🚀 BERT-Enhanced DeepLog System Startup"
echo "========================================"

# Configuration
PROJECT_ROOT="/home/yc/development/Capstone-Project/abm-anomaly-ml-first"
VENV_PATH="$PROJECT_ROOT/venv"
API_URL="http://localhost:8000"
DASHBOARD_URL="http://localhost:3000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/bert_deeplog_model.py" ]; then
    error "BERT-DeepLog model not found. Are you in the correct directory?"
    exit 1
fi

cd "$PROJECT_ROOT"

# Function to check if a service is running
check_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking if $service_name is running at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            success "$service_name is running!"
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            log "Waiting for $service_name to start..."
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "$service_name failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Function to install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        success "Virtual environment activated"
    else
        warning "No virtual environment found. Using system Python."
    fi
    
    # Install required packages
    pip install -q torch transformers scikit-learn numpy pandas matplotlib seaborn
    pip install -q fastapi uvicorn pydantic python-multipart
    pip install -q requests aiohttp asyncio
    
    success "Dependencies installed"
}

# Function to start the API service
start_api() {
    log "Starting BERT-DeepLog API service..."
    
    # Check if already running
    if curl -f -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
        success "API service is already running"
        return 0
    fi
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Start the API in background
    nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > api.log 2>&1 &
    API_PID=$!
    
    log "API started with PID: $API_PID"
    echo $API_PID > api.pid
    
    # Wait for API to be ready
    check_service "$API_URL/api/v1/health" "API Service"
}

# Function to start the dashboard
start_dashboard() {
    log "Starting React dashboard..."
    
    # Check if dashboard directory exists
    if [ ! -d "services/dashboard" ]; then
        warning "Dashboard directory not found. Skipping dashboard startup."
        return 0
    fi
    
    cd services/dashboard
    
    # Check if already running
    if curl -f -s "$DASHBOARD_URL" > /dev/null 2>&1; then
        success "Dashboard is already running"
        cd "$PROJECT_ROOT"
        return 0
    fi
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log "Installing dashboard dependencies..."
        npm install
    fi
    
    # Start dashboard in background
    nohup npm start > ../../dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    
    log "Dashboard started with PID: $DASHBOARD_PID"
    echo $DASHBOARD_PID > ../../dashboard.pid
    
    cd "$PROJECT_ROOT"
    
    # Wait for dashboard to be ready
    check_service "$DASHBOARD_URL" "Dashboard"
}

# Function to run basic system tests
run_tests() {
    log "Running BERT-DeepLog system tests..."
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Run the test suite
    python test_bert_deeplog_system.py --api-url "$API_URL"
    
    if [ $? -eq 0 ]; then
        success "All tests passed!"
    else
        warning "Some tests failed. Check test output above."
    fi
}

# Function to show system status  
show_status() {
    echo ""
    echo "🎯 BERT-DeepLog System Status"
    echo "============================="
    
    # Check API
    if curl -f -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
        echo -e "API Service:   ${GREEN}✅ Running${NC} ($API_URL)"
    else
        echo -e "API Service:   ${RED}❌ Not Running${NC}"
    fi
    
    # Check Dashboard
    if curl -f -s "$DASHBOARD_URL" > /dev/null 2>&1; then
        echo -e "Dashboard:     ${GREEN}✅ Running${NC} ($DASHBOARD_URL)"
    else
        echo -e "Dashboard:     ${RED}❌ Not Running${NC}"
    fi
    
    # Check model files
    if [ -f "$PROJECT_ROOT/bert_deeplog_model.py" ]; then
        echo -e "Model Code:    ${GREEN}✅ Available${NC}"
    else
        echo -e "Model Code:    ${RED}❌ Missing${NC}"
    fi
    
    if [ -f "$PROJECT_ROOT/bert_deeplog_api.py" ]; then
        echo -e "API Code:      ${GREEN}✅ Available${NC}"
    else
        echo -e "API Code:      ${RED}❌ Missing${NC}"
    fi
    
    echo ""
    echo "📊 Access Points:"
    echo "  • API Documentation: $API_URL/docs"
    echo "  • DeepLog Dashboard: $DASHBOARD_URL/dashboard/deeplog"
    echo "  • Main Dashboard: $DASHBOARD_URL/dashboard"
    echo ""
}

# Function to stop services
stop_services() {
    log "Stopping services..."
    
    # Stop API
    if [ -f "api.pid" ]; then
        API_PID=$(cat api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            log "API service stopped (PID: $API_PID)"
        fi
        rm -f api.pid
    fi
    
    # Stop Dashboard  
    if [ -f "dashboard.pid" ]; then
        DASHBOARD_PID=$(cat dashboard.pid)
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            kill $DASHBOARD_PID
            log "Dashboard stopped (PID: $DASHBOARD_PID)"
        fi
        rm -f dashboard.pid
    fi
    
    success "Services stopped"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all services (default)"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  test      - Run system tests only"
    echo "  status    - Show system status"
    echo "  logs      - Show service logs"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start API and dashboard"
    echo "  $0 test     # Run comprehensive tests"
    echo "  $0 status   # Check what's running"
}

# Function to show logs
show_logs() {
    echo "📜 Service Logs"
    echo "==============="
    
    if [ -f "api.log" ]; then
        echo ""
        echo "API Logs (last 20 lines):"
        echo "-------------------------"
        tail -n 20 api.log
    fi
    
    if [ -f "dashboard.log" ]; then
        echo ""
        echo "Dashboard Logs (last 20 lines):"
        echo "-------------------------------"
        tail -n 20 dashboard.log
    fi
}

# Main execution
case "${1:-start}" in
    "start")
        log "Starting BERT-DeepLog system..."
        install_dependencies
        start_api
        start_dashboard
        show_status
        log "System startup complete!"
        echo ""
        echo "🎉 BERT-DeepLog system is ready!"
        echo "   • Visit $DASHBOARD_URL/dashboard/deeplog to access the DeepLog dashboard"
        echo "   • Visit $API_URL/docs for API documentation"
        echo ""
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart") 
        stop_services
        sleep 2
        install_dependencies
        start_api
        start_dashboard
        show_status
        ;;
    
    "test")
        log "Running tests only..."
        run_tests
        ;;
    
    "status")
        show_status
        ;;
    
    "logs")
        show_logs
        ;;
    
    "help"|"-h"|"--help")
        show_usage
        ;;
    
    *)
        error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

# If starting or restarting, run tests at the end
if [[ "${1:-start}" == "start" ]] || [[ "$1" == "restart" ]]; then
    echo ""
    read -p "🧪 Run system tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
fi
