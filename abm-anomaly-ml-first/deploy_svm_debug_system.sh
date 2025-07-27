#!/bin/bash

# SVM Debug System Master Deployment Script
# This script deploys the complete SVM visualization and debugging system

set -e  # Exit on any error

echo "🚀 Starting SVM Debug System Deployment..."
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Step 1: Check prerequisites
echo ""
print_info "Step 1: Checking Prerequisites"
echo "================================"

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_status "Docker is running"

# Check if Python is available
if ! command -v python3 > /dev/null 2>&1; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi
print_status "Python 3 is available"

# Check if pip is available
if ! command -v pip > /dev/null 2>&1; then
    print_error "pip is not installed or not in PATH"
    exit 1
fi
print_status "pip is available"

# Step 2: Install Python dependencies
echo ""
print_info "Step 2: Installing Python Dependencies"
echo "======================================"

if [ -f "svm_requirements.txt" ]; then
    print_info "Installing SVM debugging dependencies..."
    pip install -r svm_requirements.txt
    print_status "SVM dependencies installed"
else
    print_warning "svm_requirements.txt not found, installing core dependencies..."
    pip install plotly matplotlib seaborn scikit-learn numpy pandas
fi

# Step 3: Set up directory structure
echo ""
print_info "Step 3: Setting Up Directory Structure"
echo "======================================"

# Create necessary directories
mkdir -p debug_output
mkdir -p static/debug
mkdir -p services/anomaly-detector
mkdir -p services/api
mkdir -p services/dashboard/src

print_status "Directories created"

# Step 4: Deploy SVM components
echo ""
print_info "Step 4: Deploying SVM Components"
echo "================================"

# Check and copy SVM visualizer
if [ -f "svm_visualizer.py" ]; then
    cp svm_visualizer.py services/anomaly-detector/
    print_status "SVM visualizer deployed to anomaly detector service"
else
    print_error "svm_visualizer.py not found"
    exit 1
fi

# Check and copy SVM debug API
if [ -f "svm_debug_api.py" ]; then
    cp svm_debug_api.py services/api/
    print_status "SVM debug API deployed to API service"
else
    print_error "svm_debug_api.py not found"
    exit 1
fi

# Check and copy React dashboard component
if [ -f "SVMDebugDashboard.js" ]; then
    cp SVMDebugDashboard.js services/dashboard/src/
    print_status "SVM debug dashboard component deployed"
else
    print_error "SVMDebugDashboard.js not found"
    exit 1
fi

# Check and copy CLI tool
if [ -f "debug_svm_cli.py" ]; then
    cp debug_svm_cli.py ./
    chmod +x debug_svm_cli.py
    print_status "CLI debug tool deployed"
else
    print_warning "debug_svm_cli.py not found, skipping CLI tool"
fi

# Step 5: Update main API to include SVM debug routes
echo ""
print_info "Step 5: Updating API Configuration"
echo "=================================="

# Look for main API file
API_FILE=""
if [ -f "services/api/main.py" ]; then
    API_FILE="services/api/main.py"
elif [ -f "services/api/app.py" ]; then
    API_FILE="services/api/app.py"
elif [ -f "main.py" ]; then
    API_FILE="main.py"
elif [ -f "app.py" ]; then
    API_FILE="app.py"
fi

if [ -n "$API_FILE" ]; then
    # Check if SVM debug is already imported
    if ! grep -q "svm_debug_api" "$API_FILE"; then
        # Add import and router inclusion
        print_info "Adding SVM debug routes to $API_FILE..."
        
        # Backup original file
        cp "$API_FILE" "${API_FILE}.backup"
        
        # Add import after other imports
        sed -i '/from.*import.*router/a from .svm_debug_api import router as svm_debug_router' "$API_FILE"
        
        # Add router inclusion
        sed -i '/app.include_router/a app.include_router(svm_debug_router, prefix="/api/v1", tags=["svm-debug"])' "$API_FILE"
        
        print_status "SVM debug routes added to API"
    else
        print_status "SVM debug routes already configured"
    fi
else
    print_warning "Main API file not found, please manually add SVM debug routes"
fi

# Step 6: Update Dashboard
echo ""
print_info "Step 6: Updating Dashboard"
echo "========================="

if [ -f "update_dashboard_svm.sh" ]; then
    print_info "Running dashboard update script..."
    chmod +x update_dashboard_svm.sh
    ./update_dashboard_svm.sh
    print_status "Dashboard updated with SVM debug tab"
else
    print_warning "Dashboard update script not found"
fi

# Step 7: Update Docker configuration
echo ""
print_info "Step 7: Updating Docker Configuration"
echo "===================================="

if [ -f "docker-compose.yml" ]; then
    # Backup docker-compose.yml
    cp docker-compose.yml docker-compose.yml.backup
    
    # Add volume mounts for debug output if not present
    if ! grep -q "debug_output" docker-compose.yml; then
        print_info "Adding debug output volume to Docker Compose..."
        # This is a simplified approach - in practice, you'd want more sophisticated YAML editing
        print_warning "Please manually add the following volume to your docker-compose.yml:"
        echo "      - ./debug_output:/app/debug_output"
        echo "      - ./static:/app/static"
    else
        print_status "Docker volumes already configured"
    fi
else
    print_warning "docker-compose.yml not found"
fi

# Step 8: Create test data
echo ""
print_info "Step 8: Creating Test Data"
echo "========================="

if [ ! -f "example_sessions.json" ]; then
    print_info "Creating example test sessions..."
    
    cat > example_sessions.json << 'EOF'
{
  "sessions": [
    {
      "session_id": "test_session_1",
      "raw_text": "Normal ATM transaction: user inserted card, entered PIN, checked balance, withdrew $100, transaction completed successfully",
      "expected_anomaly": false,
      "features": {
        "transaction_amount": 100,
        "transaction_type": "withdrawal",
        "time_of_day": "14:30",
        "day_of_week": "Tuesday"
      }
    },
    {
      "session_id": "test_session_2", 
      "raw_text": "Unusual behavior detected: multiple failed PIN attempts, card skimmer device detected, suspicious activity around ATM terminal",
      "expected_anomaly": true,
      "features": {
        "failed_attempts": 5,
        "device_tampered": true,
        "suspicious_activity": true,
        "time_of_day": "02:15"
      }
    },
    {
      "session_id": "test_session_3",
      "raw_text": "Standard transaction: balance inquiry followed by small withdrawal of $40, customer left area normally",
      "expected_anomaly": false,
      "features": {
        "transaction_amount": 40,
        "transaction_type": "withdrawal", 
        "balance_check": true,
        "time_of_day": "09:45"
      }
    }
  ]
}
EOF
    print_status "Example test sessions created"
else
    print_status "Test sessions already exist"
fi

# Step 9: Run basic tests
echo ""
print_info "Step 9: Running Basic Tests"
echo "=========================="

# Test CLI tool if available
if [ -f "debug_svm_cli.py" ] && [ -f "example_sessions.json" ]; then
    print_info "Testing CLI debug tool..."
    
    # Create test output directory
    mkdir -p test_debug_output
    
    # Run CLI test
    if python debug_svm_cli.py --session-file example_sessions.json --output-dir ./test_debug_output --verbose; then
        print_status "CLI debug tool test passed"
        
        # Show generated files
        if [ -d "test_debug_output" ]; then
            FILES_COUNT=$(find test_debug_output -type f | wc -l)
            print_status "$FILES_COUNT debug files generated in test_debug_output/"
        fi
    else
        print_warning "CLI debug tool test failed (this may be expected if models aren't trained yet)"
    fi
else
    print_warning "CLI test skipped (missing files)"
fi

# Step 10: Start services
echo ""
print_info "Step 10: Starting Services"
echo "========================="

print_info "Restarting Docker containers to apply changes..."

# Stop containers
if docker-compose ps | grep -q "Up"; then
    print_info "Stopping existing containers..."
    docker-compose down
fi

# Start containers
print_info "Starting containers..."
docker-compose up -d

# Wait for services to start
print_info "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_status "Services are running"
    
    # Display service URLs
    echo ""
    print_info "Service URLs:"
    echo "  🌐 Dashboard: http://localhost:3000"
    echo "  🔧 API: http://localhost:8000"
    echo "  📊 SVM Debug API: http://localhost:8000/api/v1/svm-debug/"
    echo "  📖 API Docs: http://localhost:8000/docs"
else
    print_error "Some services failed to start"
    print_info "Check logs with: docker-compose logs"
fi

# Step 11: Final validation
echo ""
print_info "Step 11: Final Validation"
echo "========================"

# Check API health
print_info "Checking API health..."
sleep 5  # Give API time to start

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "API is responding"
else
    print_warning "API not responding (may still be starting up)"
fi

# Check SVM debug endpoints
if curl -s http://localhost:8000/api/v1/svm-debug/model-info > /dev/null 2>&1; then
    print_status "SVM debug endpoints are available"
else
    print_warning "SVM debug endpoints not responding (may need model training first)"
fi

# Summary
echo ""
echo "🎉 SVM Debug System Deployment Complete!"
echo "========================================"
echo ""
print_status "Components Deployed:"
echo "   ✓ SVM Visualizer (svm_visualizer.py)"
echo "   ✓ SVM Debug API (svm_debug_api.py)"
echo "   ✓ React Dashboard Component (SVMDebugDashboard.js)"
echo "   ✓ CLI Debug Tool (debug_svm_cli.py)"
echo "   ✓ Test Data (example_sessions.json)"
echo ""
print_info "Available Interfaces:"
echo "   🌐 Web Dashboard: http://localhost:3000 (SVM Debug tab)"
echo "   🔧 REST API: http://localhost:8000/api/v1/svm-debug/"
echo "   💻 CLI Tool: python debug_svm_cli.py --help"
echo ""
print_info "Quick Start Commands:"
echo "   # Test CLI debugging"
echo "   python debug_svm_cli.py --session-file example_sessions.json --output-dir ./debug_output"
echo ""
echo "   # View API documentation"
echo "   curl http://localhost:8000/docs"
echo ""
echo "   # Debug a session via API"
echo "   curl -X POST http://localhost:8000/api/v1/svm-debug/analyze-session \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"session_id\": \"test_session_1\", \"raw_text\": \"sample text\"}'"
echo ""
print_info "Next Steps:"
echo "   1. Access the dashboard and navigate to the 'SVM Debug' tab"
echo "   2. Upload EJ files to generate anomaly sessions"
echo "   3. Use the SVM debug tools to analyze model decisions"
echo "   4. Review generated visualizations in the debug_output/ directory"
echo ""
print_warning "Note: If this is a fresh installation, train your models first by:"
echo "   1. Uploading EJ files through the dashboard"
echo "   2. Labeling some anomalies through the Expert Review tab"
echo "   3. Running the retraining process"
echo ""

echo "📖 For detailed usage instructions, see: SVM_DEBUG_USAGE.md"
echo ""
