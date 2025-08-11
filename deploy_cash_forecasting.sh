#!/bin/bash

# Cash Forecasting Production Deployment Script
# ============================================

set -e

echo "🚀 Starting Cash Forecasting Production Deployment..."

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed ✓"
    
    # Check Docker Compose
    if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Desktop or Docker Compose plugin."
        exit 1
    fi
    print_status "Docker Compose is available ✓"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    print_status "Docker daemon is running ✓"
}

# Create required directories
create_directories() {
    print_header "Creating Required Directories"
    
    mkdir -p logs
    mkdir -p data/models
    mkdir -p data/postgres
    mkdir -p data/redis
    mkdir -p integration
    
    print_status "Created directories ✓"
}

# Set up environment
setup_environment() {
    print_header "Setting Up Environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/abm_database
POSTGRES_DB=abm_database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Cash Forecasting Configuration
FLASK_ENV=production
MODEL_RETRAIN_HOURS=24
DASHBOARD_REFRESH_MINUTES=15

# Ports
CASH_FORECASTING_PORT=5001
NGINX_PORT=80
POSTGRES_PORT=5432
REDIS_PORT=6379
EOF
        print_status "Created .env file ✓"
    else
        print_status "Using existing .env file ✓"
    fi
}

# Generate integration files
generate_integration() {
    print_header "Generating Integration Files"
    
    if [ -f "create_dashboard_integration.py" ]; then
        python create_dashboard_integration.py
        print_status "Generated integration files ✓"
    else
        print_warning "Integration generator not found, skipping..."
    fi
}

# Build Docker images
build_images() {
    print_header "Building Docker Images"
    
    print_status "Building cash forecasting image..."
    docker compose build cash-forecasting
    
    print_status "Docker images built successfully ✓"
}

# Start services
start_services() {
    print_header "Starting Services"
    
    print_status "Starting PostgreSQL and Redis..."
    docker compose up -d postgres redis
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
    
    print_status "Starting cash forecasting service..."
    docker compose up -d cash-forecasting
    
    print_status "Starting nginx reverse proxy..."
    docker compose up -d nginx
    
    print_status "All services started ✓"
}

# Check service health
check_health() {
    print_header "Checking Service Health"
    
    # Wait for services to be fully ready
    sleep 15
    
    # Check cash forecasting service
    if curl -f -s http://localhost:5001/health > /dev/null; then
        print_status "Cash forecasting service is healthy ✓"
    else
        print_warning "Cash forecasting service health check failed"
    fi
    
    # Check main dashboard integration
    if curl -f -s http://localhost/cash-forecasting/ > /dev/null; then
        print_status "Dashboard integration is working ✓"
    else
        print_warning "Dashboard integration may not be working properly"
    fi
    
    # Check API endpoints
    if curl -f -s http://localhost/api/cash-forecasting/terminal-status > /dev/null; then
        print_status "API endpoints are accessible ✓"
    else
        print_warning "API endpoints may not be accessible"
    fi
}

# Show service status
show_status() {
    print_header "Service Status"
    
    echo "Docker containers:"
    docker compose ps
    
    echo ""
    echo "Service URLs:"
    echo "📊 Main Dashboard: http://localhost/"
    echo "💰 Cash Forecasting: http://localhost/cash-forecasting/"
    echo "🔌 Direct API: http://localhost:5001/"
    echo "📡 API Endpoints: http://localhost/api/cash-forecasting/"
    echo ""
    
    echo "Available API endpoints:"
    echo "• GET /api/cash-forecasting/terminal-status"
    echo "• GET /api/cash-forecasting/alerts"
    echo "• GET /api/cash-forecasting/predictions"
    echo "• GET /api/cash-forecasting/performance"
    echo "• POST /api/cash-forecasting/retrain"
    echo ""
    
    print_status "Deployment completed successfully! 🎉"
}

# Show logs function
show_logs() {
    print_header "Service Logs"
    echo "Use these commands to view logs:"
    echo "• All services: docker compose logs -f"
    echo "• Cash forecasting only: docker compose logs -f cash-forecasting"
    echo "• Database: docker compose logs -f postgres"
    echo "• Nginx: docker compose logs -f nginx"
}

# Stop services function
stop_services() {
    print_header "Stopping Services"
    docker compose down
    print_status "All services stopped ✓"
}

# Restart services function
restart_services() {
    print_header "Restarting Services"
    docker compose restart
    print_status "All services restarted ✓"
}

# Update models function
update_models() {
    print_header "Updating ML Models"
    
    # Trigger model retraining
    curl -X POST http://localhost/api/cash-forecasting/retrain
    print_status "Model retraining triggered ✓"
}

# Main deployment function
deploy() {
    check_prerequisites
    create_directories
    setup_environment
    generate_integration
    build_images
    start_services
    check_health
    show_status
    show_logs
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "health")
        check_health
        ;;
    "update")
        update_models
        ;;
    "help"|"--help"|"-h")
        echo "Cash Forecasting Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy     Full deployment (default)"
        echo "  start      Start all services"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  status     Show service status"
        echo "  logs       Show log commands"
        echo "  health     Check service health"
        echo "  update     Update ML models"
        echo "  help       Show this help"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac
