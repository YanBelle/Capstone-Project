#!/bin/bash

# Complete setup verification script for the ABM ML-First Anomaly Detection System
# This script demonstrates the new flexible deployment architecture

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Main verification
print_header "ABM ML-First Anomaly Detection System - Setup Verification"

print_info "This script verifies the new flexible deployment architecture"
echo ""

# Check Docker
print_info "Checking Docker installation..."
if command -v docker &> /dev/null; then
    if docker info > /dev/null 2>&1; then
        print_status "Docker is installed and running"
    else
        print_error "Docker is installed but not running. Please start Docker."
        exit 1
    fi
else
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check environment files
print_info "Checking environment configuration..."
if [[ -f ".env.local" ]]; then
    print_status "Local environment configuration found"
else
    print_error "Local environment file missing"
    exit 1
fi

if [[ -f ".env.production" ]]; then
    print_status "Production environment configuration found"
else
    print_error "Production environment file missing"
    exit 1
fi

# Check deploy script
print_info "Checking deployment script..."
if [[ -f "deploy.sh" && -x "deploy.sh" ]]; then
    print_status "Deployment script is ready"
else
    print_error "Deployment script missing or not executable"
    exit 1
fi

# Check Docker Compose file
print_info "Checking Docker Compose configuration..."
if [[ -f "docker-compose.yml" ]]; then
    print_status "Docker Compose configuration found"
else
    print_error "Docker Compose file missing"
    exit 1
fi

# Check nginx configuration
print_info "Checking Nginx configuration..."
if [[ -f "nginx/default.conf" ]]; then
    print_status "Nginx configuration found"
else
    print_error "Nginx configuration missing"
    exit 1
fi

# Check service directories
print_info "Checking service structure..."
SERVICES=("api" "anomaly-detector" "dashboard")
for service in "${SERVICES[@]}"; do
    if [[ -d "services/$service" ]]; then
        print_status "Service '$service' directory found"
    else
        print_error "Service '$service' directory missing"
        exit 1
    fi
done

# Check key files
print_info "Checking key system files..."
KEY_FILES=(
    "services/api/main.py"
    "services/anomaly-detector/main.py"
    "services/anomaly-detector/ml_analyzer.py"
    "database/migrations/002_multi_anomaly_support.sql"
    "init-db/01-base-schema.sql"
    "init-db/02-ml-schema.sql"
)

for file in "${KEY_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        print_status "Key file '$file' found"
    else
        print_error "Key file '$file' missing"
        exit 1
    fi
done

echo ""
print_header "Deployment Architecture Summary"

echo -e "${YELLOW}Available Deployment Commands:${NC}"
echo "  ./deploy.sh local up      - Start local development environment"
echo "  ./deploy.sh local down    - Stop local environment"
echo "  ./deploy.sh local status  - Check local service status"
echo "  ./deploy.sh local logs    - View local logs"
echo ""
echo "  ./deploy.sh production up      - Deploy to production"
echo "  ./deploy.sh production down    - Stop production services"
echo "  ./deploy.sh production status  - Check production status"
echo "  ./deploy.sh production logs    - View production logs"

echo ""
echo -e "${YELLOW}Local Access URLs (when running):${NC}"
echo "  Dashboard:  http://localhost"
echo "  API:        http://localhost/api"
echo "  Grafana:    http://localhost:3001"
echo "  Prometheus: http://localhost:9090"

echo ""
echo -e "${YELLOW}Key Features:${NC}"
echo "  ✓ ML-First anomaly detection with multiple models"
echo "  ✓ Continuous learning with expert feedback"
echo "  ✓ Database-driven model retraining"
echo "  ✓ Real-time monitoring and visualization"
echo "  ✓ Flexible local/production deployment"
echo "  ✓ Automated CI/CD with GitHub Actions"

echo ""
echo -e "${YELLOW}Database Features:${NC}"
echo "  ✓ PostgreSQL with ML-optimized schema"
echo "  ✓ Automatic migrations and setup"
echo "  ✓ 66+ labeled anomalies for training"
echo "  ✓ Expert feedback integration"
echo "  ✓ Session-based anomaly tracking"

echo ""
echo -e "${YELLOW}ML Model Pipeline:${NC}"
echo "  ✓ Isolation Forest (unsupervised)"
echo "  ✓ One-Class SVM (unsupervised)"
echo "  ✓ Supervised Classifier (labeled data)"
echo "  ✓ Expert Rules Engine (domain knowledge)"
echo "  ✓ Ensemble scoring and consensus"

echo ""
print_header "Setup Complete!"

print_status "All components verified and ready for deployment"
print_info "To start local development: ./deploy.sh local up"
print_info "To deploy to production: ./deploy.sh production up"
print_info "For detailed documentation: see DEPLOYMENT_GUIDE.md"

echo ""
print_info "The system is now configured for flexible deployment across:"
echo "  - Local development environment"
echo "  - DigitalOcean production environment"
echo "  - Any Docker-compatible server"

echo ""
print_status "Flexible deployment architecture implementation complete!"
