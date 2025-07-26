#!/bin/bash

# Deployment script for flexible local/remote setup
# Usage: ./deploy.sh [local|production] [action]
# Actions: up, down, build, logs

set -e

# Default values
ENVIRONMENT=${1:-local}
ACTION=${2:-up}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
if [[ "$ENVIRONMENT" != "local" && "$ENVIRONMENT" != "production" ]]; then
    print_error "Invalid environment. Use 'local' or 'production'"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Set environment file
ENV_FILE=".env.${ENVIRONMENT}"

if [[ ! -f "$ENV_FILE" ]]; then
    print_error "Environment file $ENV_FILE not found!"
    exit 1
fi

print_status "Using environment: $ENVIRONMENT"
print_status "Environment file: $ENV_FILE"

# Execute action
case $ACTION in
    "up")
        print_status "Starting services in $ENVIRONMENT mode..."
        docker compose --env-file "$ENV_FILE" up -d
        
        print_status "Waiting for services to be ready..."
        sleep 10
        
        # Check if services are running
        if docker compose --env-file "$ENV_FILE" ps | grep -q "Up"; then
            print_status "Services started successfully!"
            
            if [[ "$ENVIRONMENT" == "local" ]]; then
                print_status "Local deployment ready:"
                print_status "  Dashboard: http://localhost"
                print_status "  API: http://localhost/api"
                print_status "  Grafana: http://localhost:3001"
                print_status "  Prometheus: http://localhost:9090"
            else
                print_status "Production deployment ready on configured domain"
            fi
        else
            print_error "Some services failed to start. Check logs with: ./deploy.sh $ENVIRONMENT logs"
        fi
        ;;
        
    "down")
        print_status "Stopping services..."
        docker compose --env-file "$ENV_FILE" down
        print_status "Services stopped."
        ;;
        
    "build")
        print_status "Building services for $ENVIRONMENT..."
        docker compose --env-file "$ENV_FILE" build
        print_status "Build completed."
        ;;
        
    "logs")
        print_status "Showing logs for $ENVIRONMENT environment..."
        docker compose --env-file "$ENV_FILE" logs -f
        ;;
        
    "restart")
        print_status "Restarting services..."
        docker compose --env-file "$ENV_FILE" down
        docker compose --env-file "$ENV_FILE" up -d
        print_status "Services restarted."
        ;;
        
    "status")
        print_status "Service status for $ENVIRONMENT environment:"
        docker compose --env-file "$ENV_FILE" ps
        ;;
        
    *)
        print_error "Invalid action. Use: up, down, build, logs, restart, or status"
        exit 1
        ;;
esac
