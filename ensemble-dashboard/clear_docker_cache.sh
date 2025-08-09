#!/bin/bash

# Docker Cache Clearing Script for Ensemble Dashboard
# Note: This script assumes Docker Desktop is running

echo "🧹 Clearing Docker cache for Ensemble Dashboard..."

# Navigate to the ensemble dashboard directory
cd "$(dirname "$0")"

echo "📍 Current directory: $(pwd)"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not available or not in PATH"
    echo "Please ensure Docker Desktop is installed and running"
    exit 1
fi

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not available"
    echo "Please ensure Docker Desktop is installed and running"
    exit 1
fi

echo "✅ Using Docker Compose command: $COMPOSE_CMD"

# Step 1: Stop all running containers
echo "⏹️  Stopping all containers..."
$COMPOSE_CMD down

# Step 2: Remove containers (including stopped ones)
echo "🗑️  Removing containers..."
$COMPOSE_CMD rm -f

# Step 3: Remove images (force rebuild)
echo "🏗️  Removing built images to force rebuild..."
docker rmi ensemble-dashboard-backend:latest 2>/dev/null || echo "Backend image not found"
docker rmi ensemble-dashboard-frontend:latest 2>/dev/null || echo "Frontend image not found"
docker rmi ensemble-backend:latest 2>/dev/null || echo "ensemble-backend image not found"
docker rmi ensemble-frontend:latest 2>/dev/null || echo "ensemble-frontend image not found"

# Step 4: Remove all images with the project name pattern
echo "🧽 Removing all ensemble dashboard related images..."
docker images | grep -E "(ensemble|dashboard)" | awk '{print $3}' | xargs -r docker rmi -f

# Step 5: Prune Docker system (remove unused containers, networks, images)
echo "🗂️  Pruning Docker system..."
docker system prune -f

# Step 6: Remove build cache
echo "💾 Removing build cache..."
docker builder prune -f

# Step 7: Remove volumes (optional - uncomment if you want to clear data)
# echo "📦 Removing volumes..."
# docker volume prune -f

# Step 8: Clean up dangling images
echo "🧼 Removing dangling images..."
docker image prune -f

# Step 9: Rebuild and start with no cache
echo "🔨 Rebuilding and starting containers with fresh build..."
$COMPOSE_CMD build --no-cache

echo "🚀 Starting containers..."
$COMPOSE_CMD up -d

# Step 10: Show status
echo "📊 Container status:"
$COMPOSE_CMD ps

# Step 11: Show logs for verification
echo "📋 Showing initial logs..."
echo "Backend logs:"
$COMPOSE_CMD logs backend | tail -10
echo ""
echo "Frontend logs:"
$COMPOSE_CMD logs frontend | tail -10

echo ""
echo "✅ Docker cache cleared and containers rebuilt!"
echo "🌐 Frontend should be available at: http://localhost:3000"
echo "🔗 Backend should be available at: http://localhost:8001"
echo ""
echo "To follow logs in real-time, run:"
echo "  $COMPOSE_CMD logs -f"
