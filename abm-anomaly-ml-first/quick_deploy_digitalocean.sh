#!/bin/bash
# One-liner deployment for DigitalOcean
# Copy and paste this entire command block into your DigitalOcean server terminal

# Quick deployment script for BERT-Enhanced DeepLog on DigitalOcean
echo "🚀 Starting BERT-Enhanced DeepLog Deployment..."

# Navigate to project directory
cd /root

# Clone or update repository
if [ -d "Capstone-Project" ]; then
    echo "📥 Updating existing repository..."
    cd Capstone-Project
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone https://github.com/YanBelle/Capstone-Project.git
    cd Capstone-Project
fi

# Navigate to project
cd abm-anomaly-ml-first

# Use production environment
echo "⚙️  Configuring production environment..."
cp .env.production .env

# Stop any existing services
echo "⏹️  Stopping existing services..."
docker-compose down

# Build and start services
echo "🔨 Building services..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check status
echo "📊 Service Status:"
docker-compose ps

# Test endpoints
echo "🧪 Testing endpoints..."
echo "Health check:"
curl -s http://localhost/health || echo "Health check failed"

echo -e "\nModel info:"
curl -s http://localhost/api/v1/bert-deeplog/model-info | head -100 || echo "Model info failed"

echo -e "\n✅ Deployment complete!"
echo "🎯 Access your BERT-Enhanced DeepLog Dashboard at:"
echo "   http://64.227.16.180/dashboard/deeplog"
