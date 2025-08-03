#!/bin/bash
# deploy_deeplog_to_digitalocean.sh
# Script to deploy BERT-Enhanced DeepLog Dashboard to DigitalOcean

set -e

echo "🚀 BERT-Enhanced DeepLog DigitalOcean Deployment"
echo "=================================================="

# Configuration
DIGITALOCEAN_IP="64.227.16.180"
PROJECT_DIR="/root/abm-anomaly-ml-first"
REPO_URL="https://github.com/YanBelle/Capstone-Project.git"

echo "📋 Deployment Configuration:"
echo "   Server IP: $DIGITALOCEAN_IP"
echo "   Project Directory: $PROJECT_DIR"
echo "   Repository: $REPO_URL"

echo -e "\n📝 Manual Steps Required on DigitalOcean Server:"
echo "=================================================="

echo "1. SSH to your DigitalOcean server:"
echo "   ssh root@$DIGITALOCEAN_IP"

echo -e "\n2. Install Docker and Docker Compose (if not already installed):"
echo "   apt update"
echo "   apt install -y docker.io docker-compose-plugin"
echo "   systemctl start docker"
echo "   systemctl enable docker"

echo -e "\n3. Clone/Update the project:"
echo "   cd /root"
echo "   if [ -d '$PROJECT_DIR' ]; then"
echo "       cd $PROJECT_DIR"
echo "       git pull origin main"
echo "   else"
echo "       git clone $REPO_URL"
echo "       cd Capstone-Project/abm-anomaly-ml-first"
echo "   fi"

echo -e "\n4. Configure environment for production:"
echo "   cp .env.production .env"
echo "   # Edit .env if needed to ensure REACT_APP_API_URL=http://64.227.16.180"

echo -e "\n5. Deploy the services:"
echo "   docker-compose down  # Stop any existing services"
echo "   docker-compose build  # Build the images"
echo "   docker-compose up -d  # Start services in background"

echo -e "\n6. Verify deployment:"
echo "   docker-compose ps    # Check service status"
echo "   docker-compose logs api | tail -20  # Check API logs"

echo -e "\n7. Test the endpoints:"
echo "   curl http://localhost/health"
echo "   curl http://localhost/api/v1/bert-deeplog/model-info"

echo -e "\n🔧 Troubleshooting Commands:"
echo "=================================================="
echo "# Check service logs:"
echo "docker-compose logs api"
echo "docker-compose logs dashboard"
echo "docker-compose logs nginx"

echo -e "\n# Restart specific service:"
echo "docker-compose restart api"
echo "docker-compose restart dashboard"

echo -e "\n# Full restart:"
echo "docker-compose down && docker-compose up -d"

echo -e "\n📍 Expected Access URLs after deployment:"
echo "=================================================="
echo "🎯 Main Dashboard: http://$DIGITALOCEAN_IP/"
echo "🧠 DeepLog Dashboard: http://$DIGITALOCEAN_IP/dashboard/deeplog"
echo "🔧 API Health: http://$DIGITALOCEAN_IP/health"
echo "📊 Model Info: http://$DIGITALOCEAN_IP/api/v1/bert-deeplog/model-info"
echo "📈 Grafana: http://$DIGITALOCEAN_IP:3001"
echo "📊 Prometheus: http://$DIGITALOCEAN_IP:9090"

echo -e "\n⚠️  Important Notes:"
echo "=================================================="
echo "• Make sure DigitalOcean firewall allows ports 80, 443, 3001, 9090"
echo "• The .env.production file should have REACT_APP_API_URL=http://64.227.16.180"
echo "• Services might take 2-3 minutes to fully start up"
echo "• Check 'docker-compose logs' if services fail to start"

echo -e "\n🔍 Quick Deployment Check Script:"
echo "=================================================="

cat << 'EOF' > check_deployment.sh
#!/bin/bash
echo "🔍 Checking DigitalOcean Deployment Status..."

# Test basic connectivity
echo "Testing basic connectivity..."
if curl -s --connect-timeout 5 http://localhost/health > /dev/null; then
    echo "✅ Server is responding"
else
    echo "❌ Server not responding on port 80"
fi

# Check Docker services
echo -e "\nChecking Docker services..."
docker-compose ps

# Test API endpoints
echo -e "\nTesting API endpoints..."
curl -s http://localhost/api/v1/bert-deeplog/model-info | head -100

echo -e "\n🎯 If everything is working, access via:"
echo "   http://64.227.16.180/dashboard/deeplog"
EOF

chmod +x check_deployment.sh

echo -e "\nCreated 'check_deployment.sh' - copy this to your DigitalOcean server and run it after deployment"

echo -e "\n🚀 Ready to deploy! Follow the manual steps above on your DigitalOcean server."
