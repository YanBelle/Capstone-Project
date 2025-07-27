#!/bin/bash

# DigitalOcean Development Environment Setup Script
# Run this on your DigitalOcean server to set up VS Code Remote development

set -e

echo "🚀 Setting up DigitalOcean Development Environment"
echo "================================================="

# Server details
SERVER_IP="64.227.16.180"
DEV_USER="yc"
GITHUB_REPO="https://github.com/YanBelle/Capstone-Project.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running as correct user
if [ "$USER" != "root" ]; then
    print_error "Please run this script as root user"
    exit 1
fi

print_info "Setting up development environment for user: $DEV_USER"

# Step 1: Create development user and directories
print_info "Step 1: Setting up user and directories"

# Ensure user exists and has proper setup
if id "$DEV_USER" &>/dev/null; then
    print_status "User $DEV_USER already exists"
else
    useradd -m -s /bin/bash $DEV_USER
    echo "$DEV_USER:yc" | chpasswd
    usermod -aG sudo $DEV_USER
    print_status "Created user $DEV_USER"
fi

# Create development directory
sudo -u $DEV_USER mkdir -p /home/$DEV_USER/development
sudo -u $DEV_USER mkdir -p /home/$DEV_USER/.ssh
chown -R $DEV_USER:$DEV_USER /home/$DEV_USER/

print_status "Development directories created"

# Step 2: Install required packages
print_info "Step 2: Installing required packages"

apt-get update
apt-get install -y git curl wget nano vim docker.io docker-compose python3 python3-pip nodejs npm

# Ensure Docker is running
systemctl start docker
systemctl enable docker

# Add user to docker group
usermod -aG docker $DEV_USER

print_status "Required packages installed"

# Step 3: Clone repository for development
print_info "Step 3: Setting up development repository"

cd /home/$DEV_USER/development

# Clone repository as development user
if [ -d "Capstone-Project" ]; then
    print_warning "Repository already exists, pulling latest changes"
    sudo -u $DEV_USER git -C Capstone-Project pull
else
    sudo -u $DEV_USER git clone $GITHUB_REPO
    print_status "Repository cloned"
fi

cd Capstone-Project/abm-anomaly-ml-first

# Step 4: Create development Docker Compose file
print_info "Step 4: Creating development Docker configuration"

# Copy main docker-compose.yml to development version
sudo -u $DEV_USER cp docker-compose.yml docker-compose.dev.yml

# Update ports in development compose file
cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: postgres_dev
    environment:
      POSTGRES_DB: abmdb_dev
      POSTGRES_USER: abmuser
      POSTGRES_PASSWORD: abmpass123
      POSTGRES_HOST_AUTH_METHOD: trust
    ports:
      - "5433:5432"
    volumes:
      - postgres_data_dev:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d
    networks:
      - abm_network_dev

  redis:
    image: redis:7-alpine
    container_name: redis_dev
    ports:
      - "6380:6379"
    volumes:
      - redis_data_dev:/data
    networks:
      - abm_network_dev

  api:
    build:
      context: ./services/api
      dockerfile: Dockerfile
    container_name: api_dev
    ports:
      - "8001:8000"
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=abmdb_dev
      - POSTGRES_USER=abmuser
      - POSTGRES_PASSWORD=abmpass123
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://abmuser:abmpass123@postgres:5432/abmdb_dev
    volumes:
      - ./services/api:/app
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    networks:
      - abm_network_dev
    restart: unless-stopped

  dashboard:
    build:
      context: ./services/dashboard
      dockerfile: Dockerfile
    container_name: dashboard_dev
    ports:
      - "3001:3000"
    environment:
      - REACT_APP_API_URL=http://64.227.16.180:8001
    volumes:
      - ./services/dashboard/src:/app/src
      - ./services/dashboard/public:/app/public
    depends_on:
      - api
    networks:
      - abm_network_dev
    restart: unless-stopped

  anomaly-detector:
    build:
      context: ./services/anomaly-detector
      dockerfile: Dockerfile
    container_name: anomaly_detector_dev
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=abmdb_dev
      - POSTGRES_USER=abmuser
      - POSTGRES_PASSWORD=abmpass123
      - REDIS_HOST=redis
    volumes:
      - ./services/anomaly-detector:/app
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    networks:
      - abm_network_dev
    restart: unless-stopped

volumes:
  postgres_data_dev:
  redis_data_dev:

networks:
  abm_network_dev:
    driver: bridge
EOF

chown $DEV_USER:$DEV_USER docker-compose.dev.yml
print_status "Development Docker configuration created"

# Step 5: Create environment file for development
print_info "Step 5: Creating development environment file"

cat > .env.dev << 'EOF'
# Development Environment Variables
POSTGRES_HOST=postgres
POSTGRES_DB=abmdb_dev
POSTGRES_USER=abmuser
POSTGRES_PASSWORD=abmpass123
REDIS_HOST=redis
REDIS_PASSWORD=
DATABASE_URL=postgresql://abmuser:abmpass123@postgres:5432/abmdb_dev

# API Configuration
API_PORT=8001
REACT_APP_API_URL=http://64.227.16.180:8001

# ML Configuration
MODEL_PATH=/app/models
BERT_MODEL=bert-base-uncased

# Development flags
NODE_ENV=development
DEBUG=true
EOF

chown $DEV_USER:$DEV_USER .env.dev
print_status "Development environment file created"

# Step 6: Set up Git configuration
print_info "Step 6: Setting up Git configuration"

# Configure git for development user
sudo -u $DEV_USER git config --global user.name "Development User"
sudo -u $DEV_USER git config --global user.email "yc@dev.local"
sudo -u $DEV_USER git config --global init.defaultBranch main

print_status "Git configuration completed"

# Step 7: Install Python dependencies
print_info "Step 7: Installing Python dependencies"

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

if [ -f "svm_requirements.txt" ]; then
    pip3 install -r svm_requirements.txt
fi

print_status "Python dependencies installed"

# Step 8: Create development helper scripts
print_info "Step 8: Creating development helper scripts"

# Create start development script
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Development Environment"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml up -d
echo "✅ Development environment started"
echo "📊 Dashboard: http://64.227.16.180:3001"
echo "🔧 API: http://64.227.16.180:8001"
echo "📖 API Docs: http://64.227.16.180:8001/docs"
EOF

# Create stop development script
cat > stop_dev.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping Development Environment"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml down
echo "✅ Development environment stopped"
EOF

# Create status script
cat > status_dev.sh << 'EOF'
#!/bin/bash
echo "📊 Development Environment Status"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml ps
echo ""
echo "🌐 Access URLs:"
echo "   Dashboard: http://64.227.16.180:3001"
echo "   API: http://64.227.16.180:8001"
echo "   API Docs: http://64.227.16.180:8001/docs"
EOF

# Create git helper script
cat > git_push.sh << 'EOF'
#!/bin/bash
echo "📤 Git Push Helper"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first

echo "Current status:"
git status

echo ""
read -p "Enter commit message: " commit_msg

if [ -n "$commit_msg" ]; then
    git add .
    git commit -m "$commit_msg"
    git push origin main
    echo "✅ Changes pushed to GitHub"
    echo "🚀 GitHub Actions will deploy to production"
else
    echo "❌ No commit message provided"
fi
EOF

# Make scripts executable
chmod +x start_dev.sh stop_dev.sh status_dev.sh git_push.sh
chown $DEV_USER:$DEV_USER start_dev.sh stop_dev.sh status_dev.sh git_push.sh

print_status "Development helper scripts created"

# Step 9: Set up SSH for VS Code Remote
print_info "Step 9: Setting up SSH for VS Code Remote"

# Enable password authentication for initial connection
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Restart SSH service
systemctl restart ssh

print_status "SSH configured for VS Code Remote"

# Step 10: Final setup
print_info "Step 10: Final setup and permissions"

# Ensure all files are owned by development user
chown -R $DEV_USER:$DEV_USER /home/$DEV_USER/

# Create .bashrc additions for development user
cat >> /home/$DEV_USER/.bashrc << 'EOF'

# Development environment shortcuts
alias dev-start='cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first && ./start_dev.sh'
alias dev-stop='cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first && ./stop_dev.sh'
alias dev-status='cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first && ./status_dev.sh'
alias dev-push='cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first && ./git_push.sh'
alias dev-cd='cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first'

echo "🔧 Development shortcuts loaded:"
echo "   dev-start  - Start development environment"
echo "   dev-stop   - Stop development environment"
echo "   dev-status - Check status"
echo "   dev-push   - Git push changes"
echo "   dev-cd     - Go to project directory"
EOF

print_status "Development shortcuts configured"

# Summary
echo ""
print_info "🎉 Development Environment Setup Complete!"
echo ""
print_status "Development Environment Details:"
echo "   📍 Location: /home/yc/development/Capstone-Project/abm-anomaly-ml-first"
echo "   🌐 Dashboard: http://64.227.16.180:3001"
echo "   🔧 API: http://64.227.16.180:8001"
echo "   📖 API Docs: http://64.227.16.180:8001/docs"
echo ""
print_status "VS Code Remote SSH Connection:"
echo "   📡 Host: 64.227.16.180"
echo "   👤 User: yc"
echo "   🔑 Password: yc"
echo "   📂 Folder: /home/yc/development/Capstone-Project/abm-anomaly-ml-first"
echo ""
print_status "Quick Commands (after SSH):"
echo "   dev-start  - Start development containers"
echo "   dev-stop   - Stop development containers"
echo "   dev-status - Check container status"
echo "   dev-push   - Push changes to GitHub"
echo ""
print_info "Next Steps:"
echo "   1. Connect VS Code Remote SSH to yc@64.227.16.180"
echo "   2. Open folder: /home/yc/development/Capstone-Project/abm-anomaly-ml-first"
echo "   3. Run: dev-start (to start development environment)"
echo "   4. Edit code directly in VS Code"
echo "   5. Test at: http://64.227.16.180:3001"
echo "   6. When ready: dev-push (pushes to GitHub → triggers production deploy)"
echo ""

print_status "Setup completed successfully! 🎊"
