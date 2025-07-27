# VS Code Remote Development Setup for DigitalOcean
# Complete setup for development environment with git workflow

## 🎯 **Target Architecture:**
```
Local VS Code ← SSH → DigitalOcean Dev Environment → Git Push → GitHub → Production Deploy
    (Windows)              (64.227.16.180)                           (Same server, different containers)
```

## 📋 **Server Setup Plan:**

### 1. **Development Environment** (Port 3001, 8001)
- Full Docker setup for development
- Git repository clone
- Direct code editing via VS Code Remote SSH
- Separate from production

### 2. **Production Environment** (Port 3000, 8000)
- Deployed via GitHub Actions
- No direct editing
- Stable production deployment

---

## 🔧 **Step-by-Step Setup:**

### **Step 1: Connect to DigitalOcean Server**
```bash
# From your Windows machine
ssh yc@64.227.16.180

# Switch to root if needed
sudo su -
```

### **Step 2: Create Development Environment**
```bash
# Create separate development directory
mkdir -p /home/yc/development
cd /home/yc/development

# Clone your repository for development
git clone https://github.com/YanBelle/Capstone-Project.git
cd Capstone-Project/abm-anomaly-ml-first

# Create development docker-compose file
cp docker-compose.yml docker-compose.dev.yml
```

### **Step 3: Configure Development Ports**
Edit `docker-compose.dev.yml` to use different ports:
- Dashboard: 3001 (instead of 3000)
- API: 8001 (instead of 8000)
- Database: 5433 (instead of 5432)
- Redis: 6380 (instead of 6379)

### **Step 4: Set Up Git Configuration**
```bash
# Configure git user
git config --global user.name "Your Name"
git config --global user.email "your.email@domain.com"

# Set up SSH key for GitHub (if not exists)
ssh-keygen -t rsa -b 4096 -C "your.email@domain.com"
cat ~/.ssh/id_rsa.pub
# Copy this key to GitHub SSH keys
```

### **Step 5: VS Code Remote SSH Setup**
Install "Remote - SSH" extension in VS Code, then connect to:
- Host: 64.227.16.180
- User: yc
- Password: yc

---

## 🚀 **Development Workflow:**

### **Daily Development:**
1. **Connect**: VS Code Remote SSH to DigitalOcean
2. **Develop**: Edit files directly on server
3. **Test**: Run development containers (ports 3001, 8001)
4. **Commit**: Git add/commit locally on server
5. **Push**: Git push to GitHub
6. **Deploy**: GitHub Actions auto-deploy to production (ports 3000, 8000)

### **Development Commands:**
```bash
# Start development environment
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml up -d

# View development dashboard
http://64.227.16.180:3001

# Test development API
http://64.227.16.180:8001/docs

# When ready, push changes
git add .
git commit -m "Your changes"
git push origin main
# GitHub Actions will deploy to production automatically
```

---

## 🔒 **Security & Separation:**

### **Development Environment:**
- **Path**: `/home/yc/development/Capstone-Project/`
- **Ports**: 3001 (dashboard), 8001 (API)
- **Purpose**: Active development, testing, iteration
- **Access**: Direct VS Code editing

### **Production Environment:**
- **Path**: `/root/Capstone-Project/` (or wherever GitHub Actions deploys)
- **Ports**: 3000 (dashboard), 8000 (API)
- **Purpose**: Stable production deployment
- **Access**: Only via GitHub Actions

---

## 📱 **Access URLs:**

### **Development (for testing):**
- Dashboard: http://64.227.16.180:3001
- API: http://64.227.16.180:8001
- SVM Debug: http://64.227.16.180:8001/api/v1/svm-debug/

### **Production (live system):**
- Dashboard: http://64.227.16.180:3000
- API: http://64.227.16.180:8000
- SVM Debug: http://64.227.16.180:8000/api/v1/svm-debug/

---

## 🛠 **Implementation Scripts:**

I'll create scripts to:
1. **Set up the development environment** on DigitalOcean
2. **Configure VS Code Remote SSH** connection
3. **Create development Docker configuration**
4. **Set up git workflow** with proper branching

Would you like me to create these setup scripts now?
