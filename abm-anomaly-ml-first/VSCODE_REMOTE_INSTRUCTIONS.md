# VS Code Remote SSH Configuration
# Add this to your VS Code SSH config for easy connection

## 🔧 **VS Code Remote SSH Setup Instructions**

### **Step 1: Install VS Code Extension**
1. Open VS Code
2. Install "Remote - SSH" extension
3. Install "Remote - SSH: Editing Configuration Files" extension

### **Step 2: Configure SSH Connection**
1. Press `Ctrl+Shift+P` (Windows) to open command palette
2. Type "Remote-SSH: Open SSH Configuration File"
3. Select your SSH config file (usually `C:\Users\YourName\.ssh\config`)
4. Add this configuration:

```ssh
# DigitalOcean Development Environment
Host digitalocean-dev
    HostName 64.227.16.180
    User yc
    Port 22
    PasswordAuthentication yes
    PreferredAuthentications password,keyboard-interactive
    
# Alternative connection (if needed)
Host do-dev
    HostName 64.227.16.180
    User yc
    Port 22
```

### **Step 3: Connect to Server**
1. Press `Ctrl+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Select "digitalocean-dev"
4. Enter password: `yc`
5. Wait for VS Code to connect and install VS Code Server

### **Step 4: Open Development Folder**
1. Once connected, click "Open Folder"
2. Navigate to: `/home/yc/development/Capstone-Project/abm-anomaly-ml-first`
3. Click "OK"

### **Step 5: Install Extensions on Remote**
Install these VS Code extensions on the remote server:
- Python
- Docker
- GitLens
- ES7+ React/Redux/React-Native snippets
- JavaScript and TypeScript Nightly
- Prettier - Code formatter

---

## 🚀 **Your Development Workflow**

### **Daily Workflow:**
1. **Connect**: VS Code Remote SSH to digitalocean-dev
2. **Start Environment**: Open terminal and run `dev-start`
3. **Develop**: Edit files directly in VS Code
4. **Test**: Access http://64.227.16.180:3001 (development)
5. **Commit & Push**: Run `dev-push` when ready
6. **Production Deploy**: GitHub Actions automatically deploys to production

### **Terminal Commands Available:**
```bash
# Quick commands (available after SSH connection)
dev-start    # Start development containers
dev-stop     # Stop development containers  
dev-status   # Check container status
dev-push     # Git add, commit, and push to GitHub
dev-cd       # Navigate to project directory

# Manual commands
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first
docker-compose -f docker-compose.dev.yml up -d     # Start dev environment
docker-compose -f docker-compose.dev.yml down      # Stop dev environment
docker-compose -f docker-compose.dev.yml logs      # View logs
```

### **Environment URLs:**

#### **Development Environment** (for testing):
- 🎨 Dashboard: http://64.227.16.180:3001
- 🔧 API: http://64.227.16.180:8001
- 📖 API Docs: http://64.227.16.180:8001/docs
- 🔍 SVM Debug: http://64.227.16.180:8001/api/v1/svm-debug/

#### **Production Environment** (live system):
- 🎨 Dashboard: http://64.227.16.180:3000
- 🔧 API: http://64.227.16.180:8000
- 📖 API Docs: http://64.227.16.180:8000/docs
- 🔍 SVM Debug: http://64.227.16.180:8000/api/v1/svm-debug/

---

## 🔒 **Security & Separation**

### **Development Environment:**
- **Path**: `/home/yc/development/Capstone-Project/`
- **User**: `yc` (non-root)
- **Ports**: 3001, 8001, 5433, 6380
- **Database**: `abmdb_dev`
- **Purpose**: Active development and testing

### **Production Environment:**
- **Path**: `/root/Capstone-Project/` (deployed by GitHub Actions)
- **User**: `root` (managed by GitHub Actions)
- **Ports**: 3000, 8000, 5432, 6379
- **Database**: `abmdb`
- **Purpose**: Live production system

**No direct editing of production** - only through GitHub Actions!

---

## ⚡ **Quick Start Guide**

### **First Time Setup:**
1. Run setup script on DigitalOcean server
2. Configure VS Code SSH (instructions above)
3. Connect to remote server
4. Open development folder
5. Run `dev-start` in terminal

### **Daily Development:**
1. Connect VS Code Remote SSH
2. `dev-start` (if not running)
3. Edit code in VS Code
4. Test at http://64.227.16.180:3001
5. `dev-push` when ready
6. GitHub Actions deploys to production

### **Troubleshooting:**
- Can't connect SSH? Check password: `yc`
- Containers not starting? Run `dev-status`
- Need to restart? Run `dev-stop` then `dev-start`
- Git issues? Check with `git status` in project directory

---

## 🎯 **Benefits of This Setup:**

✅ **Real Docker Environment**: Full development environment with Docker
✅ **Immediate Testing**: No deployment delays, instant feedback
✅ **Git Workflow**: Proper version control with GitHub integration
✅ **Production Separation**: Dev and prod environments isolated
✅ **VS Code Integration**: Full IDE experience on remote server
✅ **Automatic Deployment**: GitHub Actions handle production deployment
✅ **Team Collaboration**: Multiple developers can use same setup

This setup gives you the best development experience while maintaining proper separation between development and production environments!
