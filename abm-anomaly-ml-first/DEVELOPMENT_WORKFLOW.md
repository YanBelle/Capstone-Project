# Development Workflow Setup for Local → DigitalOcean

## 🎯 **Optimal Workflow for Your Setup**

Since you don't have Docker locally but have it on DigitalOcean, here's the most efficient development workflow:

### **Current Situation:**
- ✅ Local development on Windows (no Docker)
- ✅ DigitalOcean server at 64.227.16.180 (with Docker)
- ✅ GitHub repository for version control
- ✅ SVM debug system ready to deploy

---

## 🚀 **Recommended Workflow: Quick SSH Deploy**

### **Step 1: One-Time Setup**
```powershell
# Generate SSH key (if not exists)
ssh-keygen -t rsa -b 4096 -C "your_email@domain.com"

# Copy key to server (replace with your actual path)
ssh-copy-id -i ~/.ssh/id_rsa root@64.227.16.180
```

### **Step 2: Daily Development Cycle**
```powershell
# 1. Code locally in VS Code
# 2. Quick deploy to test
.\deploy-svm-debug.ps1

# 3. Test immediately
.\test-remote.ps1 -QuickTest

# 4. View logs if needed
.\test-remote.ps1 -ViewLogs

# 5. Iterate (repeat steps 1-4)
```

### **Step 3: Final Commit** (when feature is working)
```powershell
git add .
git commit -m "Add SVM debug system"
git push origin main
```

---

## 📋 **Available Scripts Created:**

### 1. **`quick-deploy.ps1`** - Full deployment script
- Syncs all files to server
- Installs dependencies
- Restarts services
- Tests endpoints

### 2. **`deploy-svm-debug.ps1`** - SVM-specific deployment
- Deploys only SVM debug components
- Faster than full deployment
- Perfect for iterating on SVM features

### 3. **`test-remote.ps1`** - Remote testing
- Tests all endpoints without local Docker
- Quick health checks
- Full SVM functionality testing

---

## 🔧 **Usage Examples:**

### **Quick SVM Debug Deployment:**
```powershell
# Deploy SVM debug system
.\deploy-svm-debug.ps1

# Test it works
.\test-remote.ps1 -QuickTest

# View the dashboard
# Open: http://64.227.16.180:3000
# Click: "SVM Debug" tab
```

### **Full Development Cycle:**
```powershell
# Deploy everything
.\quick-deploy.ps1

# Run comprehensive tests
.\test-remote.ps1 -FullTest

# Check service status
.\test-remote.ps1 -CheckStatus

# View logs
.\test-remote.ps1 -ViewLogs
```

### **Component-Specific Updates:**
```powershell
# API changes only
.\quick-deploy.ps1 -OnlyAPI

# Dashboard changes only
.\quick-deploy.ps1 -OnlyDashboard

# ML model changes only
.\quick-deploy.ps1 -OnlyML
```

---

## 🌐 **Access Your Deployed System:**

Once deployed, access these URLs:

- **🎨 Dashboard**: http://64.227.16.180:3000
- **🔍 SVM Debug Tab**: http://64.227.16.180:3000 (click "SVM Debug")
- **🔧 API Documentation**: http://64.227.16.180:8000/docs
- **⚙️ SVM Debug API**: http://64.227.16.180:8000/api/v1/svm-debug/

---

## 💡 **Pro Tips:**

### **Faster Iteration:**
1. Use `deploy-svm-debug.ps1` for SVM-only changes
2. Use `test-remote.ps1 -QuickTest` for fast validation
3. Keep a browser tab open to your dashboard for immediate visual feedback

### **Debugging Issues:**
1. `.\test-remote.ps1 -ViewLogs` - See what's happening
2. `.\test-remote.ps1 -CheckStatus` - Check service health
3. `.\quick-deploy.ps1 -RestartAll` - Nuclear option (restart everything)

### **Version Control:**
1. Develop and test with quick deploy scripts
2. Only commit to git when features are working
3. Use GitHub Actions for production deployments

---

## 🎯 **Ready to Test Your SVM Debug System?**

Run this now:
```powershell
.\deploy-svm-debug.ps1
```

Then open: **http://64.227.16.180:3000** and click the **"SVM Debug"** tab!

This workflow gives you the best of both worlds:
- ✅ **Fast iteration** (no local Docker needed)
- ✅ **Real testing environment** (actual Docker containers)
- ✅ **Professional deployment** (when ready, use GitHub Actions)
- ✅ **Easy debugging** (remote logs and testing)
