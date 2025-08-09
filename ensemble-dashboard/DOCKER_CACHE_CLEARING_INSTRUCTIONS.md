# Docker Cache Clearing Instructions for Ensemble Dashboard

## Prerequisites
1. **Start Docker Desktop** - Ensure Docker Desktop is running on your Mac
2. **Open Terminal** - Navigate to the ensemble-dashboard directory

## Option 1: Use the Automated Script
Once Docker is running, execute:
```bash
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard
./clear_docker_cache.sh
```

## Option 2: Manual Step-by-Step Commands

### 1. Navigate to the project directory
```bash
cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/ensemble-dashboard
```

### 2. Stop and remove containers
```bash
# Stop all containers
docker compose down
# OR if using older Docker version:
# docker-compose down

# Remove containers
docker compose rm -f
# OR: docker-compose rm -f
```

### 3. Remove built images to force rebuild
```bash
# Remove specific images
docker rmi ensemble-dashboard-backend:latest 2>/dev/null || true
docker rmi ensemble-dashboard-frontend:latest 2>/dev/null || true
docker rmi ensemble-backend:latest 2>/dev/null || true
docker rmi ensemble-frontend:latest 2>/dev/null || true

# Remove all ensemble-related images
docker images | grep -E "(ensemble|dashboard)" | awk '{print $3}' | xargs docker rmi -f
```

### 4. Clean Docker system
```bash
# Remove unused containers, networks, and images
docker system prune -f

# Remove build cache
docker builder prune -f

# Remove dangling images
docker image prune -f
```

### 5. Rebuild from scratch (no cache)
```bash
# Build without using cache
docker compose build --no-cache
# OR: docker-compose build --no-cache

# Start containers
docker compose up -d
# OR: docker-compose up -d
```

### 6. Verify containers are running
```bash
# Check container status
docker compose ps
# OR: docker-compose ps

# View logs
docker compose logs backend
docker compose logs frontend
# OR: docker-compose logs backend && docker-compose logs frontend
```

## Expected Outcome
After clearing the cache and rebuilding:

1. **Frontend (http://localhost:3000)** should show:
   - Console logs starting with "🔧 DBSCANVisualization enhanced version 2.0 loaded"
   - "🛡️ safeToFixed utility function loaded" message
   - Detailed API request/response logging

2. **Backend (http://localhost:8001)** should show:
   - Enhanced CORS headers (Access-Control-Allow-Origin: *)
   - Improved error handling and logging
   - Fixed numpy serialization issues

3. **Browser Console** should display:
   - All the debugging logs we added
   - No more "Cannot read properties of undefined (reading 'toFixed')" errors
   - Successful API calls with proper responses

## Troubleshooting

### If Docker Desktop is not running:
1. Open Docker Desktop application
2. Wait for it to fully start (Docker whale icon should be steady)
3. Run the commands above

### If containers won't start:
1. Check Docker Desktop has enough resources allocated
2. Verify ports 3000 and 8001 are not being used by other applications
3. Check the docker-compose.yml file is in the current directory

### If errors persist after cache clearing:
1. The cache clearing worked if you see the new console logs
2. If you still don't see logs, try a hard browser refresh (Cmd+Shift+R)
3. Check browser developer tools Network tab for failed requests

## Files Modified with Fixes
All these fixes should take effect after cache clearing:

1. **DBSCANVisualization.jsx** - Added safeToFixed utility and comprehensive logging
2. **main.py** - Enhanced CORS, numpy serialization, and error handling  
3. **Overview.js** - Added request/response logging and error handling
4. **debug-console.html** - Created testing tool for verification

The Docker cache was preventing these fixes from loading. Once cleared, all improvements should be visible immediately.
