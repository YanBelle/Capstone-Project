#!/bin/bash

echo "Updating Dashboard to include SVM Debug functionality..."

# Update the main Dashboard component to include SVM debug tab
DASHBOARD_FILE="services/dashboard/src/Dashboard.js"

# Check if Dashboard file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "Dashboard file not found at $DASHBOARD_FILE. Please ensure the file exists."
    exit 1
fi

# Add SVM Debug import and tab to Dashboard
cat > dashboard_svm_integration.patch << 'PATCH'
--- Dashboard.js.orig
+++ Dashboard.js
@@ -3,6 +3,7 @@
 import { AlertCircle, Activity, TrendingUp, Clock, Shield, Database, Brain } from 'lucide-react';
 import ExpertLabelingInterface from './ExpertLabelingInterface';
+import SVMDebugDashboard from './SVMDebugDashboard';
 
 const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
 
@@ -334,7 +335,7 @@
       {/* Navigation Tabs */}
       <div className="bg-white border-b">
         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
           <div className="flex space-x-8">
-            {['overview', 'anomalies', 'alerts', 'expert-labeling', 'analytics'].map((tab) => (
+            {['overview', 'anomalies', 'alerts', 'expert-labeling', 'analytics', 'svm-debug'].map((tab) => (
               <button
                 key={tab}
                 onClick={() => setActiveTab(tab)}
@@ -345,7 +346,11 @@
                     : 'border-transparent text-gray-500 hover:text-gray-700'
                 }`}
               >
-                {tab === 'expert-labeling' ? 'Expert Review' : tab}
+                {tab === 'expert-labeling' 
+                  ? 'Expert Review' 
+                  : tab === 'svm-debug' 
+                  ? 'SVM Debug' 
+                  : tab}
               </button>
             ))}
           </div>
@@ -584,6 +589,10 @@
         {activeTab === 'expert-labeling' && (
           <ExpertLabelingInterface />
         )}
+
+        {activeTab === 'svm-debug' && (
+          <SVMDebugDashboard />
+        )}
 
         {activeTab === 'analytics' && (
           <div className="space-y-6">
PATCH

# Apply the patch manually by updating the Dashboard.js file
# First, backup the original
cp "$DASHBOARD_FILE" "${DASHBOARD_FILE}.backup"

# Add SVM Debug import to the imports section
sed -i '/import ExpertLabelingInterface/a import SVMDebugDashboard from '\''./SVMDebugDashboard'\'';' "$DASHBOARD_FILE"

# Add svm-debug to the navigation tabs array
sed -i "s/\['overview', 'anomalies', 'alerts', 'expert-labeling', 'analytics'\]/['overview', 'anomalies', 'alerts', 'expert-labeling', 'analytics', 'svm-debug']/" "$DASHBOARD_FILE"

# Update the tab label logic
sed -i '/tab === '\''expert-labeling'\'' ? '\''Expert Review'\'' : tab/c\
                tab === '\''expert-labeling'\'' \
                  ? '\''Expert Review'\'' \
                  : tab === '\''svm-debug'\'' \
                  ? '\''SVM Debug'\'' \
                  : tab' "$DASHBOARD_FILE"

# Add the SVM debug tab content
sed -i '/activeTab === '\''expert-labeling'\'' && (/,/)}/ {
/)}$/a\
\
        {activeTab === '\''svm-debug'\'' && (\
          <SVMDebugDashboard />\
        )}
}' "$DASHBOARD_FILE"

echo "✓ Dashboard updated with SVM Debug tab"

# Create a comprehensive installation script
cat > install_svm_debug_system.sh << 'INSTALL'
#!/bin/bash

echo "Installing SVM Visualization and Debugging System..."

# Step 1: Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Create necessary directories
echo "Creating directories..."
mkdir -p /app/debug_output
mkdir -p /app/static/debug
mkdir -p services/anomaly-detector
mkdir -p services/api
mkdir -p services/dashboard/src

# Step 3: Copy SVM visualizer to anomaly detector service
if [ -f "svm_visualizer.py" ]; then
    cp svm_visualizer.py services/anomaly-detector/
    echo "✓ SVM visualizer copied to anomaly detector service"
fi

# Step 4: Copy SVM debug API to API service
if [ -f "svm_debug_api.py" ]; then
    cp svm_debug_api.py services/api/
    echo "✓ SVM debug API copied to API service"
fi

# Step 5: Copy React dashboard component
if [ -f "SVMDebugDashboard.js" ]; then
    cp SVMDebugDashboard.js services/dashboard/src/
    echo "✓ SVM debug dashboard component copied"
fi

# Step 6: Update Docker Compose to include visualization dependencies
echo "Updating Docker Compose configuration..."

# Add volume for debug output
if ! grep -q "debug_output" docker-compose.yml; then
    sed -i '/volumes:/a\      - ./debug_output:/app/debug_output' docker-compose.yml
fi

# Add static files volume
if ! grep -q "static/debug" docker-compose.yml; then
    sed -i '/volumes:/a\      - ./static:/app/static' docker-compose.yml
fi

# Step 7: Test the installation
echo "Testing SVM debug system..."

# Test CLI tool
if [ -f "debug_svm_cli.py" ] && [ -f "example_sessions.json" ]; then
    echo "Testing CLI debug tool..."
    python debug_svm_cli.py --session-file example_sessions.json --output-dir ./test_debug_output --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ CLI debug tool test passed"
    else
        echo "✗ CLI debug tool test failed"
    fi
fi

# Step 8: Create documentation
cat > SVM_DEBUG_USAGE.md << 'USAGE'
# SVM Debug System Usage Guide

## Overview
The SVM Visualization and Debugging System provides comprehensive tools for understanding and debugging One-Class SVM anomaly detection decisions.

## Components

### 1. SVM Visualizer (svm_visualizer.py)
- **Purpose**: Creates interactive visualizations of SVM decision boundaries and data points
- **Features**: 
  - 2D decision boundary plots using PCA
  - Parameter sensitivity analysis
  - Feature importance analysis
  - Comprehensive HTML reports

### 2. SVM Debug API (svm_debug_api.py)
- **Purpose**: REST API endpoints for SVM debugging
- **Endpoints**:
  - `POST /api/v1/svm-debug/analyze-session` - Debug specific session
  - `GET /api/v1/svm-debug/model-info` - Get model information
  - `POST /api/v1/svm-debug/batch-analyze` - Analyze multiple sessions
  - `GET /api/v1/svm-debug/performance-metrics` - Get performance metrics
  - `POST /api/v1/svm-debug/tune-parameters` - Auto-tune parameters

### 3. React Dashboard (SVMDebugDashboard.js)
- **Purpose**: Interactive web interface for SVM debugging
- **Features**:
  - Real-time session analysis
  - Performance monitoring
  - Parameter tuning interface
  - Visual decision boundary exploration

### 4. CLI Debug Tool (debug_svm_cli.py)
- **Purpose**: Command-line interface for batch debugging
- **Usage**: `python debug_svm_cli.py --session-file sessions.json --output-dir ./debug_output`

## Quick Start

### 1. Web Interface
1. Start the system: `docker-compose up`
2. Navigate to: `http://localhost:3000`
3. Click on "SVM Debug" tab
4. Enter a session ID to analyze

### 2. API Usage
```bash
# Debug a specific session
curl -X POST http://localhost:8000/api/v1/svm-debug/analyze-session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_session_1", "raw_text": "sample text", "include_visualization": true}'

# Get model information
curl http://localhost:8000/api/v1/svm-debug/model-info

# Get performance metrics
curl http://localhost:8000/api/v1/svm-debug/performance-metrics
```

### 3. CLI Usage
```bash
# Debug sessions from JSON file
python debug_svm_cli.py --session-file example_sessions.json --output-dir ./debug_output

# Debug specific session with verbose output
python debug_svm_cli.py --session-file sessions.json --session-id "session_123" --verbose

# Run parameter tuning
python debug_svm_cli.py --session-file sessions.json --tune-parameters
```

## Understanding SVM Decisions

### Decision Scores
- **Positive scores**: Normal behavior (inside decision boundary)
- **Negative scores**: Anomalous behavior (outside decision boundary)
- **Magnitude**: Distance from decision boundary (confidence)

### Visualization Elements
- **Blue points**: Normal sessions
- **Red points**: Anomalous sessions
- **Contour lines**: Decision boundary levels
- **Feature contributions**: Which features influence decisions most

### Parameter Impact
- **Nu**: Controls the fraction of outliers (lower = fewer anomalies detected)
- **Gamma**: Controls decision boundary smoothness (higher = more complex boundaries)

## Troubleshooting

### Common Issues
1. **"No embeddings found"**: Ensure sessions have valid text and BERT is working
2. **"Model not fitted"**: Run anomaly detection first to train the SVM
3. **"Visualization failed"**: Check that all dependencies are installed

### Dependencies
Ensure these packages are installed:
- plotly>=5.17.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- scikit-learn>=1.3.0
- numpy>=1.24.0
- pandas>=2.1.0

## Advanced Features

### Custom Parameter Tuning
```python
# In your ML analyzer
tuning_results = analyzer.tune_svm_parameters(
    nu_range=[0.01, 0.05, 0.1, 0.2],
    gamma_range=['scale', 0.001, 0.01, 0.1]
)
```

### Real-time Monitoring
```python
# Monitor SVM performance
performance = analyzer.monitor_svm_performance()
print(f"Anomaly rate: {performance['anomaly_rate']:.2%}")
```

### Debug Specific Sessions
```python
# Debug individual session
debug_info = analyzer.real_time_svm_debug(session)
print(f"Decision score: {debug_info['decision_score']}")
```
USAGE

echo "✓ Documentation created: SVM_DEBUG_USAGE.md"

echo ""
echo "🎉 SVM Visualization and Debugging System Installation Complete!"
echo ""
echo "📁 Files Created:"
echo "   - svm_visualizer.py (Core visualization engine)"
echo "   - svm_debug_api.py (REST API endpoints)"
echo "   - SVMDebugDashboard.js (React component)"
echo "   - debug_svm_cli.py (Command-line tool)"
echo "   - example_sessions.json (Test data)"
echo "   - SVM_DEBUG_USAGE.md (Documentation)"
echo ""
echo "🚀 Next Steps:"
echo "   1. Restart your Docker containers: docker-compose restart"
echo "   2. Access SVM Debug tab in dashboard: http://localhost:3000"
echo "   3. Test CLI tool: python debug_svm_cli.py --session-file example_sessions.json"
echo "   4. Read documentation: cat SVM_DEBUG_USAGE.md"
echo ""
echo "🔧 API Endpoints Available:"
echo "   - POST /api/v1/svm-debug/analyze-session"
echo "   - GET /api/v1/svm-debug/model-info"
echo "   - POST /api/v1/svm-debug/batch-analyze"
echo "   - GET /api/v1/svm-debug/performance-metrics"
echo "   - POST /api/v1/svm-debug/tune-parameters"
echo ""
INSTALL

chmod +x install_svm_debug_system.sh

echo "✅ SVM Debug System Implementation Complete!"
echo ""
echo "📦 Created Files:"
echo "   - implement_svm_debug.sh (Main implementation script)"
echo "   - install_svm_debug_system.sh (Installation script)"
echo "   - dashboard_svm_integration.patch (Dashboard integration)"
echo ""
echo "🎯 To Install and Use:"
echo "   1. Run: ./implement_svm_debug.sh"
echo "   2. Run: ./install_svm_debug_system.sh"
echo "   3. Restart containers: docker-compose restart"
echo "   4. Access SVM Debug tab in dashboard"
echo ""
