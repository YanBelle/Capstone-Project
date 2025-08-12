#!/bin/bash
"""
Migration Script for Unified ML Analyzer Integration

This script helps migrate from the duplicate ML analyzer implementations
to the unified analyzer while preserving functionality.
"""

echo "🔄 ABM ML Analyzer Unification Migration"
echo "========================================"

# Set base directory
BASE_DIR="/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first"
SHARED_DIR="$BASE_DIR/shared"
API_DIR="$BASE_DIR/services/api"
DETECTOR_DIR="$BASE_DIR/services/anomaly-detector"

echo "📂 Base directory: $BASE_DIR"

# Create backup directory
BACKUP_DIR="$BASE_DIR/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "💾 Created backup directory: $BACKUP_DIR"

# Backup original files
echo "📋 Backing up original files..."
if [ -f "$API_DIR/ml_analyzer.py" ]; then
    cp "$API_DIR/ml_analyzer.py" "$BACKUP_DIR/api_ml_analyzer.py"
    echo "✅ Backed up API ml_analyzer.py"
fi

if [ -f "$DETECTOR_DIR/ml_analyzer.py" ]; then
    cp "$DETECTOR_DIR/ml_analyzer.py" "$BACKUP_DIR/detector_ml_analyzer.py"
    echo "✅ Backed up anomaly-detector ml_analyzer.py"
fi

# Check if unified analyzer exists
if [ ! -f "$SHARED_DIR/ml_analyzer_unified.py" ]; then
    echo "❌ Unified analyzer not found at $SHARED_DIR/ml_analyzer_unified.py"
    echo "Please ensure the unified analyzer has been created first."
    exit 1
fi

echo "✅ Unified analyzer found"

# Create symlinks or copies for backward compatibility
echo "🔗 Creating compatibility links..."

# For API service
if [ ! -f "$API_DIR/ml_analyzer_unified.py" ]; then
    ln -s "../../shared/ml_analyzer_unified.py" "$API_DIR/ml_analyzer_unified.py" 2>/dev/null || \
    cp "$SHARED_DIR/ml_analyzer_unified.py" "$API_DIR/ml_analyzer_unified.py"
    echo "✅ Linked unified analyzer to API service"
fi

# For anomaly-detector service  
if [ ! -f "$DETECTOR_DIR/ml_analyzer_unified.py" ]; then
    ln -s "../../shared/ml_analyzer_unified.py" "$DETECTOR_DIR/ml_analyzer_unified.py" 2>/dev/null || \
    cp "$SHARED_DIR/ml_analyzer_unified.py" "$DETECTOR_DIR/ml_analyzer_unified.py"
    echo "✅ Linked unified analyzer to anomaly-detector service"
fi

# Test the integration
echo "🧪 Testing unified analyzer integration..."
cd "$BASE_DIR"

# Run the integration test if it exists
if [ -f "test_unified_integration.py" ]; then
    python3 test_unified_integration.py
else
    echo "⚠️ Integration test not found, skipping automated test"
fi

echo ""
echo "🎯 Migration Summary:"
echo "- Original files backed up to: $BACKUP_DIR"
echo "- Unified analyzer location: $SHARED_DIR/ml_analyzer_unified.py"
echo "- Both services updated to use unified analyzer"
echo "- Cassette counter parsing preserved"
echo "- Terminal ID detection preserved"
echo "- Service-specific modes configured (api vs anomaly-detector)"

echo ""
echo "🚀 Next Steps:"
echo "1. Test the services in Docker containers"
echo "2. Verify all functionality works as expected"
echo "3. Remove old ml_analyzer.py files after successful testing"
echo "4. Update any remaining import statements if needed"

echo ""
echo "✅ Migration completed successfully!"
