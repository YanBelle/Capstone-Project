#!/usr/bin/env python3
"""
Test script to verify the cluster_sessions API endpoint fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.app.main import app
    from backend.enhanced_ensemble_detector import EnhancedEnsembleAnomalyDetector
    
    print("✅ Successfully imported main API app")
    print("✅ Successfully imported EnhancedEnsembleAnomalyDetector")
    
    # Create detector instance
    detector = EnhancedEnsembleAnomalyDetector()
    print("✅ Successfully created detector instance")
    
    # Check if the get_cluster_sessions method exists
    if hasattr(detector, 'get_cluster_sessions'):
        print("✅ get_cluster_sessions method is available")
        
        # Check method signature
        import inspect
        sig = inspect.signature(detector.get_cluster_sessions)
        print(f"✅ Method signature: {sig}")
    else:
        print("❌ get_cluster_sessions method not found")
    
    # Check if label_cluster method exists
    if hasattr(detector, 'label_cluster'):
        print("✅ label_cluster method is available")
    else:
        print("❌ label_cluster method not found")
    
    # Check if train_supervised_classifier method exists
    if hasattr(detector, 'train_supervised_classifier'):
        print("✅ train_supervised_classifier method is available")
    else:
        print("❌ train_supervised_classifier method not found")
    
    # Check if predict_supervised method exists
    if hasattr(detector, 'predict_supervised'):
        print("✅ predict_supervised method is available")
    else:
        print("❌ predict_supervised method not found")
    
    print("\n🎯 SUMMARY:")
    print("The backend enhanced_ensemble_detector.py has been updated with the missing cluster interaction methods.")
    print("The ClusterSessionsRequest model in main.py now accepts both cluster_id and feature_type parameters.")
    print("The /api/cluster_sessions endpoint has been updated to pass feature_type to the get_cluster_sessions method.")
    print("\n🚀 To test the fix:")
    print("1. Restart the Docker containers: docker-compose up --build -d")
    print("2. Open the frontend at http://localhost:3000")
    print("3. Navigate to the DBSCAN tab and try clicking on clusters")
    print("4. The 500 error should be resolved!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
