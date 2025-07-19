#!/usr/bin/env python3
"""
Simple diagnostic script to check retraining functionality
Uses only standard library - no external dependencies
"""

import urllib.request
import urllib.error
import json
import sys

def check_api_health():
    """Check if API is responding"""
    try:
        with urllib.request.urlopen("http://localhost:8000/api/v1/health", timeout=5) as response:
            if response.getcode() == 200:
                print("✅ API is responding")
                return True
            else:
                print(f"❌ API responded with status: {response.getcode()}")
                return False
    except Exception as e:
        print(f"❌ API not responding: {e}")
        return False

def test_retraining():
    """Test retraining endpoint"""
    try:
        req = urllib.request.Request(
            "http://localhost:8000/api/v1/continuous-learning/trigger-retraining",
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                print("✅ Retraining endpoint is working!")
                print(f"Response: {data}")
                return True
            else:
                print(f"❌ Retraining failed with status: {response.getcode()}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing retraining: {e}")
        return False

def main():
    print("🧪 Testing ABM ML-First Anomaly Detection API")
    print("=" * 50)
    
    if not check_api_health():
        print("\n❌ API is not responding. Please run:")
        print("   cd /Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first")
        print("   chmod +x fix_retraining.sh")
        print("   ./fix_retraining.sh")
        return
    
    if test_retraining():
        print("\n🎉 All tests passed! Retraining functionality is working.")
        print("You can now use the 'Trigger Retraining' button in the dashboard at:")
        print("http://localhost:3000")
    else:
        print("\n❌ Retraining test failed. Please run the fix script:")
        print("   ./fix_retraining.sh")

if __name__ == "__main__":
    main()
