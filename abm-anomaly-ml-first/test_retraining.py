#!/usr/bin/env python3
"""
Test script to verify retraining functionality
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is responding"""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        print(f"API Health Check: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is responding")
            return True
        else:
            print(f"❌ API responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not responding: {e}")
        return False

def test_retraining_endpoint():
    """Test the retraining endpoint"""
    try:
        print("🔄 Testing retraining endpoint...")
        response = requests.post(
            f"{API_URL}/api/v1/continuous-learning/trigger-retraining",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Retraining endpoint is working!")
            return True
        else:
            print(f"❌ Retraining failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing retraining: {e}")
        return False

def main():
    print("🧪 Testing ABM ML-First Anomaly Detection API")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\n❌ API is not responding. Please check if the containers are running:")
        print("   docker compose up -d")
        return
    
    # Test retraining
    if test_retraining_endpoint():
        print("\n🎉 All tests passed! Retraining functionality is working.")
    else:
        print("\n❌ Retraining test failed. Check the API logs for details.")

if __name__ == "__main__":
    main()
