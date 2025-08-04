#!/usr/bin/env python3
"""
Quick API Test Script - Verify fixes are working
"""

import requests
import json
import time

def test_backend():
    """Test backend API endpoints"""
    print("🧪 TESTING BACKEND API")
    print("=" * 30)
    
    base_url = "http://localhost:8001"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint not reachable: {e}")
        return False
    
    # Test 2: Model info
    try:
        response = requests.get(f"{base_url}/api/model_info", timeout=5)
        if response.status_code == 200:
            print("✅ Model info endpoint working")
            data = response.json()
            print(f"   Model trained: {data.get('model_info', {}).get('is_trained', False)}")
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info endpoint error: {e}")
    
    # Test 3: Cluster sessions (should fail gracefully if model not trained)
    try:
        test_payload = {
            "cluster_id": 1,
            "feature_type": "combined"
        }
        response = requests.post(f"{base_url}/api/cluster_sessions", 
                               json=test_payload, timeout=5)
        if response.status_code == 400:
            print("✅ Cluster sessions endpoint working (model not trained)")
            data = response.json()
            print(f"   Message: {data.get('detail')}")
        elif response.status_code == 200:
            print("✅ Cluster sessions endpoint working (model trained)")
            data = response.json()
            print(f"   Sessions found: {data.get('count', 0)}")
        else:
            print(f"⚠️  Cluster sessions unexpected status: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cluster sessions endpoint error: {e}")
    
    # Test 4: API docs
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            print(f"   URL: {base_url}/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs error: {e}")
    
    return True

def test_frontend():
    """Test frontend accessibility"""
    print("\n🌐 TESTING FRONTEND")
    print("=" * 20)
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend accessible")
            print("   URL: http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend not reachable: {e}")
        return False

def main():
    print("🔍 CLUSTER SESSIONS FIX VERIFICATION")
    print("=" * 40)
    print()
    
    backend_ok = test_backend()
    frontend_ok = test_frontend()
    
    print("\n📊 SUMMARY")
    print("=" * 10)
    
    if backend_ok and frontend_ok:
        print("🎉 All services are running!")
        print()
        print("🚀 Next steps:")
        print("1. Open http://localhost:3000")
        print("2. Navigate to DBSCAN tab")  
        print("3. If model not trained, go to Training tab first")
        print("4. Try clicking on clusters - should work now!")
    elif backend_ok:
        print("⚠️  Backend OK, but frontend needs attention")
        print("Try starting frontend manually with:")
        print("./start_frontend_manual.sh")
    else:
        print("❌ Backend needs attention")
        print("Try restarting with:")
        print("docker-compose up --build -d")
        print("OR")
        print("./start_backend_manual.sh")

if __name__ == "__main__":
    main()
