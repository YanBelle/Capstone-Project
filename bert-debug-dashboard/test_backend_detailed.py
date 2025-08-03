#!/usr/bin/env python3
import requests
import time
import sys

def test_backend():
    base_url = "http://localhost:8000"
    
    print("=== Backend API Test ===")
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Failed: {e}")
        return False
    
    # Test 2: Health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 3: Model info endpoint
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/model_info", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 4: Analyze endpoint
    print("\n4. Testing analyze endpoint...")
    try:
        start_time = time.time()
        data = {"text": "Test EJ log message for analysis"}
        response = requests.post(f"{base_url}/api/analyze", data=data, timeout=60)
        end_time = time.time()
        
        print(f"   Status: {response.status_code}")
        print(f"   Time taken: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response keys: {list(result.keys())}")
            if 'predicted_class' in result:
                print(f"   Predicted class: {result['predicted_class']}")
            if 'analysis_time' in result:
                print(f"   Analysis time: {result['analysis_time']}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    # Wait for backend to start
    print("Waiting 10 seconds for backend to start...")
    time.sleep(10)
    test_backend()
