#!/usr/bin/env python3
"""
Simple test to verify enhanced preprocessing methodology integration
"""

import requests
import json

def test_initialize():
    """Test if we can initialize the analyzer directly"""
    print("=== Testing Analyzer Initialization ===")
    
    # First try to get current status
    try:
        response = requests.get("http://localhost:8000/api/v1/bert-deeplog/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status endpoint working: {status}")
        else:
            print(f"Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"Status endpoint error: {e}")
    
    # Try to reinitialize
    try:
        response = requests.post("http://localhost:8000/api/v1/bert-deeplog/initialize")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Initialize successful: {result}")
            return True
        else:
            print(f"Initialize failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Initialize error: {e}")
        return False

def test_simple_preprocessing():
    """Test the preprocessing directly through prediction"""
    print("\n=== Testing Simple Preprocessing ===")
    
    # Use a very simple session that should work
    simple_session = {
        "session_text": "TRANSACTION START CARD INSERTED PIN ENTERED CASH DISPENSED CARD TAKEN TRANSACTION END",
        "session_id": "simple_test_001"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/bert-deeplog/predict",
            json=simple_session
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Simple preprocessing test successful")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Simple preprocessing error: {e}")
        return False

def main():
    print("🔧 Simple BERT-DeepLog Integration Test")
    print("=" * 50)
    
    # Test initialization
    if test_initialize():
        print("✅ Initialization successful")
    else:
        print("❌ Initialization failed")
    
    # Test simple preprocessing
    if test_simple_preprocessing():
        print("✅ Preprocessing working")
    else:
        print("❌ Preprocessing failed")
    
    print("\n" + "=" * 50)
    print("🔧 Simple test complete")

if __name__ == "__main__":
    main()
