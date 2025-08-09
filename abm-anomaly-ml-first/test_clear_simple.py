#!/usr/bin/env python3
"""
Direct test of clear data API endpoint
"""

import requests
import json
import time

def test_clear_data_direct():
    """Test clear data with proper error handling"""
    
    print("🧪 Testing Clear Data API Endpoint")
    print("=" * 40)
    
    # Wait for API
    print("⏳ Waiting for API...")
    for i in range(5):
        try:
            response = requests.get("http://localhost:8000", timeout=3)
            print("✅ API responding")
            break
        except:
            time.sleep(2)
    
    # Test clear data
    print("\n🧽 Testing clear data...")
    try:
        response = requests.delete(
            "http://localhost:8000/api/v1/clear-data?confirm=true",
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Tables cleared: {result.get('total_tables', 0)}")
            print(f"Files cleared: {result.get('total_file_groups', 0)}")
            print(f"Redis cleared: {result.get('redis_cleared', False)}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_clear_data_direct()
