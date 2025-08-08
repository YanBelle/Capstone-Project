#!/usr/bin/env python3
"""
Test script for enhanced clear data functionality
"""

import requests
import json
import time

def test_clear_data():
    """Test the enhanced clear data endpoint"""
    
    print("🧪 Testing Enhanced Clear Data Functionality")
    print("=" * 50)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    for i in range(10):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready")
                break
        except:
            print(f"   Attempt {i+1}/10 - API not ready yet...")
            time.sleep(3)
    else:
        print("❌ API failed to start in time")
        return
    
    # Test without confirmation (should fail)
    print("\n📋 Test 1: Clear data without confirmation")
    try:
        response = requests.delete("http://localhost:8000/api/v1/clear-data", timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 400:
            print("✅ Correctly rejected request without confirmation")
        else:
            print(f"❌ Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test with confirmation (should succeed)
    print("\n📋 Test 2: Clear data with confirmation")
    try:
        response = requests.delete(
            "http://localhost:8000/api/v1/clear-data?confirm=true", 
            timeout=60
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Clear data successful!")
            print(f"   Database tables cleared: {result.get('total_tables', 0)}")
            print(f"   File groups cleared: {result.get('total_file_groups', 0)}")
            print(f"   Redis cleared: {result.get('redis_cleared', False)}")
            
            # Print details
            if 'database_tables_cleared' in result:
                print(f"   Tables: {', '.join(result['database_tables_cleared'])}")
            if 'files_cleared' in result:
                print(f"   Files: {', '.join(result['files_cleared'])}")
        else:
            print(f"❌ Clear failed: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_clear_data()
