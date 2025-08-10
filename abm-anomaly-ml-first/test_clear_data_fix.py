#!/usr/bin/env python3
"""
Test script to verify the clear data foreign key constraint fix
"""

import requests
import time
import sys

def test_clear_data_endpoint():
    """Test the clear data endpoint"""
    
    print("🧪 Testing Clear Data Foreign Key Constraint Fix")
    print("=" * 50)
    
    # Test 1: Call without confirm parameter (should fail)
    print("\n1. Testing without confirmation parameter...")
    try:
        response = requests.delete("http://localhost/api/v1/clear-data")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
        if response.status_code == 400:
            print("   ✅ Correctly requires confirmation")
        else:
            print("   ❌ Should require confirmation")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Call with confirm=true parameter
    print("\n2. Testing with confirmation parameter...")
    try:
        response = requests.delete("http://localhost/api/v1/clear-data?confirm=true")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result.get('message', 'N/A')}")
            print(f"   Tables cleared: {result.get('tables_cleared', [])}")
            print(f"   Total tables: {result.get('total_tables', 0)}")
        else:
            print(f"   ❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_clear_data_endpoint()
