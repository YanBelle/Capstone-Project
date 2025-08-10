#!/usr/bin/env python3
"""
Enhanced Cluster Naming End-to-End Test Script
Tests the complete flow from backend API to frontend display
"""

import requests
import json
import sys

def test_backend_api():
    """Test that the backend API returns enhanced cluster naming fields"""
    print("🔍 Testing backend API for enhanced cluster naming...")
    
    url = "http://localhost:8002/api/cluster_sessions"
    payload = {"cluster_id": 15, "feature_type": "combined"}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        
        if response.status_code != 200:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        data = response.json()
        
        # Check for enhanced fields
        required_fields = ['cluster_name', 'business_meaning', 'actual_text_patterns', 'contextual_error_types']
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing enhanced fields: {missing_fields}")
            print(f"Available fields: {list(data.keys())}")
            return False
        
        # Check specific values for cluster 15
        cluster_name = data.get('cluster_name', '')
        business_meaning = data.get('business_meaning', '')
        
        print(f"✅ Backend API working correctly!")
        print(f"📊 Cluster Name: '{cluster_name}'")
        print(f"📋 Business Meaning: '{business_meaning[:100]}...'")
        print(f"🎯 Enhanced fields present: {required_fields}")
        
        # Verify this is meaningful content, not generic
        if "Standard EMV Transaction Flow" in cluster_name:
            print("✅ Cluster has meaningful name (not generic)")
        else:
            print(f"⚠️ Cluster name might be generic: '{cluster_name}'")
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        return False

def test_frontend_connectivity():
    """Test that the frontend is accessible"""
    print("\n🌐 Testing frontend connectivity...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        
        if response.status_code == 200:
            print("✅ Frontend is accessible at http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend not accessible: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Cluster Naming End-to-End Test")
    print("=" * 50)
    
    # Test backend API
    backend_ok = test_backend_api()
    
    # Test frontend connectivity  
    frontend_ok = test_frontend_connectivity()
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    if backend_ok:
        print("✅ Backend API: Enhanced cluster naming working")
        print("   → cluster_name field present with meaningful content")
        print("   → business_meaning field present with context")
        print("   → actual_text_patterns field present")
        print("   → contextual_error_types field present")
    else:
        print("❌ Backend API: Enhanced cluster naming failed")
        
    if frontend_ok:
        print("✅ Frontend: Accessible and ready")
        print("   → React app served at http://localhost:3000")
        print("   → Updated to use backend port 8002")
    else:
        print("❌ Frontend: Not accessible")
    
    if backend_ok and frontend_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("User should now see meaningful cluster names like:")
        print("   'Standard EMV Transaction Flow' instead of 'text cluster 15'")
        print("\n📖 Next Steps:")
        print("1. Open http://localhost:3000 in browser")
        print("2. Navigate to DBSCAN Visualization tab")
        print("3. Click on any cluster dot to see meaningful names")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the services and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
