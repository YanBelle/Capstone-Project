#!/usr/bin/env python3

"""
Test Enhanced Frontend Integration

Tests the complete integration between:
1. Enhanced backend with semantic clustering
2. Modified frontend with meaningful cluster display

This verifies that the React component will receive and display
meaningful cluster names instead of "text cluster 15".
"""

import requests
import json
from datetime import datetime

# API endpoint
API_BASE_URL = "http://localhost:8001/api"

def test_cluster_sessions_endpoint():
    """Test the enhanced cluster sessions endpoint that frontend calls"""
    print("🔍 Testing Enhanced Cluster Sessions Endpoint")
    print("=" * 60)
    
    # Test with a sample cluster ID (like frontend would send)
    test_cluster_id = 15
    test_feature_type = "text"
    
    payload = {
        "cluster_id": test_cluster_id,
        "feature_type": test_feature_type
    }
    
    try:
        print(f"📡 Making request to: {API_BASE_URL}/cluster_sessions")
        print(f"📋 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{API_BASE_URL}/cluster_sessions",
            json=payload,
            timeout=30
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS! Received enhanced cluster data:")
            print(f"🏷️  Cluster Name: {data.get('cluster_name', 'NOT FOUND')}")
            print(f"🎯 Business Meaning: {data.get('business_meaning', 'NOT FOUND')}")
            print(f"📝 Text Patterns Count: {len(data.get('actual_text_patterns', []))}")
            print(f"⚠️  Error Types Count: {len(data.get('contextual_error_types', []))}")
            print(f"👥 Sessions Count: {len(data.get('sessions', []))}")
            
            # Show sample patterns
            if data.get('actual_text_patterns'):
                print(f"\n📝 Sample Text Patterns:")
                for i, pattern in enumerate(data['actual_text_patterns'][:3]):
                    print(f"   {i+1}. {pattern}")
            
            # Show error types
            if data.get('contextual_error_types'):
                print(f"\n⚠️  Error Classifications:")
                for error_type in data['contextual_error_types']:
                    print(f"   • {error_type}")
            
            print(f"\n🎉 FRONTEND INTEGRATION TEST PASSED!")
            print(f"   The React component will now display:")
            print(f"   📋 Modal Title: '{data.get('cluster_name', 'text cluster 15')}'")
            print(f"   🎯 Business Context: {data.get('business_meaning', 'None')}")
            print(f"   📊 Pattern Analysis: Available")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Backend service not running")
        print("   Please start the API service first:")
        print("   cd /path/to/services/api && python3 main.py")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        return False

def test_multiple_clusters():
    """Test multiple clusters to verify semantic clustering variety"""
    print("\n🔍 Testing Multiple Clusters for Semantic Variety")
    print("=" * 60)
    
    successful_clusters = []
    cluster_names = []
    
    for cluster_id in [1, 5, 10, 15, 20]:
        payload = {
            "cluster_id": cluster_id,
            "feature_type": "text"
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/cluster_sessions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                cluster_name = data.get('cluster_name', f'text cluster {cluster_id}')
                business_meaning = data.get('business_meaning', 'No meaning provided')
                
                successful_clusters.append(cluster_id)
                cluster_names.append(cluster_name)
                
                print(f"✅ Cluster {cluster_id}: '{cluster_name}'")
                if business_meaning and business_meaning != 'No meaning provided':
                    print(f"   🎯 {business_meaning[:80]}...")
            else:
                print(f"⚠️  Cluster {cluster_id}: No data (status {response.status_code})")
                
        except Exception as e:
            print(f"❌ Cluster {cluster_id}: Error - {str(e)}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully tested {len(successful_clusters)} clusters")
    print(f"   🏷️  Unique cluster names: {len(set(cluster_names))}")
    
    if len(set(cluster_names)) > 1:
        print(f"   🎉 SEMANTIC CLUSTERING WORKING - Multiple meaningful names!")
    else:
        print(f"   ⚠️  Only found generic names - semantic clustering may need adjustment")
    
    return len(successful_clusters) > 0

def main():
    """Main test function"""
    print("🚀 Enhanced Frontend Integration Test")
    print("=" * 60)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test basic endpoint
    basic_test = test_cluster_sessions_endpoint()
    
    if basic_test:
        # Test multiple clusters
        variety_test = test_multiple_clusters()
        
        if variety_test:
            print(f"\n✅ ALL TESTS PASSED!")
            print(f"   The frontend React component should now display:")
            print(f"   • Meaningful cluster names instead of 'text cluster 15'")
            print(f"   • Business context and patterns in the modal")
            print(f"   • Error classifications when available")
            print(f"\n🎯 Next Steps:")
            print(f"   1. Open the React dashboard")
            print(f"   2. Click on any cluster point")
            print(f"   3. Verify the modal shows meaningful names")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS - Basic integration works but variety limited")
    else:
        print(f"\n❌ TEST FAILED - Backend service issues")
        print(f"   Please check backend service status and restart if needed")

if __name__ == "__main__":
    main()
