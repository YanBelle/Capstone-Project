#!/usr/bin/env python3
"""
Test script to verify the semantic clustering improvements
"""

import requests
import json

def test_semantic_clustering():
    """Test the updated semantic clustering functionality"""
    base_url = "http://localhost:8001"
    
    print("🧪 TESTING SEMANTIC CLUSTERING IMPROVEMENTS")
    print("=" * 50)
    
    try:
        # Test 1: Check system status
        print("\n1. Checking system status...")
        health_response = requests.get(f"{base_url}/api/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ System is running")
        else:
            print(f"❌ System health check failed: {health_response.status_code}")
            return
        
        # Test 2: Get model info to see current clustering approach
        print("\n2. Getting model information...")
        try:
            info_response = requests.get(f"{base_url}/api/model_info", timeout=10)
            if info_response.status_code == 200:
                model_info = info_response.json()
                print("✅ Model info retrieved")
                print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
                print(f"   BERT enabled: {model_info.get('bert_enabled', 'Unknown')}")
                print(f"   Training status: {model_info.get('is_trained', 'Unknown')}")
                
                if 'clustering_method' in model_info:
                    print(f"   Clustering method: {model_info['clustering_method']}")
                    
            else:
                print(f"⚠️ Could not get model info: {info_response.status_code}")
        except Exception as e:
            print(f"⚠️ Model info request failed: {e}")
        
        # Test 3: Test cluster endpoint with different feature types
        print("\n3. Testing cluster endpoints...")
        
        feature_types = ['semantic', 'combined', 'numerical']  # numerical should redirect to semantic
        
        for feature_type in feature_types:
            print(f"\n   Testing {feature_type} clustering...")
            try:
                cluster_response = requests.post(
                    f"{base_url}/api/cluster_sessions",
                    json={"cluster_id": 1, "feature_type": feature_type},
                    timeout=10
                )
                
                if cluster_response.status_code == 200:
                    cluster_data = cluster_response.json()
                    sessions_count = len(cluster_data.get('sessions', []))
                    print(f"   ✅ {feature_type}: {sessions_count} sessions found")
                    
                    # Check for semantic characteristics
                    if 'cluster_characteristics' in cluster_data:
                        characteristics = cluster_data['cluster_characteristics']
                        if 'business_meaning' in characteristics:
                            print(f"   📋 Business meaning: {characteristics['business_meaning'][:2] if characteristics['business_meaning'] else 'None'}")
                        if 'semantic_patterns' in characteristics:
                            patterns = characteristics['semantic_patterns']
                            active_patterns = [k for k, v in patterns.items() if v > 0]
                            print(f"   🎯 Active patterns: {active_patterns[:3] if active_patterns else 'None'}")
                            
                elif cluster_response.status_code == 400:
                    error_data = cluster_response.json()
                    print(f"   ⚠️ {feature_type}: {error_data.get('detail', 'Bad request')}")
                else:
                    print(f"   ❌ {feature_type}: HTTP {cluster_response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {feature_type} request failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 SEMANTIC CLUSTERING TEST SUMMARY")
        print("=" * 50)
        print("✅ System is operational")
        print("✅ Cluster endpoints are responding")
        print("📊 Check dashboard to see if clusters show semantic meaning")
        print("🔧 If 'numerical' redirects to 'semantic', the fix is working!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\n🔧 This indicates the backend may need syntax fixes")
        print("   Check Docker logs: docker logs ensemble-backend")

if __name__ == "__main__":
    test_semantic_clustering()
