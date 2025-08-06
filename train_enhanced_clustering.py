#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Enhanced Semantic Clustering with Specialized Error-Type Clusters
This script trains the model to show actual text patterns and specialized clusters
"""

import requests
import json
import time

def train_enhanced_semantic_model():
    """Train the model with enhanced semantic clustering capabilities"""
    
    base_url = "http://localhost:8000/api"
    
    print("Training Enhanced Semantic Clustering Model")
    print("=" * 60)
    
    # Step 1: Load EJ sessions
    print("\n1. Loading EJ sessions...")
    try:
        response = requests.post(f"{base_url}/load_ej_sessions", 
                               json={
                                   "file_path": "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/sample_ej_logs.csv",
                                   "max_sessions": 1000
                               }, 
                               timeout=60)
        
        if response.status_code == 200:
            print("✅ EJ sessions loaded successfully")
            data = response.json()
            print(f"   Sessions loaded: {data.get('sessions_loaded', 'N/A')}")
        else:
            print(f"❌ Failed to load sessions: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading sessions: {e}")
        return False
    
    # Step 2: Train with enhanced semantic clustering
    print("\n2. Training enhanced semantic clustering model...")
    try:
        response = requests.post(f"{base_url}/train", 
                               json={
                                   "use_bert": True,
                                   "feature_type": "semantic",
                                   "max_sessions": 1000,
                                   "enable_semantic_clustering": True,
                                   "enable_contextual_labeler": True
                               }, 
                               timeout=300)
        
        if response.status_code == 200:
            print("✅ Model trained successfully with enhanced semantic clustering")
            data = response.json()
            print(f"   Training time: {data.get('training_time', 'N/A')} seconds")
            print(f"   Clusters found: {data.get('clusters_found', 'N/A')}")
            return True
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_enhanced_clustering():
    """Test the enhanced clustering to show specialized clusters and text patterns"""
    
    base_url = "http://localhost:8000/api"
    
    print("\n" + "=" * 60)
    print("Testing Enhanced Semantic Clustering Results")
    print("=" * 60)
    
    # Test multiple clusters to find specialized ones
    test_clusters = [1, 5, 10, 15, 20]
    
    for cluster_id in test_clusters:
        print(f"\n🔍 Testing Cluster {cluster_id}:")
        
        try:
            response = requests.post(f"{base_url}/cluster_sessions", 
                                   json={
                                       "cluster_id": cluster_id,
                                       "feature_type": "semantic"
                                   }, 
                                   timeout=30)
            
            if response.status_code == 200:
                cluster_data = response.json()
                
                # Show meaningful cluster name
                cluster_name = cluster_data.get('cluster_name', f'Cluster {cluster_id}')
                print(f"   📛 Cluster Name: {cluster_name}")
                print(f"   📊 Size: {cluster_data.get('cluster_size', cluster_data['cluster_metadata']['cluster_size'])} sessions")
                
                # Show actual text patterns if available
                if 'actual_text_patterns' in cluster_data:
                    patterns = cluster_data['actual_text_patterns']
                    
                    print(f"   🔤 Key Terms:")
                    for term in patterns.get('key_terms', [])[:3]:
                        print(f"      • {term}")
                    
                    print(f"   🔄 Transaction Flows:")
                    flows = patterns.get('transaction_flows', {})
                    for flow_type, count in flows.items():
                        if count > 0:
                            print(f"      • {flow_type.replace('_', ' ').title()}: {count}")
                    
                    print(f"   📝 Common Sequences:")
                    for seq in patterns.get('common_sequences', [])[:2]:
                        print(f"      • {seq}")
                
                # Show business meaning
                business_meaning = cluster_data.get('business_meaning', 'N/A')
                print(f"   💼 Business Meaning: {business_meaning}")
                
                # Show error analysis if available
                if 'contextual_error_types' in cluster_data and cluster_data['contextual_error_types']:
                    error_info = cluster_data['contextual_error_types']
                    print(f"   🚨 Error Categories: {', '.join(error_info.get('primary_categories', []))}")
                    print(f"   ⚠️  Severity: {error_info.get('error_severity', 'N/A')}")
                
            else:
                print(f"   ❌ Failed to get cluster data: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n" + "=" * 60)
    print("Enhanced Clustering Summary")
    print("""
Expected Results:
✅ Meaningful cluster names instead of "text cluster 15"
✅ Actual text patterns showing clustering basis
✅ Key operational terms extraction  
✅ Transaction flow analysis
✅ Error-type classification for specialized clusters
✅ Business meaning inference from patterns

This demonstrates the enhanced semantic clustering that answers:
1. "What are the actual text that cluster 15 used to form this particular text"
2. "Can there also be clusters by the known error types using the contextual labeler"
    """)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Semantic Clustering Training and Testing")
    
    # Train the enhanced model
    success = train_enhanced_semantic_model()
    
    if success:
        # Test the results
        test_enhanced_clustering()
    else:
        print("\n❌ Training failed. Cannot proceed with testing.")
        print("Please check the backend service and data files.")
