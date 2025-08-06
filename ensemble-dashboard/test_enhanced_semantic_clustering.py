#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Semantic Clustering with Text Pattern Extraction
Shows actual text patterns that BERT uses for clustering and meaningful cluster names
"""

import requests
import json
import time

def test_enhanced_clustering():
    """Test the enhanced semantic clustering with pattern visualization"""
    
    base_url = "http://localhost:8000"
    
    print("Testing Enhanced Semantic Clustering with Text Pattern Extraction")
    print("=" * 80)
    
    # Step 1: Get cluster sessions with enhanced details
    print("\n1. Retrieving Cluster 15 with enhanced pattern analysis...")
    
    try:
        response = requests.post(base_url + "/cluster_sessions", 
                               json={"cluster_id": 15}, 
                               timeout=30)
        
        if response.status_code == 200:
            cluster_data = response.json()
            
            print("\nCluster 15 Analysis:")
            print("   Size: {} sessions".format(cluster_data.get('cluster_size', 'N/A')))
            print("   Business Meaning: {}".format(cluster_data.get('business_meaning', 'N/A')))
            
            # Check for enhanced pattern data
            if 'actual_text_patterns' in cluster_data:
                patterns = cluster_data['actual_text_patterns']
                
                print(f"\nActual Text Patterns BERT Used for Clustering:")
                print(f"   Common Text Sequences:")
                for seq in patterns.get('common_sequences', [])[:3]:
                    print(f"     • {seq}")
                
                print(f"\n   Key Operational Terms:")
                for term in patterns.get('key_terms', [])[:5]:
                    print(f"     • {term}")
                
                print(f"\n   Transaction Flow Patterns:")
                flows = patterns.get('transaction_flows', {})
                for flow_type, count in flows.items():
                    if count > 0:
                        print(f"     • {flow_type.replace('_', ' ').title()}: {count}")
            
            # Show meaningful cluster name
            if 'cluster_name' in cluster_data:
                print(f"\nMeaningful Cluster Name: '{cluster_data['cluster_name']}'")
            
            # Show contextual error analysis if available
            if 'contextual_error_types' in cluster_data and cluster_data['contextual_error_types']:
                error_types = cluster_data['contextual_error_types']
                print(f"\nError Type Analysis:")
                print(f"   Primary Categories: {', '.join(error_types.get('primary_categories', []))}")
                print(f"   Error Severity: {error_types.get('error_severity', 'N/A')}")
                print(f"   Contextual Labels: {', '.join(error_types.get('contextual_labels', []))}")
            
            # Show sample sessions
            if 'sessions' in cluster_data:
                print(f"\nSample Session Text:")
                for i, session in enumerate(cluster_data['sessions'][:2]):
                    print(f"   Session {i+1}: {session[:200]}...")
                    
        else:
            print(f"Error retrieving cluster data: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Exception during cluster retrieval: {e}")
    
    # Step 2: Test another cluster for comparison
    print(f"\n" + "=" * 80)
    print("2. Testing another cluster for pattern comparison...")
    
    try:
        response = requests.post(f"{base_url}/cluster_sessions", 
                               json={"cluster_id": 1}, 
                               timeout=30)
        
        if response.status_code == 200:
            cluster_data = response.json()
            
            print(f"\nCluster 1 Analysis:")
            print(f"   Size: {cluster_data.get('cluster_size', 'N/A')} sessions")
            
            if 'cluster_name' in cluster_data:
                print(f"   Meaningful Name: '{cluster_data['cluster_name']}'")
            
            if 'actual_text_patterns' in cluster_data:
                patterns = cluster_data['actual_text_patterns']
                print(f"   Key Terms: {', '.join(patterns.get('key_terms', [])[:3])}")
                
                flows = patterns.get('transaction_flows', {})
                active_flows = [f"{flow}: {count}" for flow, count in flows.items() if count > 0]
                if active_flows:
                    print(f"   Active Flows: {', '.join(active_flows[:2])}")
                    
    except Exception as e:
        print(f"Exception during second cluster test: {e}")
    
    # Step 3: Demonstrate clustering reasoning
    print(f"\n" + "=" * 80)
    print("3. Summary: Enhanced Semantic Clustering Benefits")
    print("""
BERT Semantic Clustering now provides:
   • Actual text sequences that drive clustering decisions
   • Meaningful business cluster names instead of "text cluster 15"
   • Transaction flow pattern analysis
   • Contextual error type classification
   • Key operational term extraction
   • Business meaning inference
   
This answers your question: "what are the actual text that cluster 15 used to form this particular text"
   
Next step: Integrate contextual labeler for error-type specific clustering
    """)

if __name__ == "__main__":
    test_enhanced_clustering()
