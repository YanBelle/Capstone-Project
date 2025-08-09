#!/usr/bin/env python3
"""
Train the model with sample ATM data to demonstrate semantic clustering
"""

import requests
import json

def train_semantic_model():
    """Train the model with sample ATM transaction data"""
    base_url = "http://localhost:8001"
    
    print("TRAINING SEMANTIC CLUSTERING MODEL")
    print("=" * 50)
    
    # Sample ATM sessions with different semantic categories
    sample_data = [
        # Authentication Issues (Cluster 1)
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED PIN_VERIFICATION_FAILED CARD_CAPTURE security response authentication failure",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED PIN_ENTERED PIN_VERIFICATION_FAILED timeout authentication error",
        "TRANSACTION_START CARD_INSERTED PIN_VERIFICATION_FAILED E-45 authentication failure card retained",
        
        # Hardware Failures (Cluster 2)  
        "TRANSACTION_START CARD_INSERTED DEVICE_ERROR M-65 SUPERVISOR_MODE maintenance required device initialization failure",
        "DEVICE_ERROR M-01 critical system error hardware malfunction immediate service required",
        "TRANSACTION_START CARD_INSERTED DEVICE_ERROR M-15 dispenser mechanism fault cash jam service needed",
        
        # Successful Transactions (Cluster 3)
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED AMOUNT_SELECTED CASH_DISPENSED 100 RECEIPT_PRINTED CARD_EJECTED successful",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED AMOUNT_SELECTED CASH_DISPENSED 200 RECEIPT_PRINTED CARD_EJECTED completed successfully",
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED CASH_DISPENSED 50 RECEIPT_PRINTED CARD_EJECTED transaction completed",
        
        # Communication Issues (Cluster 4)
        "TRANSACTION_START CARD_INSERTED PIN_ENTERED COMMUNICATION_FAILURE M-23 timeout network error server unreachable",
        "COMMUNICATION_FAILURE network timeout unable to process authentication server connectivity issues",
        "TRANSACTION_START CARD_INSERTED COMMUNICATION_FAILURE timeout network error unable to complete"
    ]
    
    try:
        print(f"Training with {len(sample_data)} diverse ATM sessions...")
        print("   * Authentication failures")
        print("   * Hardware malfunctions") 
        print("   * Successful transactions")
        print("   * Communication issues")
        
        train_response = requests.post(
            f"{base_url}/api/train",
            json={"sessions": sample_data},
            timeout=60
        )
        
        if train_response.status_code == 200:
            train_result = train_response.json()
            print("\nSEMANTIC CLUSTERING TRAINING SUCCESSFUL!")
            
            # Display training results
            if 'training_stats' in train_result:
                stats = train_result['training_stats']
                print(f"   Sessions processed: {stats.get('num_training_sessions', 'Unknown')}")
                print(f"   BERT embeddings: {stats.get('text_feature_dims', 'Unknown')} dimensions")
                print(f"   Clustering method: {stats.get('cluster_analysis', {}).get('clustering_method', 'Unknown')}")
            
            return True
            
        else:
            print(f"Training failed: HTTP {train_response.status_code}")
            print(train_response.text)
            return False
            
    except Exception as e:
        print(f"Training error: {e}")
        return False

def test_semantic_clusters():
    """Test the semantic clustering results"""
    base_url = "http://localhost:8001"
    
    print("\nTESTING SEMANTIC CLUSTER RESULTS")
    print("=" * 50)
    
    # Test different clusters to see semantic groupings
    for cluster_id in range(0, 4):  # Test first 4 clusters
        print(f"\nCLUSTER {cluster_id} (Semantic Analysis):")
        print("-" * 30)
        
        try:
            cluster_response = requests.post(
                f"{base_url}/api/cluster_sessions",
                json={"cluster_id": cluster_id, "feature_type": "semantic"},
                timeout=10
            )
            
            if cluster_response.status_code == 200:
                cluster_data = cluster_response.json()
                sessions = cluster_data.get('sessions', [])
                characteristics = cluster_data.get('cluster_characteristics', {})
                
                print(f"Sessions: {len(sessions)}")
                
                # Show semantic characteristics
                if 'business_meaning' in characteristics:
                    meanings = characteristics['business_meaning']
                    print(f"Business Meaning:")
                    for meaning in meanings[:2]:  # Show first 2
                        print(f"   * {meaning}")
                
                if 'semantic_patterns' in characteristics:
                    patterns = characteristics['semantic_patterns']
                    active_patterns = [(k, v) for k, v in patterns.items() if v > 0]
                    if active_patterns:
                        print(f"Semantic Patterns:")
                        for pattern, count in active_patterns[:3]:  # Show top 3
                            pattern_name = pattern.replace('_', ' ').title()
                            print(f"   * {pattern_name}: {count}")
                
                # Show sample session
                if sessions:
                    sample_session = sessions[0].get('session_text', '')[:100]
                    print(f"Sample: {sample_session}...")
                    
            elif cluster_response.status_code == 400:
                error_data = cluster_response.json()
                if "not found" in error_data.get('detail', '').lower():
                    print(f"No cluster {cluster_id} (expected - clusters are 0-indexed)")
                    break
                else:
                    print(f"Error: {error_data.get('detail', 'Unknown error')}")
            else:
                print(f"HTTP {cluster_response.status_code}")
                
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("SEMANTIC CLUSTERING DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("* Model trained with semantic understanding")
    print("* Clusters grouped by business meaning (not word counts)")
    print("* ATM domain codes converted to semantic descriptions")
    print("* BERT embeddings capture contextual relationships")
    print("\nKey Improvement: Instead of counting 'error' words,")
    print("   clusters now understand TYPES of problems!")

if __name__ == "__main__":
    # Train the model
    if train_semantic_model():
        # Test the results
        test_semantic_clusters()
    else:
        print("Training failed - cannot test clusters")
