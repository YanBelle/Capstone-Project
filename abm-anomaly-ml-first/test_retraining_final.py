#!/usr/bin/env python3

"""
Simple test to verify the retraining system is working with labeled anomalies
"""

import requests
import json
import time

def test_retraining_with_labeled_data():
    """Test the retraining system with the user's labeled data"""
    
    base_url = "http://localhost:8000/api/v1"
    
    print("Testing retraining with labeled anomalies...")
    
    # 1. Check continuous learning status
    print("\n1. Checking continuous learning status...")
    try:
        response = requests.get(f"{base_url}/continuous-learning/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Status: {json.dumps(status, indent=2)}")
            
            # Check if we have labeled data
            feedback_size = status.get('feedback_buffer_size', 0)
            db_count = status.get('feedback_database_count', 0)
            
            print(f"   Feedback buffer size: {feedback_size}")
            print(f"   Database labeled count: {db_count}")
            
            if feedback_size >= 5:
                print("   ✅ Sufficient labeled data for retraining!")
            else:
                print("   ❌ Insufficient labeled data for retraining")
                
        else:
            print(f"   Status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"   Error checking status: {e}")
    
    # 2. Trigger retraining
    print("\n2. Triggering retraining...")
    try:
        response = requests.post(f"{base_url}/continuous-learning/trigger-retraining")
        if response.status_code == 200:
            result = response.json()
            print(f"   Retraining response: {json.dumps(result, indent=2)}")
            
            if result.get('status') == 'success':
                print("   ✅ Retraining triggered successfully!")
            else:
                print("   ❌ Retraining failed")
                
        else:
            print(f"   Retraining failed: {response.status_code}")
            
    except Exception as e:
        print(f"   Error triggering retraining: {e}")
    
    # 3. Wait and check status again
    print("\n3. Waiting 10 seconds and checking status again...")
    time.sleep(10)
    
    try:
        response = requests.get(f"{base_url}/continuous-learning/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Updated status: {json.dumps(status, indent=2)}")
        else:
            print(f"   Status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"   Error checking final status: {e}")

if __name__ == "__main__":
    test_retraining_with_labeled_data()
