#!/usr/bin/env python3
"""
Test script to trigger supervised training and monitor progress
"""

import requests
import time
import json
import sys

def test_training():
    base_url = "http://localhost"
    
    print("🔍 Testing supervised training...")
    
    # First check training data availability
    try:
        print("\n1. Checking training data availability...")
        response = requests.get(f"{base_url}/api/v1/expert/training-data-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training data info:")
            print(f"   - Labeled anomalies: {data.get('labeled_anomalies', {})}")
            print(f"   - ML sessions: {data.get('ml_sessions', {})}")
            print(f"   - Training ready: {data.get('training_ready', {})}")
            print(f"   - Can train: {data.get('training_possible', False)}")
        else:
            print(f"❌ Failed to get training data info: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error checking training data: {e}")
    
    # Trigger training
    try:
        print("\n2. Triggering supervised training...")
        response = requests.post(f"{base_url}/api/v1/expert/train-supervised", 
                               headers={"Content-Type": "application/json"},
                               timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training triggered successfully:")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            print(f"   Labeled samples: {data.get('labeled_samples')}")
            print(f"   Unique labels: {data.get('unique_labels')}")
            if 'labels_distribution' in data:
                print(f"   Label distribution: {data['labels_distribution']}")
        else:
            print(f"❌ Failed to trigger training: {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error triggering training: {e}")
        return
    
    # Monitor training progress
    print("\n3. Monitoring training progress...")
    for i in range(30):  # Monitor for up to 30 seconds
        try:
            response = requests.get(f"{base_url}/api/v1/expert/training-status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"   [{i+1:2d}] Status: {status.get('status', 'unknown')} - {status.get('message', 'No message')} ({status.get('progress', 0)}%)")
                
                if status.get('status') == 'completed':
                    print("🎉 Training completed successfully!")
                    break
                elif status.get('status') == 'error':
                    print(f"❌ Training failed: {status.get('message')}")
                    break
            else:
                print(f"   [{i+1:2d}] Failed to get status: {response.status_code}")
        except Exception as e:
            print(f"   [{i+1:2d}] Error getting status: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_training()
