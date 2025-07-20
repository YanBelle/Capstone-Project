#!/usr/bin/env python3

"""
Final test script to verify retraining system is working
"""

import requests
import json
import time
import subprocess

def test_api_connection():
    """Test basic API connection"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is responding")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_continuous_learning_status():
    """Test continuous learning status endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/v1/continuous-learning/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Continuous learning status working")
            print(f"   Feedback buffer size: {data.get('feedback_buffer_size', 'unknown')}")
            print(f"   Database labeled count: {data.get('feedback_database_count', 'unknown')}")
            return True, data
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
        return False, None

def test_retraining_trigger():
    """Test retraining trigger endpoint"""
    try:
        response = requests.post("http://localhost:8000/api/v1/continuous-learning/trigger-retraining", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Retraining trigger working")
            print(f"   Response: {json.dumps(data, indent=2)}")
            return True, data
        else:
            print(f"❌ Retraining trigger failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Retraining trigger error: {e}")
        return False, None

def check_database_labeled_count():
    """Check labeled anomalies count in database"""
    try:
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'postgres', 'psql', 
            '-U', 'abm_user', '-d', 'abm_db', '-t', '-c', 
            'SELECT COUNT(*) FROM labeled_anomalies;'
        ], capture_output=True, text=True, cwd='/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first')
        
        if result.returncode == 0:
            count = result.stdout.strip()
            print(f"✅ Database labeled anomalies count: {count}")
            return int(count)
        else:
            print(f"❌ Database query failed: {result.stderr}")
            return 0
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return 0

def main():
    print("=== Final Retraining System Test ===")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: API Connection
    print("\n1. Testing API connection...")
    if not test_api_connection():
        print("   API not responding, check if services are running")
        return
    
    # Test 2: Database labeled count
    print("\n2. Checking database labeled anomalies...")
    labeled_count = check_database_labeled_count()
    
    # Test 3: Continuous learning status
    print("\n3. Testing continuous learning status...")
    status_ok, status_data = test_continuous_learning_status()
    
    # Test 4: Retraining trigger
    print("\n4. Testing retraining trigger...")
    if status_ok and status_data:
        feedback_size = status_data.get('feedback_buffer_size', 0)
        if feedback_size >= 5:
            print(f"   Sufficient feedback ({feedback_size} samples) - triggering retraining...")
            retraining_ok, retraining_data = test_retraining_trigger()
            if retraining_ok:
                print("   🎉 RETRAINING SYSTEM IS WORKING!")
            else:
                print("   ❌ Retraining failed")
        else:
            print(f"   Insufficient feedback ({feedback_size} samples) - need at least 5")
            print("   You may need to add more labeled anomalies to the database")
    else:
        print("   Cannot test retraining - status endpoint failed")
    
    print("\n=== Summary ===")
    print(f"- Database labeled anomalies: {labeled_count}")
    print(f"- API status: {'✅ Working' if status_ok else '❌ Failed'}")
    print(f"- Retraining ready: {'✅ Yes' if status_ok and status_data and status_data.get('feedback_buffer_size', 0) >= 5 else '❌ No'}")

if __name__ == "__main__":
    main()
