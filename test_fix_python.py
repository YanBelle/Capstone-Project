import requests
import json
import os
import time

def test_process_input_fix():
    print("=== Testing Process Input Fix ===")
    
    # Create test file
    base_path = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first"
    os.makedirs(f"{base_path}/data/input", exist_ok=True)
    
    test_content = """2025-08-11 10:00:00,ATM001,DEPOSIT,100.00,SUCCESS
2025-08-11 10:05:00,ATM001,WITHDRAW,50.00,SUCCESS
2025-08-11 10:10:00,ATM001,BALANCE_INQUIRY,0.00,SUCCESS
2025-08-11 10:15:00,ATM001,DEPOSIT,200.00,SUCCESS
2025-08-11 10:20:00,ATM001,WITHDRAW,75.00,SUCCESS"""
    
    with open(f"{base_path}/data/input/test_session.txt", "w") as f:
        f.write(test_content)
    
    print("Created test file with EJ content")
    
    # Test debug endpoint
    try:
        print("\nTesting debug file status...")
        response = requests.get("http://localhost/api/v1/debug/file-status", timeout=10)
        if response.status_code == 200:
            debug_info = response.json()
            print(f"Debug info: {json.dumps(debug_info, indent=2)}")
        else:
            print(f"Debug endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"Debug endpoint error: {e}")
    
    # Test process input
    try:
        print("\nTesting process input...")
        response = requests.post("http://localhost/api/v1/process-input", timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"Process result: {json.dumps(result, indent=2)}")
        else:
            print(f"Process input failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Process input error: {e}")
    
    # Check final status
    try:
        print("\nChecking final file status...")
        response = requests.get("http://localhost/api/v1/debug/file-status", timeout=10)
        if response.status_code == 200:
            debug_info = response.json()
            print(f"Final debug info: {json.dumps(debug_info, indent=2)}")
        else:
            print(f"Final debug check failed: {response.status_code}")
    except Exception as e:
        print(f"Final debug error: {e}")

if __name__ == "__main__":
    test_process_input_fix()
