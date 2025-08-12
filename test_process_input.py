#!/usr/bin/env python3
"""
Test script to debug the process input functionality
"""
import requests
import os
import time

def test_process_input():
    print("Testing process input functionality...")
    
    # Create test directories
    print("Creating test directories...")
    os.makedirs("abm-anomaly-ml-first/data/input", exist_ok=True)
    
    # Create sample EJ file
    sample_content = """2025-08-11 10:00:00,ATM001,DEPOSIT,100.00,SUCCESS
2025-08-11 10:05:00,ATM001,WITHDRAW,50.00,SUCCESS
2025-08-11 10:10:00,ATM001,BALANCE_INQUIRY,0.00,SUCCESS
2025-08-11 10:15:00,ATM001,DEPOSIT,200.00,SUCCESS
2025-08-11 10:20:00,ATM001,WITHDRAW,75.00,SUCCESS"""
    
    with open("abm-anomaly-ml-first/data/input/test_session.txt", "w") as f:
        f.write(sample_content)
    
    print("Created test file: abm-anomaly-ml-first/data/input/test_session.txt")
    
    # Test process input endpoint
    try:
        print("Calling process input endpoint...")
        response = requests.post("http://localhost/api/v1/process-input")
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("Process input response:")
            print(f"  Status: {result.get('status')}")
            print(f"  Message: {result.get('message')}")
            print(f"  Summary: {result.get('summary')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error calling API: {e}")
    
    # Check if file was moved
    input_file = "abm-anomaly-ml-first/data/input/test_session.txt"
    processed_file = "abm-anomaly-ml-first/data/input/processed/test_session.txt"
    
    print(f"\nFile status:")
    print(f"  Original file exists: {os.path.exists(input_file)}")
    print(f"  Processed file exists: {os.path.exists(processed_file)}")

if __name__ == "__main__":
    test_process_input()
