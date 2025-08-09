#!/usr/bin/env python3
"""
Test the API data loading directly
"""

import requests
import json

def test_api_data_loading():
    print("Testing API Data Loading")
    print("=" * 40)
    
    # Test the load_ej_sessions endpoint
    try:
        # Start by testing if the API is running
        response = requests.post(
            "http://localhost:8001/api/load_ej_sessions",
            json={"include_errors": False},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success', False)}")
            print(f"Message: {data.get('message', 'No message')}")
            print(f"Count: {data.get('count', 0)}")
            print(f"Data Source: {data.get('data_source', 'Unknown')}")
            
            return True
        else:
            print(f"API Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API at localhost:8001")
        print("Please start the backend first:")
        print("cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"Error testing API: {e}")
        return False

if __name__ == "__main__":
    test_api_data_loading()
