import requests
import time
import json

# Test the backend API
def test_backend():
    base_url = "http://localhost:8000"
    
    print("Testing backend health...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        else:
            print(f"Health response: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\nTesting model info...")
    try:
        response = requests.get(f"{base_url}/api/model_info", timeout=10)
        print(f"Model info status: {response.status_code}")
        if response.status_code == 200:
            print(f"Model info: {response.json()}")
        else:
            print(f"Model info response: {response.text}")
    except Exception as e:
        print(f"Model info failed: {e}")
    
    print("\nTesting analyze endpoint...")
    try:
        data = {"text": "Test EJ log message"}
        response = requests.post(f"{base_url}/api/analyze", data=data, timeout=30)
        print(f"Analyze status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Analyze response keys: {list(result.keys())}")
            print(f"Predicted class: {result.get('predicted_class')}")
            print(f"Probabilities: {result.get('probabilities')}")
        else:
            print(f"Analyze response: {response.text}")
    except Exception as e:
        print(f"Analyze failed: {e}")

if __name__ == "__main__":
    test_backend()
