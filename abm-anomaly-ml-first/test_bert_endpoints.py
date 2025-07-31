#!/usr/bin/env python3
import requests
import json

def test_endpoint(url, method='GET', data=None):
    try:
        if method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
            
        print(f"{method} {url}: {response.status_code}")
        if response.status_code == 200:
            print("  ✅ Success!")
        else:
            print(f"  ❌ Error: {response.text[:200]}")
        return response.status_code == 200
    except Exception as e:
        print(f"{method} {url}: ❌ Exception: {e}")
        return False

print("Testing BERT API endpoints...")
print("=" * 50)

# Test analyze endpoint
success = test_endpoint(
    "http://localhost/api/v1/bert/analyze", 
    "POST", 
    {"text": "test ATM transaction", "analysis_type": "full"}
)

# Test visualize endpoint  
test_endpoint(
    "http://localhost/api/v1/bert/visualize",
    "POST",
    {"text": "test ATM transaction"}
)

# Test patterns endpoint
test_endpoint("http://localhost/api/v1/bert/patterns")

# Test optimize endpoint
test_endpoint(
    "http://localhost/api/v1/bert/optimize",
    "POST", 
    {"text": "test ATM transaction"}
)

print("=" * 50)
if success:
    print("✅ BERT endpoints are working! The 503 errors should be resolved.")
else:
    print("❌ There are still issues with BERT endpoints.")
