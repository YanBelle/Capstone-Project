#!/usr/bin/env python3
import json
import requests

# Test BERT API with simple text
url = "http://localhost:8000/api/v1/bert/visualize"

# Test with clear anomaly text
test_cases = [
    "DEVICE ERROR 001",
    "REJECTS:000",
    "DEVICE ERROR REJECTS:000",
    "NORMAL OPERATION"
]

for test_text in test_cases:
    print(f"\n=== Testing: {test_text} ===")
    
    payload = {
        "text": test_text,
        "return_tokens": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'token_importance' in result:
                print("Token importance scores:")
                sorted_tokens = sorted(result['token_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for token, score in sorted_tokens:
                    print(f"  {token}: {score:.4f}")
            else:
                print("No token importance in response")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
