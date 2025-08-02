#!/usr/bin/env python3
import requests
import json
import sys

# Simple test of the BERT API
test_ej_log = "DEVICE ERROR 001 REJECTS:000 FATAL ERROR IN LOG PROCESSING"

try:
    # Test API call to BERT visualization endpoint
    url = "http://localhost:8000/api/v1/bert/visualize"
    
    payload = {
        "text": test_ej_log,
        "return_tokens": True
    }
    
    print(f"Testing BERT API with text: {test_ej_log}")
    print(f"Calling: {url}")
    
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nAPI Response successful!")
        
        # Check if contextual enhancement is working
        if 'token_importance' in result:
            print("\nToken importance scores:")
            for token, score in result['token_importance'].items():
                print(f"  {token}: {score:.4f}")
            
            # Check if important terms have higher scores
            device_score = result['token_importance'].get('DEVICE', 0)
            error_score = result['token_importance'].get('ERROR', 0)
            rejects_score = result['token_importance'].get('REJECTS', 0)
            
            print(f"\nKey anomaly term scores:")
            print(f"  DEVICE: {device_score:.4f}")
            print(f"  ERROR: {error_score:.4f}")
            print(f"  REJECTS: {rejects_score:.4f}")
            
            # Check for contextual enhancement metadata
            if 'metadata' in result:
                print(f"\nMetadata: {result['metadata']}")
        
        if 'attention_data' in result:
            print("\nAttention data available")
        
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error testing API: {e}")
    sys.exit(1)
