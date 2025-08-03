#!/usr/bin/env python3
"""
Simple test to verify EJ timestamp cleaning works by checking log output
"""

import requests
import json

def simple_ej_test():
    """Test EJ cleaning with simple logging check"""
    
    print("Testing EJ timestamp cleaning...")
    print("=" * 50)
    
    test_text = "[020t 00:47:13 DEVICE ERROR detected REJECTS:000"
    print(f"Original text: '{test_text}'")
    print(f"Original length: {len(test_text)}")
    
    api_url = "http://localhost/api/v1/bert/analyze"
    headers = {"Content-Type": "application/json"}
    payload = {"text": test_text}
    
    try:
        print("\nMaking API request...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response Status: {response.status_code}")
            
            # Check if we have results
            if 'results' in data:
                results = data['results']
                print(f"Results keys: {list(results.keys())}")
                
                # Check for processed text
                if 'processed_text' in results:
                    processed = results['processed_text']
                    print(f"Processed text: '{processed}'")
                    print(f"Processed length: {len(processed)}")
                    
                    # Check if timestamp was removed
                    if "[020t" in processed:
                        print("❌ Timestamp pattern still present!")
                    else:
                        print("✅ Timestamp pattern successfully removed!")
                        
                    # Check if critical content preserved
                    if "DEVICE ERROR" in processed and "REJECTS:000" in processed:
                        print("✅ Critical content preserved!")
                    else:
                        print("❌ Critical content may be missing")
                else:
                    print("⚠️  No processed_text in results")
                    
                # Check text length
                if 'text_length' in results:
                    print(f"Text length: {results['text_length']}")
                    
                # Check tokens
                if 'tokens' in results:
                    tokens = results['tokens']
                    print(f"Tokens ({len(tokens)}): {tokens[:10]}...")  # First 10 tokens
                    
            else:
                print(f"⚠️  No results in response. Keys: {list(data.keys())}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_ej_test()
