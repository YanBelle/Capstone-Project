#!/usr/bin/env python3
"""
Test script to verify EJ timestamp cleaning functionality through the API
"""

import requests
import json
import time

def test_ej_cleaning():
    """Test EJ timestamp cleaning through API endpoint"""
    
    # Test cases with EJ timestamp patterns
    test_cases = [
        {
            "name": "Simple timestamp",
            "text": "[020t 00:47:13 DEVICE ERROR detected",
            "expected_cleaned": "DEVICE ERROR detected"
        },
        {
            "name": "Multiple timestamps", 
            "text": "[020t 00:47:13 ERROR [020t 00:48:15 REJECTS:000",
            "expected_cleaned": "ERROR REJECTS:000"
        },
        {
            "name": "No timestamp",
            "text": "DEVICE ERROR detected REJECTS:000",
            "expected_cleaned": "DEVICE ERROR detected REJECTS:000"
        }
    ]
    
    api_url = "http://localhost/api/v1/bert/analyze"
    headers = {"Content-Type": "application/json"}
    
    print("Testing EJ timestamp cleaning through API...")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Original text: '{test_case['text']}'")
        print(f"Original length: {len(test_case['text'])}")
        
        payload = {"text": test_case['text']}
        
        try:
            print("Making API request...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                processed_text = data.get('processed_text', 'N/A')
                text_length = data.get('text_length', 'N/A')
                ej_enhancement = data.get('ej_enhancement_impact', 'N/A')
                
                print(f"✅ API Response Status: {response.status_code}")
                print(f"Processed text: '{processed_text}'")
                print(f"Processed length: {text_length}")
                print(f"EJ Enhancement Impact: {ej_enhancement}")
                
                # Check if cleaning worked
                if "[020t" in processed_text:
                    print("❌ Timestamp patterns still present!")
                else:
                    print("✅ Timestamp patterns successfully removed")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("-" * 40)
        time.sleep(1)  # Brief pause between requests

if __name__ == "__main__":
    test_ej_cleaning()
