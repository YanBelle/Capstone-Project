#!/usr/bin/env python3
import requests

# Simple API test for EJ cleaning
test_text = "[020t 00:47:13 Transaction error DEVICE ERROR in cash dispenser [020t 15:30:22 REJECTS:000"
print("Testing EJ cleaning via API...")
print("Original text:", test_text)

try:
    response = requests.post('http://localhost:8000/api/v1/bert/visualize', 
                           json={'text': test_text}, timeout=10)
    print("Status:", response.status_code)

    if response.status_code == 200:
        result = response.json()
        processed_text = result.get('data', {}).get('processed_text', '')
        print("Processed text:", processed_text)
        
        # Check cleaning results
        if '[020t' in processed_text:
            print("❌ FAILED: [020t patterns still present")
        else:
            print("✅ SUCCESS: [020t patterns removed")
        
        # Check preservation
        if 'DEVICE ERROR' in processed_text and 'REJECTS:000' in processed_text:
            print("✅ SUCCESS: Important content preserved")
        else:
            print("⚠️  WARNING: Important content missing")
            
        # Check EJ enhancement
        ej_meta = result.get('metadata', {}).get('ej_labeler', {})
        if ej_meta.get('used'):
            print("✅ SUCCESS: EJ contextual enhancement active")
            print("Enhancement impact:", ej_meta.get('enhancement_impact', 0))
        else:
            print("⚠️  WARNING: EJ enhancement not active")
    else:
        print("Error:", response.text)
        
except Exception as e:
    print("Error:", e)
