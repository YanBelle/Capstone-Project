#!/usr/bin/env python3
import requests
import time

# Quick test
text = "[020t 00:47:13 DEVICE ERROR"
print("Original:", text, "Length:", len(text))

start = time.time()
response = requests.post('http://localhost:8000/api/v1/bert/visualize', 
                       json={'text': text}, timeout=45)
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()
    processed = data.get('data', {}).get('processed_text', '')
    print("Processed:", processed, "Length:", len(processed))
    
    # Quick checks
    if '[020t' not in processed:
        print("✅ [020t removed")
    if 'DEVICE ERROR' in processed:
        print("✅ DEVICE ERROR preserved")
        
    print("EJ Enhancement:", data.get('metadata', {}).get('ej_labeler', {}).get('used', False))
else:
    print("Error:", response.status_code)
