#!/usr/bin/env python3
"""
Debug script to check what the actual API is producing vs our local preprocessing
"""

import requests
import json
import re

def local_preprocess_text(text):
    """Our enhanced preprocessing - exactly as implemented in the API"""
    # 1. Remove date/time header patterns
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # 1c. Remove complex transaction code patterns (original)
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    # 1d. NEW: Remove any remaining complex patterns
    text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
    
    # 2. Remove remaining [020t patterns
    text = re.sub(r'\[020t\s+', '', text)
    
    # 3. Remove standalone timestamps (original)
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    # 3b. NEW: Remove standalone timestamps in format hh:mm
    text = re.sub(r'\s+\d{2}:\d{2}\s+', ' ', text)
    # 3c. NEW: Remove isolated time digits
    text = re.sub(r'\b\d{2}\b(?=\s|$)', '', text)
    
    # 4. Remove transaction markers
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    
    # 5. Enhanced pattern cleaning
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    text = re.sub(r'\bA/C\b', '', text)
    
    # Clean up "REJECTS:000*(1\nS" to just "REJECTS_000"
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
    # Additional cleanup for any remaining REJECTS fragments
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
    # Handle remaining REJECTS:000 patterns
    text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
    # Remove standalone "S" that might be left
    text = re.sub(r'\bS\b(?=\s|$)', '', text)
    
    # Convert patterns to compound tokens
    text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
    text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
    
    # NEW: Additional noise cleanup
    text = re.sub(r'\b\d\b(?=\s|$)', '', text)  # Remove standalone single digits
    text = re.sub(r'\*+', '', text)             # Remove asterisks
    text = re.sub(r'[()]+', '', text)           # Remove parentheses
    
    # Create compound tokens for ATM events
    compound_patterns = [
        (r'\bTRANSACTION START\b', 'TRANSACTION_START'),
        (r'\bTRANSACTION END\b', 'TRANSACTION_END'),
        (r'\bCARD INSERTED\b', 'CARD_INSERTED'),
        (r'\bCARD TAKEN\b', 'CARD_TAKEN'),
        (r'\bPIN ENTERED\b', 'PIN_ENTERED'),
        (r'\bDEVICE ERROR\b', 'DEVICE_ERROR'),
        (r'\bATR RECEIVED\b', 'ATR_RECEIVED'),
    ]
    
    for pattern, replacement in compound_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def test_api_vs_local():
    """Compare what the API produces vs our local preprocessing"""
    
    sample_ej = """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
 00:46:27 ATR RECEIVED T=0
[020t 00:46:30 OPCODE = FI      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
[020t 00:46:42 PIN ENTERED
[020t 00:46:47 OPCODE = IB      

  PAN 0004263********1897
  ---START OF TRANSACTION---
 
*630*06/18/2025*00:46*
*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 00:47:13 CARD TAKEN
[020t 00:47:15 TRANSACTION END
[020t*631*06/18/2025*00:47*"""

    print("=== API vs LOCAL PREPROCESSING COMPARISON ===")
    print()
    
    # Local preprocessing
    local_result = local_preprocess_text(sample_ej)
    print("LOCAL PREPROCESSING RESULT:")
    print("=" * 50)
    print(local_result)
    print("=" * 50)
    print(f"Local tokens: {local_result.split()}")
    print(f"Local token count: {len(local_result.split())}")
    print()
    
    # Test API
    print("TESTING API...")
    try:
        # Try the correct BERT endpoint
        response = requests.post(
            "http://localhost:80/api/v1/bert/analyze",
            json={"text": sample_ej},
            timeout=30
        )
        
        if response.status_code == 200:
            api_result = response.json()
            
            print("API RESPONSE SUCCESS:")
            print("=" * 50)
            
            if 'processed_text' in api_result:
                api_processed = api_result['processed_text']
                print(f"API processed text: {api_processed}")
                print(f"API tokens from processed text: {api_processed.split()}")
                print()
            
            if 'tokens' in api_result:
                api_tokens = api_result['tokens']
                print(f"API BERT tokens: {api_tokens}")
                print(f"API token count: {len(api_tokens)}")
                print()
                
                # Check for problematic tokens
                problematic = ['##1', '##w', '72', '##31', '1', '3', 's', '47', '15']
                found_problematic = [t for t in api_tokens if t in problematic]
                
                if found_problematic:
                    print(f"❌ PROBLEMATIC TOKENS FOUND IN API: {found_problematic}")
                else:
                    print("✅ NO PROBLEMATIC TOKENS FOUND IN API")
                
                # Compare with local
                if 'processed_text' in api_result:
                    if api_result['processed_text'].strip() == local_result.strip():
                        print("✅ API and LOCAL preprocessing match")
                    else:
                        print("❌ API and LOCAL preprocessing differ!")
                        print(f"LOCAL:  '{local_result}'")
                        print(f"API:    '{api_result['processed_text']}'")
                
            print("=" * 50)
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API Request failed: {e}")
        print("The API might still be loading or the endpoint might be different")

if __name__ == "__main__":
    test_api_vs_local()
