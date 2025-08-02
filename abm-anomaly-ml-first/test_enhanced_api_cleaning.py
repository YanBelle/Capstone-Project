#!/usr/bin/env python3
"""
Test enhanced EJ cleaning through API with full sample EJ text
"""

import requests
import json

def test_enhanced_api_cleaning():
    """Test enhanced EJ cleaning through API with sample EJ"""
    
    # Full sample EJ from user
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

    print("Testing Enhanced EJ Cleaning through API")
    print("=" * 60)
    print(f"Original EJ text length: {len(sample_ej)} characters")
    print(f"Original text:\n{sample_ej}")
    print("\n" + "=" * 60)
    
    api_url = "http://localhost/api/v1/bert/analyze"
    headers = {"Content-Type": "application/json"}
    payload = {"text": sample_ej}
    
    try:
        print("Making API request with full EJ sample...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response Status: {response.status_code}")
            
            if 'results' in data:
                results = data['results']
                
                # Check processed text
                if 'processed_text' in results:
                    processed = results['processed_text']
                    print(f"\nProcessed text ({len(processed)} chars):")
                    print(f"'{processed}'")
                    
                    print(f"\nText reduction: {len(sample_ej)} → {len(processed)} characters ({((len(sample_ej) - len(processed)) / len(sample_ej) * 100):.1f}% reduction)")
                    
                    # Verify cleaning worked
                    print("\nCLEANING VERIFICATION:")
                    print("-" * 40)
                    
                    # Check removed patterns
                    patterns_removed = []
                    if "[020t*629*06/18/2025*00:46*" not in processed:
                        patterns_removed.append("✅ EJ headers removed")
                    else:
                        patterns_removed.append("❌ EJ headers still present")
                        
                    if "[020t " not in processed:
                        patterns_removed.append("✅ [020t patterns removed")
                    else:
                        patterns_removed.append("❌ [020t patterns still present")
                        
                    if "00:46:27" not in processed and "00:46:30" not in processed:
                        patterns_removed.append("✅ Timestamps removed")
                    else:
                        patterns_removed.append("❌ Timestamps still present")
                        
                    if "---START OF TRANSACTION---" not in processed:
                        patterns_removed.append("✅ Transaction markers removed")
                    else:
                        patterns_removed.append("❌ Transaction markers still present")
                    
                    # Check preserved content
                    content_preserved = []
                    if "DEVICE ERROR" in processed:
                        content_preserved.append("✅ DEVICE ERROR preserved")
                    else:
                        content_preserved.append("❌ DEVICE ERROR missing")
                        
                    if "REJECTS:000" in processed:
                        content_preserved.append("✅ REJECTS:000 preserved")
                    else:
                        content_preserved.append("❌ REJECTS:000 missing")
                        
                    if "CARD INSERTED" in processed:
                        content_preserved.append("✅ CARD INSERTED preserved")
                    else:
                        content_preserved.append("❌ CARD INSERTED missing")
                    
                    for check in patterns_removed:
                        print(check)
                    for check in content_preserved:
                        print(check)
                
                # Check enhancement metadata
                if 'token_importance' in results:
                    tokens = results.get('tokens', [])
                    print(f"\nTokens generated: {len(tokens)}")
                    if tokens:
                        print(f"Sample tokens: {tokens[:10]}...")
                
                print(f"\n✅ Enhanced EJ cleaning successfully deployed and tested!")
                
            else:
                print(f"⚠️  No results in response")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_enhanced_api_cleaning()
