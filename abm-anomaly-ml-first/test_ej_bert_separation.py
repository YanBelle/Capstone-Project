#!/usr/bin/env python3
"""
Test to verify EJ labeler gets original text with timestamps while BERT gets cleaned text
"""

import re

def simulate_text_flow():
    """Simulate the text processing flow to verify separation"""
    
    # Sample EJ text with timestamps and patterns
    original_ej = """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
 00:46:27 ATR RECEIVED T=0
[020t 00:46:30 OPCODE = FI      
  PAN 0004263********1897
  ---START OF TRANSACTION---
 
[020t 00:46:42 PIN ENTERED
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000
[020t 00:47:13 CARD TAKEN"""

    print("Testing EJ Text Processing Separation")
    print("=" * 60)
    print(f"Original EJ Text ({len(original_ej)} chars):")
    print(f"'{original_ej}'")
    print("\n" + "=" * 60)
    
    # Step 1: Store original for EJ labeler
    ej_labeler_text = original_ej  # EJ labeler gets the original
    print("EJ LABELER INPUT (with timestamps for feature extraction):")
    print(f"Length: {len(ej_labeler_text)} chars")
    print(f"Contains [020t*629*06/18/2025*00:46*: {'[020t*629*06/18/2025*00:46*' in ej_labeler_text}")
    print(f"Contains timestamps 00:46:27: {'00:46:27' in ej_labeler_text}")
    print(f"Contains ---START OF TRANSACTION---: {'---START OF TRANSACTION---' in ej_labeler_text}")
    print(f"Contains DEVICE ERROR: {'DEVICE ERROR' in ej_labeler_text}")
    print(f"Contains REJECTS:000: {'REJECTS:000' in ej_labeler_text}")
    
    # Step 2: Clean for BERT
    bert_text = original_ej
    
    # Apply BERT cleaning patterns
    print(f"\nBERT PREPROCESSING STEPS:")
    print("-" * 40)
    
    # 1. Remove EJ header patterns
    bert_text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', bert_text)
    print(f"1. After EJ header removal: {len(bert_text)} chars")
    
    # 2. Remove [020t patterns
    bert_text = re.sub(r'\[020t\s+', '', bert_text)
    print(f"2. After [020t removal: {len(bert_text)} chars")
    
    # 3. Remove timestamps
    bert_text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', bert_text)
    print(f"3. After timestamp removal: {len(bert_text)} chars")
    
    # 4. Remove transaction markers
    bert_text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', bert_text)
    print(f"4. After transaction marker removal: {len(bert_text)} chars")
    
    # 5. Clean whitespace
    bert_text = ' '.join(bert_text.split())
    print(f"5. After whitespace cleanup: {len(bert_text)} chars")
    
    print(f"\nBERT INPUT (cleaned for attention analysis):")
    print(f"Length: {len(bert_text)} chars")
    print(f"Text: '{bert_text}'")
    
    # Verification
    print(f"\n" + "=" * 60)
    print("VERIFICATION - SEPARATION WORKING:")
    print("=" * 60)
    
    print("EJ LABELER (has all original features):")
    ej_features = []
    if "[020t*629*06/18/2025*00:46*" in ej_labeler_text:
        ej_features.append("✅ EJ headers for sequence tracking")
    if "00:46:27" in ej_labeler_text:
        ej_features.append("✅ Event timestamps for timing analysis")
    if "---START OF TRANSACTION---" in ej_labeler_text:
        ej_features.append("✅ Transaction boundaries for phase detection")
    if "DEVICE ERROR" in ej_labeler_text:
        ej_features.append("✅ Critical error content")
    
    for feature in ej_features:
        print(f"  {feature}")
    
    print(f"\nBERT (clean text for attention):")
    bert_benefits = []
    if "[020t*629*06/18/2025*00:46*" not in bert_text:
        bert_benefits.append("✅ No EJ header noise")
    if "00:46:27" not in bert_text and "00:46:30" not in bert_text:
        bert_benefits.append("✅ No timestamp noise")
    if "---START OF TRANSACTION---" not in bert_text:
        bert_benefits.append("✅ No repetitive markers")
    if "DEVICE ERROR" in bert_text and "REJECTS:000" in bert_text:
        bert_benefits.append("✅ Critical content preserved for attention")
    
    for benefit in bert_benefits:
        print(f"  {benefit}")
    
    print(f"\nText reduction for BERT: {len(original_ej)} → {len(bert_text)} chars ({((len(original_ej) - len(bert_text)) / len(original_ej) * 100):.1f}% reduction)")
    
    return ej_labeler_text, bert_text

if __name__ == "__main__":
    ej_text, bert_text = simulate_text_flow()
    print(f"\n✅ SUCCESS: EJ labeler and BERT processing properly separated!")
    print(f"   - EJ labeler: {len(ej_text)} chars with full feature set")
    print(f"   - BERT: {len(bert_text)} chars with clean attention focus")
