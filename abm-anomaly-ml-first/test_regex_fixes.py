#!/usr/bin/env python3
"""
Test script to verify the new regex fixes for removing complex date patterns and transaction codes
"""

import re

def test_regex_fixes():
    """Test the new regex patterns that were added to fix the complex pattern issues"""
    
    # User's sample EJ text
    sample_text = """[020t*629*06/18/2025*00:46*
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

    print("=== TESTING NEW REGEX FIXES ===")
    print()
    
    # Test the new regex patterns individually
    print("1. Testing standalone date/time pattern removal:")
    test_pattern_1 = r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*'
    matches_1 = re.findall(test_pattern_1, sample_text)
    print(f"   Pattern: {test_pattern_1}")
    print(f"   Matches found: {matches_1}")
    print(f"   ✅ Will remove: {matches_1}")
    print()
    
    print("2. Testing complex transaction code pattern removal:")
    test_pattern_2 = r'\*\d+\*\d+\*\([^,]*,?\s*'
    matches_2 = re.findall(test_pattern_2, sample_text)
    print(f"   Pattern: {test_pattern_2}")
    print(f"   Matches found: {matches_2}")
    print(f"   ✅ Will remove: {matches_2}")
    print()
    
    # Apply the complete preprocessing pipeline (same as in bertviz_analyzer.py)
    text = sample_text
    
    print("=== BEFORE PREPROCESSING ===")
    print(f"Original text length: {len(text)} characters")
    print("Key problematic patterns present:")
    print(f"  - '*630*06/18/2025*00:46*': {'✅ Found' if '*630*06/18/2025*00:46*' in text else '❌ Not found'}")
    print(f"  - '*7231*1*(Iw(1*3,': {'✅ Found' if '*7231*1*(Iw(1*3,' in text else '❌ Not found'}")
    print()
    
    # Step 1: Remove [020t*nnn*date*time* patterns
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Step 1b: NEW FIX - Remove standalone date/time patterns
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Step 1c: NEW FIX - Remove complex transaction code patterns
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    
    # Continue with rest of preprocessing pipeline
    text = re.sub(r'\[020t\s+', '', text)
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    
    # Apply compound patterns (abbreviated for test)
    text = re.sub(r'\bDEVICE\s+ERROR\b', 'DEVICE_ERROR', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCARD\s+INSERTED\b', 'CARD_INSERTED', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCARD\s+TAKEN\b', 'CARD_TAKEN', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPIN\s+ENTERED\b', 'PIN_ENTERED', text, flags=re.IGNORECASE)
    text = re.sub(r'\bATR\s+RECEIVED\b', 'ATR_RECEIVED', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTRANSACTION\s+END\b', 'TRANSACTION_END', text, flags=re.IGNORECASE)
    
    # Clean punctuation
    text = re.sub(r'\*+', '_', text)
    text = re.sub(r'[(),]+', ' ', text)
    text = re.sub(r':(\s*\d{3})\b', r' \1', text)
    text = re.sub(r'\b(REF|ESC|VAL):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\s*[=:]\s*', ' ', text)
    text = ' '.join(text.split())
    
    print("=== AFTER PREPROCESSING ===")
    print(f"Processed text length: {len(text)} characters")
    print(f"Reduction: {len(sample_text) - len(text)} characters ({((len(sample_text) - len(text)) / len(sample_text) * 100):.1f}%)")
    print()
    print("Key problematic patterns check:")
    print(f"  - '*630*06/18/2025*00:46*': {'❌ Still present' if '*630*06/18/2025*00:46*' in text else '✅ REMOVED'}")
    print(f"  - '*7231*1*(Iw(1*3,': {'❌ Still present' if '*7231*1*(Iw(1*3,' in text else '✅ REMOVED'}")
    print(f"  - 'Iw': {'❌ Still present' if 'Iw' in text else '✅ REMOVED'}")
    print()
    
    print("=== FINAL PROCESSED TEXT ===")
    print(f"'{text}'")
    print()
    
    print("=== SUCCESS VERIFICATION ===")
    if '*630*06/18/2025*00:46*' not in text and '*7231*1*(Iw(1*3,' not in text and 'Iw' not in text:
        print("✅ SUCCESS: All problematic patterns have been removed!")
        print("✅ BERT will no longer encounter these noise patterns")
        print("✅ This should eliminate the source of '##11' and '##w' tokens")
    else:
        print("❌ ISSUE: Some problematic patterns remain in the text")
        remaining = []
        if '*630*06/18/2025*00:46*' in text:
            remaining.append('*630*06/18/2025*00:46*')
        if '*7231*1*(Iw(1*3,' in text:
            remaining.append('*7231*1*(Iw(1*3,')
        if 'Iw' in text:
            remaining.append('Iw')
        print(f"   Remaining patterns: {remaining}")
    
    print()
    print("=== IMPACT ON BERT TOKENIZATION ===")
    print("Before fix: BERT would tokenize 'R-10011' from '*630*' and 'Iw' from '*7231*1*(Iw(1*3,'")
    print("After fix: Only 'R-10011' remains (from the meaningful transaction data)")
    print("Result: Major reduction in noise tokens like ##11, ##w that were confusing the attention heatmap")

if __name__ == "__main__":
    test_regex_fixes()
