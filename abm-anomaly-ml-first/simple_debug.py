#!/usr/bin/env python3
"""
Simple debug script to answer the user's questions about text preprocessing
"""

import re

def main():
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

    print("=== QUESTION 1: Which section removes '*630*06/18/2025*00:46*'? ===")
    print()
    
    # Test the current regex pattern
    pattern = r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*'
    print(f"Current regex pattern: {pattern}")
    print()
    
    # Find what it matches
    matches = re.findall(pattern, sample_text)
    print("What the current pattern DOES match:")
    for match in matches:
        print(f"  - '{match}'")
    print()
    
    # Check specific patterns
    test_patterns = [
        '[020t*629*06/18/2025*00:46*',
        '*630*06/18/2025*00:46*',
        '[020t*631*06/18/2025*00:47*'
    ]
    
    print("Testing specific patterns:")
    for test_pattern in test_patterns:
        if re.match(pattern, test_pattern):
            print(f"  ✅ MATCHES: '{test_pattern}'")
        else:
            print(f"  ❌ NO MATCH: '{test_pattern}'")
    print()
    
    print("=== ANSWER TO QUESTION 1 ===")
    print("The pattern '*630*06/18/2025*00:46*' is NOT removed by the current preprocessing!")
    print("Reason: The regex pattern requires '[020t' at the start, but '*630*' starts with '*'")
    print()
    
    print("=== QUESTION 2: Where do '##11' and '##w' tokens come from? ===")
    print()
    
    # Apply the current preprocessing step by step
    text = sample_text
    
    # Step 1: Remove [020t*nnn*date*time* patterns
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Step 1b: NEW FIX - Remove standalone date/time patterns that don't start with [020t
    # Pattern: *630*06/18/2025*00:46* (removes patterns like "*630*06/18/2025*00:46*")
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Step 1c: NEW FIX - Remove complex transaction code patterns
    # Pattern: *7231*1*(Iw(1*3, (removes patterns like "*7231*1*(Iw(1*3,")
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    
    # Step 2: Remove remaining [020t patterns  
    text = re.sub(r'\[020t\s+', '', text)
    
    # Step 3: Remove timestamps
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    
    # Step 4: Remove transaction markers
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    
    # Step 5: Replace *TRANSACTION START*
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    
    # Step 6: Replace PAN patterns
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    
    # Step 7: Remove complex transaction codes
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    
    # Apply compound patterns (simplified)
    text = re.sub(r'\bDEVICE\s+ERROR\b', 'DEVICE_ERROR', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCARD\s+TAKEN\b', 'CARD_TAKEN', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTRANSACTION\s+END\b', 'TRANSACTION_END', text, flags=re.IGNORECASE)
    
    # Clean punctuation
    text = re.sub(r'\*+', '_', text)
    text = re.sub(r'[(),]+', ' ', text)
    text = re.sub(r':(\s*\d{3})\b', r' \1', text)
    text = re.sub(r'\b(REF|ESC|VAL):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\s*[=:]\s*', ' ', text)
    text = ' '.join(text.split())
    
    print("Processed text that goes to BERT:")
    print(f"'{text}'")
    print()
    
    print("=== ANSWER TO QUESTION 2 ===")
    print("FIXED! The problematic patterns are now removed:")
    print("1. '*630*06/18/2025*00:46*' → Now REMOVED by new regex pattern")
    print("2. '*7231*1*(Iw(1*3,' → Now REMOVED by new regex pattern")
    print("3. This will eliminate the source of '##11' and '##w' tokens")
    print()
    print("New regex patterns added:")
    print("- r'\\*\\d+\\*\\d{2}/\\d{2}/\\d{4}\\*\\d{2}:\\d{2}\\*' (standalone date patterns)")
    print("- r'\\*\\d+\\*\\d+\\*\\([^,]*,?\\s*' (complex transaction codes)")
    print()
    print("Result: BERT will now receive much cleaner text without these noise patterns!")
    print("The remaining tokens like 'R-10011' may still create '##11' but the main noise sources are eliminated.")

if __name__ == "__main__":
    main()
