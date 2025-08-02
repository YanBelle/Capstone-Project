#!/usr/bin/env python3
"""
Debug script to trace preprocessing of the user's sample EJ text
"""

import re

def debug_preprocessing():
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

    print("=== ORIGINAL TEXT ===")
    print(repr(sample_text))
    print()
    
    # Step-by-step preprocessing to trace where each pattern gets removed
    text = sample_text
    
    print("=== STEP 1: Remove EJ header patterns [020t*nnn*mm/dd/yyyy*hh:mm* ===")
    print("Pattern: r'\\[020t\\*\\d+\\*\\d{2}/\\d{2}/\\d{4}\\*\\d{2}:\\d{2}\\*'")
    before = text
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    print(f"Removes: {set(re.findall(r'\\[020t\\*\\d+\\*\\d{2}/\\d{2}/\\d{4}\\*\\d{2}:\\d{2}\\*', before))}")
    print("After step 1:")
    print(repr(text))
    print()
    
    print("=== STEP 2: Remove remaining [020t patterns ===")
    print("Pattern: r'\\[020t\\s+'")
    before = text
    text = re.sub(r'\[020t\s+', '', text)
    print(f"Removes: {set(re.findall(r'\\[020t\\s+', before))}")
    print("After step 2:")
    print(repr(text))
    print()
    
    print("=== STEP 3: Remove standalone timestamps hh:mm:ss ===")
    print("Pattern: r'\\s+\\d{2}:\\d{2}:\\d{2}\\s+'")
    before = text
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    print(f"Removes: {set(re.findall(r'\\s+\\d{2}:\\d{2}:\\d{2}\\s+', before))}")
    print("After step 3:")
    print(repr(text))
    print()
    
    print("=== STEP 4: Remove transaction start markers ===")
    print("Pattern: r'\\s*---START OF TRANSACTION---\\s*'")
    before = text
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    print("After step 4:")
    print(repr(text))
    print()
    
    print("=== STEP 5: Replace *TRANSACTION START* ===")
    print("Pattern: r'\\*TRANSACTION START\\*'")
    before = text
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    print("After step 5:")
    print(repr(text))
    print()
    
    print("=== STEP 6: Replace PAN patterns ===")
    print("Pattern: r'PAN\\s+\\d{4}\\d+\\*+\\d+'")
    before = text
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    print("After step 6:")
    print(repr(text))
    print()
    
    print("=== STEP 7: Remove complex transaction codes ===")
    print("Pattern: r'\\*\\d+\\*\\d+\\*\\([^,]+,\\s*'")
    before = text
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    print("After step 7:")
    print(repr(text))
    print()
    
    print("=== COMPOUND PATTERNS (abbreviated) ===")
    # Apply key compound patterns
    compound_patterns = {
        r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
        r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
        r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
        r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
        r'\bATR\s+RECEIVED\b': 'ATR_RECEIVED',
        r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
        r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
    }
    
    for pattern, replacement in compound_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    print("After compound patterns:")
    print(repr(text))
    print()
    
    print("=== PUNCTUATION CLEANING ===")
    # Replace multiple asterisks with single underscore
    text = re.sub(r'\*+', '_', text)
    print("After asterisk replacement:")
    print(repr(text))
    print()
    
    # Clean up excessive parentheses and commas
    text = re.sub(r'[(),]+', ' ', text)
    print("After parentheses/comma cleanup:")
    print(repr(text))
    print()
    
    # Additional numeric cleaning
    text = re.sub(r':(\s*\d{3})\b', r' \1', text)  # Convert "ESC: 000" to "ESC 000"
    text = re.sub(r'\$(\d+)\.(\d{2})', r'AMOUNT_\1_\2', text)  # $100.00 -> AMOUNT_100_00
    text = re.sub(r'\b(REF|ESC|VAL):\s*(\d+)\b', r'\1_\2', text)  # REF: 000 -> REF_000
    text = re.sub(r'\s*[=:]\s*', ' ', text)  # Remove = and : with spaces
    
    print("After numeric cleaning:")
    print(repr(text))
    print()
    
    # Final whitespace cleanup
    text = ' '.join(text.split())
    
    print("=== FINAL PROCESSED TEXT ===")
    print(repr(text))
    print()
    print("FINAL TEXT:")
    print(text)
    print()
    
    print("=== ANALYSIS ===")
    print("Checking for patterns that might create ##11 and ##w tokens...")
    
    # Check if specific patterns in the original text cause these tokens
    print()
    print("=== CHECKING FOR *630* PATTERN ===")
    if '*630*' in sample_text:
        print("Found *630* in original text")
        print("The pattern '*630*06/18/2025*00:46*' is NOT removed by the current regex")
        print("Current regex only removes: [020t*nnn*date*time* (starts with [020t)")
        print("But '*630*06/18/2025*00:46*' doesn't start with [020t")
        print(f"*630* pattern still in processed text: {'*630*' in text}")
    
    if 'R-10011' in sample_text:
        print("\n=== CHECKING FOR R-10011 PATTERN ===")
        print("Found R-10011 in original text")
        r_pattern_remaining = 'R-10011' in text
        print(f"R-10011 still in processed text: {r_pattern_remaining}")
        
    print("\n=== IDENTIFYING THE ISSUE ===")
    print("The '*630*06/18/2025*00:46*' pattern is NOT being removed because:")
    print("1. Current regex: r'\\[020t\\*\\d+\\*\\d{2}/\\d{2}/\\d{4}\\*\\d{2}:\\d{2}\\*'")
    print("2. This pattern requires [020t at the start")
    print("3. But '*630*06/18/2025*00:46*' starts with * not [020t")
    print("4. So it remains in the text and gets processed by BERT")
    print("5. BERT's subword tokenization breaks '10011' into '100' + '##11'")
    print("6. And breaks other complex patterns into ## subwords")

if __name__ == "__main__":
    debug_preprocessing()
