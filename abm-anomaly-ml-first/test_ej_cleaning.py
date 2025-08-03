#!/usr/bin/env python3
"""
Test script to demonstrate EJ text cleaning process for BERT tokenization
"""

import re

def preprocess_text(text: str) -> str:
    """Preprocess ABM log text for BERT analysis with enhanced pattern cleaning"""
    print("=== BERT EJ CLEANING PROCESS ===")
    print(f"Original text length: {len(text)} characters")
    print(f"Original text:\n{repr(text)}\n")
    
    # Enhanced EJ pattern cleaning with specific fixes for BERT attention optimization
    
    # 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
    print("Step 1: Remove EJ header patterns [020t*<sequence>*<mm/dd/yyyy>*<hh:mm>*")
    before = text
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 1b. Remove standalone date/time patterns that don't start with [020t
    print("\nStep 1b: Remove standalone date/time patterns *630*06/18/2025*00:46*")
    before = text
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 1c. Remove complex transaction code patterns
    print("\nStep 1c: Remove complex transaction code patterns *7231*1*(Iw(1*3,")
    before = text
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 2. Remove remaining [020t patterns with any following content
    print("\nStep 2: Remove remaining [020t patterns")
    before = text
    text = re.sub(r'\[020t\s+', '', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 3. Remove standalone timestamps in format hh:mm:ss
    print("\nStep 3: Remove standalone timestamps hh:mm:ss")
    before = text
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 4. Remove transaction start markers
    print("\nStep 4: Remove transaction start markers")
    before = text
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    if text != before:
        print(f"  REMOVED: {len(before) - len(text)} characters")
        print(f"  After: {repr(text[:200])}...")
    
    # 5. ENHANCED PATTERN CLEANING - Fix specific issues with punctuation and compound words
    print("\nStep 5: Enhanced pattern cleaning")
    
    # Replace *TRANSACTION START* with TRANSACTION START (remove asterisks)
    print("  5a: Replace *TRANSACTION START* with TRANSACTION_START")
    before = text
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    if text != before:
        print(f"    CHANGED: {repr(before[:100])} -> {repr(text[:100])}")
    
    # Replace PAN patterns with simplified CardNumber label
    print("  5b: Replace PAN patterns with CardNumber")
    before = text
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    if text != before:
        print(f"    CHANGED: Found PAN pattern, replaced with CardNumber")
    
    # Remove complex transaction codes
    print("  5c: Remove remaining complex transaction codes")
    before = text
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    if text != before:
        print(f"    REMOVED: {len(before) - len(text)} characters")
    
    # Create compound tokens for ATM events that should stay together
    print("  5d: Create compound tokens for multi-word ATM events")
    compound_patterns = {
        # Core ATM events
        r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
        r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
        r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
        r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
        r'\bATR\s+RECEIVED\b': 'ATR_RECEIVED',
        r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
        r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
        
        # Additional ATM operations
        r'\bCASH\s+DISPENSED\b': 'CASH_DISPENSED',
        r'\bBALANCE\s+INQUIRY\b': 'BALANCE_INQUIRY',
        r'\bRECEIPT\s+PRINTED\b': 'RECEIPT_PRINTED',
        r'\bCARD\s+RETAINED\b': 'CARD_RETAINED',
        r'\bCARD\s+EJECTED\b': 'CARD_EJECTED',
        r'\bCARD\s+READ\b': 'CARD_READ',
        
        # Error states and conditions
        r'\bTIMEOUT\s+ERROR\b': 'TIMEOUT_ERROR',
        r'\bCOMMUNICATION\s+ERROR\b': 'COMMUNICATION_ERROR',
        r'\bNETWORK\s+ERROR\b': 'NETWORK_ERROR',
        r'\bCASH\s+DISPENSER\s+ERROR\b': 'CASH_DISPENSER_ERROR',
        r'\bREAD\s+ERROR\b': 'read_ERROR',
        r'\bWRITE\s+ERROR\b': 'WRITE_ERROR',
        
        # Account and validation
        r'\bACCOUNT\s+VALIDATION\b': 'ACCOUNT_VALIDATION',
        r'\bPIN\s+VALIDATION\b': 'PIN_VALIDATION',
        r'\bINSUFFICIENT\s+FUNDS\b': 'INSUFFICIENT_FUNDS',
        r'\bINVALID\s+PIN\b': 'INVALID_PIN',
        r'\bCARD\s+EXPIRED\b': 'CARD_EXPIRED',
        
        # Transaction types
        r'\bWITHDRAWAL\s+TRANSACTION\b': 'WITHDRAWAL_TRANSACTION',
        r'\bDEPOSIT\s+TRANSACTION\b': 'DEPOSIT_TRANSACTION',
        r'\bTRANSFER\s+TRANSACTION\b': 'TRANSFER_TRANSACTION',
        
        # Status indicators
        r'\bOUT\s+OF\s+SERVICE\b': 'OUT_OF_SERVICE',
        r'\bOUT\s+OF\s+CASH\b': 'OUT_OF_CASH',
        r'\bOUT\s+OF\s+ORDER\b': 'OUT_OF_ORDER',
        r'\bSERVICE\s+MODE\b': 'SERVICE_MODE',
        r'\bDIAGNOSTIC\s+MODE\b': 'DIAGNOSTIC_MODE',
    }
    
    changes_made = []
    for pattern, replacement in compound_patterns.items():
        before = text
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if text != before:
            changes_made.append(f"{pattern} -> {replacement}")
    
    if changes_made:
        print(f"    COMPOUND TOKENS CREATED: {len(changes_made)} patterns")
        for change in changes_made[:5]:  # Show first 5 changes
            print(f"      {change}")
        if len(changes_made) > 5:
            print(f"      ... and {len(changes_made) - 5} more")
    
    # Reduce excessive punctuation that gets high attention scores
    print("  5e: Clean up excessive punctuation")
    before = text
    text = re.sub(r'\*+', '_', text)  # Replace multiple asterisks with single underscore
    text = re.sub(r'[(),]+', ' ', text)  # Clean up excessive parentheses and commas
    if text != before:
        print(f"    CLEANED: Replaced excessive punctuation")
    
    # Additional punctuation cleaning for better BERT focus
    print("  5f: Normalize punctuation patterns")
    before = text
    text = re.sub(r':(\s*\d{3})\b', r' \1', text)  # Convert "ESC: 000" to "ESC 000"
    if text != before:
        print(f"    NORMALIZED: Colon patterns")
    
    # Normalize numeric patterns to reduce fragmentation
    print("  5g: Normalize numeric patterns")
    before = text
    text = re.sub(r'\$(\d+)\.(\d{2})', r'AMOUNT_\1_\2', text)  # $100.00 -> AMOUNT_100_00
    text = re.sub(r'\b(REF|ESC|VAL):\s*(\d+)\b', r'\1_\2', text)  # REF: 000 -> REF_000
    if text != before:
        print(f"    NORMALIZED: Numeric patterns")
    
    # Clean up excessive whitespace around punctuation
    print("  5h: Clean up whitespace")
    before = text
    text = re.sub(r'\s*[=:]\s*', ' ', text)  # Remove = and : with spaces
    if text != before:
        print(f"    CLEANED: Whitespace around punctuation")
    
    # 6. Remove excessive whitespace and clean up
    print("\nStep 6: Final whitespace cleanup")
    before_len = len(text)
    text = ' '.join(text.split())
    after_len = len(text)
    if before_len != after_len:
        print(f"  CLEANED: Normalized whitespace")
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Final text length: {len(text)} characters")
    original_len = len("""[020t*629*06/18/2025*00:46*
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
[020t*631*06/18/2025*00:47*""")
    print(f"Reduction: {original_len - len(text)} characters ({((original_len - len(text)) / original_len * 100):.1f}%)")
    print(f"Final text for BERT tokenization:")
    print(f"{repr(text)}")
    print(f"\nHuman-readable final text:")
    print(f"{text}")
    
    return text

# Your sample EJ text
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

if __name__ == "__main__":
    print("Testing EJ cleaning process with your sample...")
    print("=" * 50)
    
    cleaned_text = preprocess_text(sample_ej)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Original length: {len(sample_ej)} characters")
    print(f"Cleaned length: {len(cleaned_text)} characters") 
    print(f"Reduction: {len(sample_ej) - len(cleaned_text)} characters ({((len(sample_ej) - len(cleaned_text)) / len(sample_ej) * 100):.1f}%)")
    print(f"\nThis cleaned text is what gets sent to BERT for tokenization!")
