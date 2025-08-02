#!/usr/bin/env python3
"""
Test script to verify the enhanced noise removal patterns
specifically targeting the problematic tokens: "##31","1","##w", "3", "s", "47", "15"
"""

import re

def enhanced_preprocess_text(text):
    """
    Enhanced preprocessing function with new noise removal patterns
    """
    # 1. Remove date/time header patterns (same as before)
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # 1c. Remove complex transaction code patterns (original)
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    
    # 1d. NEW: Remove any remaining complex patterns with asterisks and parentheses
    # This catches patterns like "*7231*1*(Iw(1*3," more aggressively
    text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
    
    # 2. Remove remaining [020t patterns
    text = re.sub(r'\[020t\s+', '', text)
    
    # 3. Remove standalone timestamps (original)
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    
    # 3b. NEW: Remove standalone timestamps in format hh:mm (without seconds)
    text = re.sub(r'\s+\d{2}:\d{2}\s+', ' ', text)
    
    # 3c. NEW: Remove isolated time digits that could create noise tokens
    # This catches remaining time fragments like standalone "47", "15", etc.
    text = re.sub(r'\b\d{2}\b(?=\s|$)', '', text)
    
    # 4. Remove transaction markers
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    
    # 5. Enhanced pattern cleaning
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    text = re.sub(r'\bA/C\b', '', text)
    
    # Clean up "REJECTS:000*(1\nS" to just "REJECTS_000" (original)
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
    
    # NEW: Additional cleanup for any remaining REJECTS fragments
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
    
    # NEW: Handle remaining REJECTS:000 patterns that don't have the full pattern
    text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
    
    # NEW: Remove standalone "S" that might be left from REJECTS patterns
    text = re.sub(r'\bS\b(?=\s|$)', '', text)
    
    # Convert patterns to compound tokens
    text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
    text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
    
    # NEW: Additional noise cleanup - remove isolated digits and fragments
    # Remove standalone single digits that create noise
    text = re.sub(r'\b\d\b(?=\s|$)', '', text)
    
    # NEW: Remove isolated asterisks and punctuation fragments
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'[()]+', '', text)
    
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

def test_noise_removal():
    """Test the enhanced noise removal on the sample EJ text"""
    
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

    print("=== NOISE REMOVAL TEST ===")
    print()
    print("Original EJ text:")
    print("=" * 50)
    print(sample_ej)
    print("=" * 50)
    print()
    
    # Apply preprocessing
    cleaned_text = enhanced_preprocess_text(sample_ej)
    
    print("After enhanced preprocessing:")
    print("=" * 50)
    print(cleaned_text)
    print("=" * 50)
    print()
    
    # Check for specific problematic patterns
    print("=== NOISE TOKEN ANALYSIS ===")
    
    # Split into tokens for analysis
    tokens = cleaned_text.split()
    
    # Check for the problematic tokens
    problematic_tokens = ["31", "1", "w", "3", "s", "47", "15"]
    found_problematic = []
    
    for token in tokens:
        if token.lower() in problematic_tokens:
            found_problematic.append(token)
    
    print(f"Problematic tokens still found: {found_problematic}")
    
    # Check for specific patterns we're trying to remove
    pattern_checks = [
        ("*7231*1*(Iw(1*3,", "Complex transaction code pattern"),
        ("00:47:13", "Timestamp 47:13"),
        ("00:47:15", "Timestamp 47:15"),
        ("REJECTS:000*(1", "REJECTS pattern start"),
        ("\nS", "Standalone S"),
        ("A/C", "Account pattern"),
    ]
    
    print("\n=== PATTERN REMOVAL CHECKS ===")
    for pattern, description in pattern_checks:
        if pattern in cleaned_text:
            print(f"❌ {description}: STILL PRESENT - '{pattern}'")
        else:
            print(f"✅ {description}: REMOVED")
    
    # Check for successfully created compound tokens
    print("\n=== COMPOUND TOKEN CREATION ===")
    compound_checks = [
        ("TRANSACTION_START", "Transaction start"),
        ("TRANSACTION_END", "Transaction end"),
        ("CARD_INSERTED", "Card inserted"),
        ("CARD_TAKEN", "Card taken"),
        ("PIN_ENTERED", "PIN entered"),
        ("DEVICE_ERROR", "Device error"),
        ("ATR_RECEIVED_T_0", "ATR received"),
        ("OPCODE_FI", "OpCode FI"),
        ("OPCODE_IB", "OpCode IB"),
        ("CardNumber", "PAN replacement"),
        ("ESC_000", "ESC pattern"),
        ("VAL_000", "VAL pattern"),
        ("REF_000", "REF pattern"),
        ("REJECTS_000", "REJECTS pattern"),
    ]
    
    for compound, description in compound_checks:
        if compound in cleaned_text:
            print(f"✅ {description}: {compound}")
        else:
            print(f"❌ {description}: NOT FOUND")
    
    print(f"\nFinal token count: {len(tokens)}")
    print(f"Final tokens: {tokens}")
    
    return cleaned_text

if __name__ == "__main__":
    result = test_noise_removal()
    print("\n🔍 Noise removal test completed!")
