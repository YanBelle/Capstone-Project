#!/usr/bin/env python3
"""
Test the additional fragmentation fixes for M-02, R-10011
"""

import re

def enhanced_preprocess_text_v2(text):
    """Enhanced preprocessing with M-02, R-10011 fixes"""
    # All previous patterns...
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
    text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
    text = re.sub(r'\[020t\s+', '', text)
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    text = re.sub(r'\s+\d{2}:\d{2}\s+', ' ', text)
    text = re.sub(r'\b\d{2}\b(?=\s|$)', '', text)
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    text = re.sub(r'\bA/C\b', '', text)
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
    text = re.sub(r'\bS\b(?=\s|$)', '', text)
    text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
    text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
    text = re.sub(r'\b\d\b(?=\s|$)', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'[()]+', '', text)
    
    # NEW: Clean specific EJ patterns that cause fragmentation
    # Convert M-02, R-10011 to compound tokens to prevent BERT fragmentation
    text = re.sub(r'\bM-02,?\s*', 'M_02 ', text)
    text = re.sub(r'\bR-10011\b', 'R_10011', text)
    
    # Compound token creation
    compound_patterns = [
        (r'\bTRANSACTION START\b', 'TRANSACTION_START'),
        (r'\bTRANSACTION END\b', 'TRANSACTION_END'),
        (r'\bCARD INSERTED\b', 'CARD_INSERTED'),
        (r'\bCARD TAKEN\b', 'CARD_TAKEN'),
        (r'\bPIN ENTERED\b', 'PIN_ENTERED'),
        (r'\bDEVICE ERROR\b', 'DEVICE_ERROR'),
    ]
    
    for pattern, replacement in compound_patterns:
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Test with the sample
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

print("=== FRAGMENTATION FIX TEST ===")
print()

result = enhanced_preprocess_text_v2(sample_text)
print("Enhanced preprocessing result (v2):")
print("=" * 50)
print(result)
print("=" * 50)
print()

tokens = result.split()
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print()

# Check for the changes
print("FRAGMENTATION PREVENTION:")
print(f"✅ M-02, → M_02: {'M_02' in tokens}")
print(f"✅ R-10011 → R_10011: {'R_10011' in tokens}")
print(f"❌ Old problematic patterns: {[t for t in tokens if '-' in t and t not in ['M_02', 'R_10011']]}")
print()

print("🎯 This should eliminate BERT fragmentation of M-02, and R-10011!")
