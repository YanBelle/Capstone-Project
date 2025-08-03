#!/usr/bin/env python3
"""
Test the enhanced BERT preprocessing to verify specific pattern fixes
- Remove excessive punctuation attention (*,)
- Replace *TRANSACTION START* with TRANSACTION_START
- Replace PAN patterns with CardNumber
- Remove complex transaction codes
- Create compound tokens for multi-word ATM events
"""

import re

def enhanced_preprocess_text(text: str) -> str:
    """Enhanced preprocessing with all the specific fixes"""
    
    # 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # 2. Remove remaining [020t patterns with any following content
    text = re.sub(r'\[020t\s+', '', text)
    
    # 3. Remove standalone timestamps in format hh:mm:ss
    text = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', text)
    
    # 4. Remove transaction start markers
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    
    # 5. ENHANCED PATTERN CLEANING - Address specific user issues
    
    # Replace *TRANSACTION START* with TRANSACTION START (remove asterisks)
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    
    # Replace PAN patterns with simplified CardNumber label
    # Matches: "PAN 0004263********1897" or similar patterns
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    
    # Remove complex transaction codes like "*7231*1*(Iw(1*3," but keep meaningful parts
    # Pattern: *digits*digits*(complex_chars*digits, -> keep what follows after comma
    text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
    
    # Create compound tokens for ATM events that should stay together
    # This prevents BERT from splitting important multi-word terms
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
    
    # Reduce excessive punctuation that gets high attention scores
    # Replace multiple asterisks with single underscore
    text = re.sub(r'\*+', '_', text)
    
    # Clean up excessive parentheses and commas that fragment attention
    text = re.sub(r'[(),]+', ' ', text)
    
    # 6. Remove excessive whitespace and clean up
    text = ' '.join(text.split())
    
    return text

# Test with the user's sample EJ data
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

print("=== ENHANCED BERT PREPROCESSING TEST ===\n")

print("ORIGINAL TEXT:")
print(f"Length: {len(sample_ej)} characters")
print(f"Text: {repr(sample_ej)}")
print("\n" + "="*80 + "\n")

# Step-by-step processing to show each transformation
text = sample_ej

print("STEP-BY-STEP CLEANING:")

# Step 1: Remove EJ headers
step1 = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
print(f"1. After EJ header removal: {len(step1)} chars")
print(f"   {repr(step1[:100])}...")

# Step 2: Remove [020t patterns
step2 = re.sub(r'\[020t\s+', '', step1)
print(f"2. After [020t removal: {len(step2)} chars")
print(f"   {repr(step2[:100])}...")

# Step 3: Remove timestamps
step3 = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', step2)
print(f"3. After timestamp removal: {len(step3)} chars")
print(f"   {repr(step3[:100])}...")

# Step 4: Remove transaction markers
step4 = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', step3)
print(f"4. After transaction marker removal: {len(step4)} chars")
print(f"   {repr(step4[:100])}...")

# Step 5: Replace *TRANSACTION START*
step5 = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', step4)
print(f"5. After *TRANSACTION START* fix: {len(step5)} chars")
print(f"   {repr(step5[:100])}...")

# Step 6: Replace PAN patterns
step6 = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', step5)
print(f"6. After PAN replacement: {len(step6)} chars")
print(f"   {repr(step6[:100])}...")

# Step 7: Remove complex transaction codes
step7 = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', step6)
print(f"7. After complex code removal: {len(step7)} chars")
print(f"   {repr(step7[:100])}...")

# Step 8: Create compound tokens
compound_patterns = {
    r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
    r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
    r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
    r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
    r'\bATR\s+RECEIVED\b': 'ATR_RECEIVED',
    r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
    r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
}

step8 = step7
for pattern, replacement in compound_patterns.items():
    step8 = re.sub(pattern, replacement, step8, flags=re.IGNORECASE)
    
print(f"8. After compound token creation: {len(step8)} chars")
print(f"   {repr(step8[:100])}...")

# Step 9: Clean punctuation
step9 = re.sub(r'\*+', '_', step8)
step9 = re.sub(r'[(),]+', ' ', step9)
print(f"9. After punctuation cleaning: {len(step9)} chars")
print(f"   {repr(step9[:100])}...")

# Final cleanup
final_text = ' '.join(step9.split())
print(f"10. Final cleanup: {len(final_text)} chars")
print(f"    {repr(final_text)}")

print("\n" + "="*80 + "\n")

# Run the complete function
processed_text = enhanced_preprocess_text(sample_ej)

print("FINAL RESULTS:")
print(f"Original length: {len(sample_ej)} characters")
print(f"Processed length: {len(processed_text)} characters")
print(f"Reduction: {((len(sample_ej) - len(processed_text)) / len(sample_ej)) * 100:.1f}%")
print()
print("PROCESSED TEXT:")
print(f"'{processed_text}'")
print()

print("KEY IMPROVEMENTS:")
print("✅ *TRANSACTION START* → TRANSACTION_START")
print("✅ PAN 0004263********1897 → CardNumber") 
print("✅ *7231*1*(Iw(1*3, M-02, R-10011 → M-02 R-10011")
print("✅ DEVICE ERROR → DEVICE_ERROR (compound token)")
print("✅ CARD INSERTED → CARD_INSERTED (compound token)")
print("✅ PIN ENTERED → PIN_ENTERED (compound token)")
print("✅ CARD TAKEN → CARD_TAKEN (compound token)")
print("✅ TRANSACTION END → TRANSACTION_END (compound token)")
print("✅ Reduced punctuation attention (*,(,) → cleaned")
print("✅ Preserved critical content: DEVICE_ERROR, ESC, VAL, REF, REJECTS")

print("\nCRITICAL CONTENT PRESERVED:")
critical_terms = ['DEVICE_ERROR', 'ESC', 'VAL', 'REF', 'REJECTS', 'CardNumber', 'TRANSACTION_START', 'TRANSACTION_END']
for term in critical_terms:
    if term in processed_text:
        print(f"✅ {term} - PRESERVED")
    else:
        print(f"❌ {term} - MISSING")
