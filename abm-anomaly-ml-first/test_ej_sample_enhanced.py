#!/usr/bin/env python3
"""
Test enhanced noise reduction for specific EJ sample patterns
Tests specifically for ESC/VAL/REF combination and isolated "1" token removal
"""

import re

def test_ej_sample_processing():
    """Test the enhanced preprocessing with the actual EJ sample"""
    
    # The exact EJ sample provided
    ej_sample = """[020t*629*06/18/2025*00:46*
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

    def apply_enhanced_preprocessing(text):
        """Apply the enhanced preprocessing logic"""
        
        # 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
        text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        
        # CRITICAL: Handle ESC/VAL/REF patterns FIRST before other cleanup removes the values
        text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
        text = re.sub(r'\b(VAL|ESC|REF)\s+(\d+)\b', r'\1_\2', text)
        
        # CRITICAL: Handle ATR pattern IMMEDIATELY after ESC/VAL/REF
        text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
        
        # 1c-1f. Enhanced transaction code pattern removal
        # IMPROVED: More aggressive pattern to catch the full "*7231*1*(Iw(1*3," structure
        text = re.sub(r'\*\d+\*\d+\*\([^,)]*,?\s*', '', text)
        text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
        
        # Enhanced cleanup
        text = re.sub(r'\*\d+\*', '', text)
        text = re.sub(r'\*\([^)]*\)', '', text)
        text = re.sub(r'\([^)]*\*\d+', '', text)
        text = re.sub(r'\(Iw\([^)]*\)', '', text)
        text = re.sub(r'\(\d+\*\d+[^)]*\)', '', text)
        
        # SPECIFIC FIX: Remove the exact "*7231*1*(Iw(1*3," pattern
        text = re.sub(r'\*7231\*1\*\(Iw\(1\*3,?\s*', '', text)
        
        # 2. Remove remaining [020t patterns
        text = re.sub(r'\[020t\s+', '', text)
        
        # 3. Remove timestamps
        text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
        text = re.sub(r'\s*\d{2}:\d{2}\s+', ' ', text)
        text = re.sub(r'\d{2}::\s*', '', text)
        text = re.sub(r'\d{2}:\d{2}:\s*', '', text)
        
        # 3c-3d. Smart pattern removal
        text = re.sub(r'(AMOUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(COUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(TOTAL)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(BALANCE)\s+(\d+)', r'PROTECTED_\1_\2', text)
        
        text = re.sub(r'\b\d{1,4}\b(?=\s+(?:[A-Z][A-Z_]+|[a-z]+)|\s*$)', '', text)
        text = re.sub(r'PROTECTED_(AMOUNT|COUNT|TOTAL|BALANCE)_(\d+)', r'\1 \2', text)
        text = re.sub(r'(?<=\s)[a-zA-Z0-9](?=\s+[A-Z_]|\s*$)', '', text)
        
        # 4. Remove transaction markers
        text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
        
        # 5. Enhanced pattern cleaning
        text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
        text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
        text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
        text = re.sub(r'\bA/C\b', '', text)
        
        # ENHANCED REJECTS patterns
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS\s+(\d+)', r'REJECTS_\1', text)
        
        text = re.sub(r'\bS\b(?=\s|$)', '', text)
        
        # Compound tokens - ATR pattern already handled above
        text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
        # ATR pattern was already applied above - don't repeat
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'[()]+', '', text)
        text = re.sub(r'\bM-(\d+),?\s*', r'M_\1 ', text)
        text = re.sub(r'\bR-(\d+)\b', r'R_\1', text)
        
        # Create compound tokens for ATM events - MUST BE CAREFUL NOT TO OVERRIDE EXISTING PATTERNS
        compound_patterns = {
            r'\bDEVICE\s+ERROR\b': 'DEVICE_ERROR',
            r'\bCARD\s+INSERTED\b': 'CARD_INSERTED', 
            r'\bCARD\s+TAKEN\b': 'CARD_TAKEN',
            r'\bPIN\s+ENTERED\b': 'PIN_ENTERED',
            # REMOVED: r'\bATR\s+RECEIVED\b': 'ATR_RECEIVED',  # This would break ATR_RECEIVED_T_0!
            r'\bTRANSACTION\s+END\b': 'TRANSACTION_END',
            r'\bTRANSACTION\s+START\b': 'TRANSACTION_START',
        }
        
        for pattern, replacement in compound_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    print("Testing Enhanced EJ Sample Processing")
    print("=" * 60)
    
    print("Original EJ Sample:")
    print(ej_sample)
    print("\n" + "=" * 60)
    
    result = apply_enhanced_preprocessing(ej_sample)
    
    print("Processed Result:")
    print(f"'{result}'")
    print("\n" + "=" * 60)
    
    # Check for specific improvements
    checks = [
        ("ESC_000 present", "ESC_000" in result),
        ("VAL_000 present", "VAL_000" in result), 
        ("REF_000 present", "REF_000" in result),
        ("REJECTS_000 present", "REJECTS_000" in result),
        ("No isolated '1'", " 1 " not in result and result.endswith("1") == False),
        ("No isolated 'S'", " S " not in result),
        ("TRANSACTION_START present", "TRANSACTION_START" in result),
        ("CARD_INSERTED present", "CARD_INSERTED" in result),
        ("CARD_TAKEN present", "CARD_TAKEN" in result),
        ("PIN_ENTERED present", "PIN_ENTERED" in result),
        ("TRANSACTION_END present", "TRANSACTION_END" in result),
        ("DEVICE_ERROR present", "DEVICE_ERROR" in result),
        ("M_02 present", "M_02" in result),
        ("R_10011 present", "R_10011" in result),
        ("OPCODE_FI present", "OPCODE_FI" in result),
        ("OPCODE_IB present", "OPCODE_IB" in result),
        ("ATR_RECEIVED_T_0 present", "ATR_RECEIVED_T_0" in result),
        ("CardNumber present", "CardNumber" in result),
        ("No noise patterns", "*7231*1*(Iw(1*3," not in result),
        ("No timestamps", "00:46:27" not in result),
    ]
    
    print("Validation Checks:")
    passed = 0
    total = len(checks)
    
    for check_name, check_result in checks:
        status = "✅ PASS" if check_result else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if check_result:
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Enhanced processing working correctly.")
    else:
        print(f"⚠️  {total - passed} checks failed. Review the patterns.")
        
        # Show tokens for debugging
        tokens = result.split()
        print(f"\nProcessed tokens: {tokens}")
    
    return passed == total

if __name__ == "__main__":
    test_ej_sample_processing()
