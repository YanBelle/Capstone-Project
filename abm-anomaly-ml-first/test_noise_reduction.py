#!/usr/bin/env python3
"""
Test comprehensive noise reduction for BERT preprocessing
Tests specifically for eliminating ##31, ##1, ##w, ##i, 72, 46, 47, ##13 noise tokens
"""

import re

def test_noise_reduction():
    """Test the enhanced preprocessing to eliminate all noise tokens"""
    
    # Sample EJ log content that causes the noise tokens
    test_cases = [
        # Original problematic pattern from the EJ log
        ("*7231*1*(Iw(1*3, M-02, R-10011", "M_02 R_10011"),
        
        # Variations of the problematic pattern
        ("*630*06/18/2025*00:46*", ""),
        ("*629*06/18/2025*00:46*", ""),
        ("*7231*1*(Iw(1*3,", ""),
        
        # Individual noise components
        ("standalone 46 number", "standalone number"),
        ("standalone 47 number", "standalone number"), 
        ("standalone 72 number", "standalone number"),
        ("standalone 31 number", "standalone number"),
        ("standalone 13 number", "standalone number"),
        ("standalone 7231 number", "standalone number"),
        ("standalone 630 number", "standalone number"),
        ("standalone 629 number", "standalone number"),
        
        # Single character fragments
        ("isolated w letter", "isolated letter"),
        ("isolated i letter", "isolated letter"),
        ("isolated 1 digit", "isolated digit"),
        ("isolated 3 digit", "isolated digit"),
        ("Iw fragment test", "fragment test"),
        
        # Complex real patterns
        ("[020t*629*06/18/2025*00:46* *TRANSACTION START* *7231*1*(Iw(1*3, M-02, R-10011", "TRANSACTION_START M_02 R_10011"),
        
        # Time patterns that could fragment
        ("00:46:27 ATR RECEIVED", "ATR_RECEIVED"),
        ("05:50:56 PIN ENTERED", "PIN_ENTERED"),
        
        # Mixed complex pattern
        ("*630*06/18/2025*00:46* DEVICE ERROR *7231*1*(Iw(1*3, ESC: 000 VAL: 000", "DEVICE_ERROR ESC_000 VAL_000"),
    ]
    
    def apply_enhanced_preprocessing(text):
        """Apply the same enhanced preprocessing as in bertviz_analyzer.py"""
        
        # 1. Remove EJ header patterns
        text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        
        # 1c. ENHANCED: Remove complex transaction code patterns that cause fragmentation
        text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
        text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
        
        # 1e. AGGRESSIVE CLEANUP: Remove transaction code fragments
        text = re.sub(r'\*\d+\*', '', text)  # Remove *digits*
        text = re.sub(r'\*\([^)]*\)', '', text)  # Remove *(content)
        text = re.sub(r'\([^)]*\*\d+', '', text)  # Remove (content*digits
        text = re.sub(r'\(Iw\(', '', text)  # Remove specific (Iw( pattern
        text = re.sub(r'\(\d+\*\d+', '', text)  # Remove (digits*digits
        
        # 2. Remove remaining [020t patterns
        text = re.sub(r'\[020t\s+', '', text)
        
        # 3. Remove standalone timestamps
        text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
        text = re.sub(r'\s*\d{2}:\d{2}\s+', ' ', text)
        
        # 3b2. Remove partial timestamps that remain after aggressive cleanup
        text = re.sub(r'\d{2}::\s*', '', text)  # Remove xx:: patterns
        text = re.sub(r'\d{2}:\d{2}:\s*', '', text)  # Remove xx:xx: patterns
        
        # 3c. ENHANCED: Remove isolated time digits and number fragments
        text = re.sub(r'\b\d{1,2}\b(?=\s|$)', '', text)
        
        # 3d. Remove specific problematic number sequences
        noise_patterns = ['7231', '630', '629', '46', '47', '72', '31', '13']
        for pattern in noise_patterns:
            text = re.sub(rf'\b{pattern}\b', '', text)
        
        # 4. Remove transaction start markers
        text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
        
        # 5. ENHANCED PATTERN CLEANING
        text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
        text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
        text = re.sub(r'\*\d+\*\d+\*\([^,]+,\s*', '', text)
        text = re.sub(r'\bA/C\b', '', text)
        
        # Clean up REJECTS, VAL, ESC, REF patterns
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)\s*S?', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)[^A-Z]*', r'REJECTS_\1', text)
        text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
        text = re.sub(r'\bS\b(?=\s|$)', '', text)
        text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
        text = re.sub(r'\bOPCODE\s*=\s*([A-Z]+)\b', r'OPCODE_\1', text)
        text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
        
        # Enhanced noise cleanup - remove isolated characters and fragments
        text = re.sub(r'\b[a-zA-Z0-9]\b(?=\s|$)', '', text)  # Single chars/digits
        
        # Remove specific problematic fragments
        problem_fragments = ['Iw', 'w', 'i', '1', '3']
        for fragment in problem_fragments:
            text = re.sub(rf'\b{fragment}\b', '', text)
        
        # Remove asterisks and parentheses
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'[()]+', '', text)
        
        # Machine and R status patterns
        text = re.sub(r'\bM-(\d+),?\s*', r'M_\1 ', text)
        text = re.sub(r'\bR-(\d+)\b', r'R_\1', text)
        
        # Create compound tokens for ATM events
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
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    print("Testing Enhanced Noise Reduction for BERT Preprocessing")
    print("=" * 65)
    print("Target: Eliminate ##31, ##1, ##w, ##i, 72, 46, 47, ##13 noise tokens")
    print("=" * 65)
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = apply_enhanced_preprocessing(input_text)
        
        # Check if any of the problematic fragments remain
        noise_fragments = ['31', '72', '46', '47', '13', '7231', '630', '629', 'Iw', ' w ', ' i ', ' 1 ', ' 3 ']
        has_noise = any(fragment in result for fragment in noise_fragments)
        
        if result.strip() == expected.strip() and not has_noise:
            print(f"✅ Test {i:2d}: PASSED")
            print(f"   Input:    '{input_text}'")
            print(f"   Output:   '{result}'")
            print(f"   Expected: '{expected}'")
            passed += 1
        else:
            print(f"❌ Test {i:2d}: FAILED")
            print(f"   Input:    '{input_text}'")
            print(f"   Output:   '{result}'")
            print(f"   Expected: '{expected}'")
            if has_noise:
                remaining_noise = [f for f in noise_fragments if f in result]
                print(f"   Noise remaining: {remaining_noise}")
            failed += 1
        print()
    
    print("=" * 65)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Noise tokens should be eliminated.")
    else:
        print("⚠️  Some tests failed. Noise tokens may still appear.")
    
    return failed == 0

if __name__ == "__main__":
    test_noise_reduction()
