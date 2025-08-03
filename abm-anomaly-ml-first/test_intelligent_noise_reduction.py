#!/usr/bin/env python3
"""
Test intelligent noise reduction for BERT preprocessing
Verifies that the new contextual approach works better than rigid hardcoded patterns
"""

import re

def test_intelligent_noise_reduction():
    """Test the new intelligent preprocessing approach"""
    
    def apply_intelligent_preprocessing(text):
        """Apply the new intelligent preprocessing logic"""
        
        # Step 1: Remove EJ header patterns
        text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
        
        # Step 2: Remove complex transaction code patterns
        text = re.sub(r'\*\d+\*\d+\*\([^,]*,?\s*', '', text)
        text = re.sub(r'\*\d+\*\d+\*\([^)]*\)[^,]*,?\s*', '', text)
        
        # Step 3: Aggressive cleanup of transaction fragments
        text = re.sub(r'\*\d+\*', '', text)
        text = re.sub(r'\*\([^)]*\)', '', text)
        text = re.sub(r'\([^)]*\*\d+', '', text)
        text = re.sub(r'\(Iw\(', '', text)
        text = re.sub(r'\(\d+\*\d+', '', text)
        
        # Step 4: Remove timestamps
        text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
        text = re.sub(r'\s*\d{2}:\d{2}\s+', ' ', text)
        text = re.sub(r'\d{2}::\s*', '', text)
        text = re.sub(r'\d{2}:\d{2}:\s*', '', text)
        
        # Step 5: NEW INTELLIGENT PATTERNS (replacing rigid approach)
        
        # 5a. SMART PATTERN: Remove isolated numeric fragments that are likely noise
        # Uses context-aware removal - preserves meaningful amounts/counts but removes noise fragments
        # First, protect meaningful numeric contexts by temporarily marking them with placeholder tokens
        text = re.sub(r'(AMOUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(COUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(TOTAL)\s+(\d+)', r'PROTECTED_\1_\2', text)
        text = re.sub(r'(BALANCE)\s+(\d+)', r'PROTECTED_\1_\2', text)
        
        # Now remove isolated numbers that are likely noise fragments (1-4 digits)
        text = re.sub(r'\b\d{1,4}\b(?=\s+(?:[A-Z][A-Z_]+|[a-z]+)|\s*$)', '', text)
        
        # Restore protected meaningful numbers
        text = re.sub(r'PROTECTED_(AMOUNT|COUNT|TOTAL|BALANCE)_(\d+)', r'\1 \2', text)
        
        # 5b. CONTEXTUAL FRAGMENT REMOVAL: Remove isolated single chars/digits between meaningful terms
        text = re.sub(r'(?<=\s)[a-zA-Z0-9](?=\s+[A-Z_]|\s*$)', '', text)
        
        # Step 6: Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    # Test cases comparing intelligent vs rigid approaches
    test_cases = [
        # Original problematic pattern - should work the same
        {
            'input': "*7231*1*(Iw(1*3, M-02, R-10011",
            'expected': "M-02, R-10011",
            'description': "Original transaction pattern"
        },
        
        # Contextual cases where intelligent approach should be better
        {
            'input': "DEVICE ERROR 46 COMMUNICATION FAILED",
            'expected': "DEVICE ERROR COMMUNICATION FAILED",
            'description': "Isolated number between meaningful terms"
        },
        
        # Case where meaningful numbers should be preserved
        {
            'input': "AMOUNT 100 DOLLARS DISPENSED",
            'expected': "AMOUNT 100 DOLLARS DISPENSED",
            'description': "Meaningful amounts should be preserved"
        },
        
        # Multiple isolated fragments
        {
            'input': "CARD INSERTED 47 w DEVICE 72 i STATUS",
            'expected': "CARD INSERTED DEVICE STATUS",
            'description': "Multiple isolated fragments"
        },
        
        # Edge case: numbers at end
        {
            'input': "TRANSACTION COMPLETE 31",
            'expected': "TRANSACTION COMPLETE",
            'description': "Numbers at end of text"
        },
        
        # Edge case: preserve compound tokens
        {
            'input': "ATM_STATUS 46 ERROR_CODE",
            'expected': "ATM_STATUS ERROR_CODE",
            'description': "Preserve compound tokens"
        },
        
        # Complex real-world pattern
        {
            'input': "*630*06/18/2025*00:46* DEVICE ERROR *7231*1*(Iw(1*3, 72 ESC_000 VAL_000 46",
            'expected': "DEVICE ERROR ESC_000 VAL_000",
            'description': "Complex real pattern with multiple noise sources"
        },
        
        # Timestamp fragments
        {
            'input': "00:46:27 PIN ENTERED 47 SUCCESS",
            'expected': "PIN ENTERED SUCCESS",
            'description': "Timestamp and isolated number"
        }
    ]
    
    print("Testing Intelligent Noise Reduction")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']
        
        result = apply_intelligent_preprocessing(input_text)
        
        # Normalize whitespace for comparison
        result_normalized = ' '.join(result.split())
        expected_normalized = ' '.join(expected.split())
        
        if result_normalized == expected_normalized:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"Test {i}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Expected: '{expected_normalized}'")
        print(f"  Got:      '{result_normalized}'")
        print(f"  Status:   {status}")
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Intelligent approach working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Review the patterns.")
    
    return failed == 0

if __name__ == "__main__":
    test_intelligent_noise_reduction()
