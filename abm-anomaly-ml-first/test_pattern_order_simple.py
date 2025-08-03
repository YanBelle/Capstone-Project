#!/usr/bin/env python3
"""
Simple test for pattern execution order fix - testing preprocessing only
"""

import re

def preprocess_text_test(text: str) -> str:
    """Test version of the preprocessing function with optimized pattern execution order"""
    
    # CRITICAL FIRST: Handle ESC/VAL/REF patterns BEFORE any other cleanup removes the values
    # Convert VAL: 000, ESC: 000, REF: 000 patterns to compound tokens
    text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
    # Handle cases like "ESC 000" (without colon), "VAL   000" (multiple spaces)
    text = re.sub(r'\b(VAL|ESC|REF)\s+(\d+)\b', r'\1_\2', text)
    
    # CRITICAL SECOND: Handle ATR pattern IMMEDIATELY after ESC/VAL/REF
    text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
    
    # CRITICAL THIRD: Handle REJECTS patterns early to prevent "1" token isolation
    # Clean up "REJECTS:000*(1" patterns that create isolated "1" tokens
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS\s+(\d+)', r'REJECTS_\1', text)
    
    # Remove EJ header patterns: [020t*629*06/18/2025*00:46*
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Handle other transaction elements
    text = re.sub(r'\bTRACK\s+\d+\s+DATA\b', 'TRACK_DATA', text)
    text = re.sub(r'\bT=(\d+)\b', r'T_\1', text)
    text = re.sub(r'\bSTEP\s+(\d+)\b', r'STEP_\1', text)
    
    # ENHANCED: Aggressively remove isolated numeric fragments
    # Protect meaningful patterns first
    text = re.sub(r'(STEP)_(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(T)_(\d+)', r'PROTECTED_\1_\2', text)
    
    # Remove ALL isolated single digits
    text = re.sub(r'\b\d\b', '', text)
    
    # Restore protected patterns
    text = re.sub(r'PROTECTED_(STEP|T)_(\d+)', r'\1_\2', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    
    return text.strip()

def test_pattern_order_fix():
    """Test that pattern execution order correctly handles ESC/VAL/REF and REJECTS patterns"""
    
    print("🔍 Testing Pattern Execution Order Fix...")
    print("=" * 60)
    
    # Test samples with specific patterns that were causing issues
    test_samples = [
        # Sample with ESC/VAL/REF patterns that should become compound tokens
        {
            'name': 'ESC/VAL/REF Patterns',
            'text': 'CARD ESC: 000 VAL: 000 REF: 000 processing complete',
            'expected_patterns': ['ESC_000', 'VAL_000', 'REF_000'],
            'unwanted_patterns': ['ESC: 000', 'VAL: 000', 'REF: 000']
        },
        
        # Sample with REJECTS pattern that was creating isolated "1" tokens
        {
            'name': 'REJECTS Pattern',
            'text': 'Transaction REJECTS:000*(1 COMPLETED successfully',
            'expected_patterns': ['REJECTS_000'],
            'unwanted_patterns': ['REJECTS:000*(1', 'REJECTS:000*']
        },
        
        # Complex sample combining both issues
        {
            'name': 'Combined ESC/VAL/REF + REJECTS',
            'text': 'CARD ESC: 000 VAL: 000 failed REJECTS:000*(1 retry',
            'expected_patterns': ['ESC_000', 'VAL_000', 'REJECTS_000'],
            'unwanted_patterns': ['ESC: 000', 'VAL: 000', 'REJECTS:000*(1']
        },
        
        # Sample from your EJ log to test real-world scenario
        {
            'name': 'Real EJ Sample',
            'text': '[020t*629*06/18/2025*00:46* CARD ESC: 000 STEP 1 VAL: 000 TRACK 1 DATA REF: 000 *1*1*(T=1,REJECTS:000*(1',
            'expected_patterns': ['ESC_000', 'VAL_000', 'REF_000', 'REJECTS_000', 'STEP_1', 'TRACK_DATA', 'T_1'],
            'unwanted_patterns': ['ESC: 000', 'VAL: 000', 'REF: 000', 'REJECTS:000*(1', '[020t*629*06/18/2025*00:46*', ' 1 ']
        },
        
        # Specific test for isolated "1" token elimination
        {
            'name': 'Isolated Digit Elimination',
            'text': 'CARD ESC: 000 STEP 1 VAL: 000 some 1 noise 3 TRACK 1 DATA and 7 fragments',
            'expected_patterns': ['ESC_000', 'VAL_000', 'STEP_1', 'TRACK_DATA'],
            'unwanted_patterns': [' 1 ', ' 3 ', ' 7 ', 'ESC: 000', 'VAL: 000']
        }
    ]
    
    all_tests_passed = True
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n🧪 Test {i}: {sample['name']}")
        print(f"Input: {sample['text']}")
        
        # Preprocess the text
        preprocessed = preprocess_text_test(sample['text'])
        print(f"Output: {preprocessed}")
        
        # Check for expected patterns
        expected_found = []
        for pattern in sample['expected_patterns']:
            if pattern in preprocessed:
                expected_found.append(pattern)
                print(f"  ✅ Found expected pattern: {pattern}")
            else:
                print(f"  ❌ Missing expected pattern: {pattern}")
                all_tests_passed = False
        
        # Check for unwanted patterns
        unwanted_found = []
        for pattern in sample['unwanted_patterns']:
            if pattern in preprocessed:
                unwanted_found.append(pattern)
                print(f"  ❌ Found unwanted pattern: {pattern}")
                all_tests_passed = False
            else:
                print(f"  ✅ Successfully removed: {pattern}")
        
        print(f"  📊 Expected found: {len(expected_found)}/{len(sample['expected_patterns'])}")
        print(f"  📊 Unwanted avoided: {len(sample['unwanted_patterns']) - len(unwanted_found)}/{len(sample['unwanted_patterns'])}")
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL PATTERN ORDER TESTS PASSED!")
        print("✅ ESC/VAL/REF patterns are properly converted to compound tokens")
        print("✅ REJECTS patterns no longer create isolated '1' tokens")
        print("✅ Pattern execution order is optimized")
    else:
        print("❌ SOME PATTERN ORDER TESTS FAILED")
        print("⚠️  Pattern execution order may need further optimization")
    
    return all_tests_passed

if __name__ == '__main__':
    import sys
    success = test_pattern_order_fix()
    sys.exit(0 if success else 1)
