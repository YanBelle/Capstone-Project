#!/usr/bin/env python3
"""
Test the isolated digit fix to verify that the "1" token is eliminated
"""

import re

def enhanced_preprocess_text(text: str) -> str:
    """Enhanced preprocessing with aggressive isolated digit removal"""
    
    # CRITICAL FIRST: Handle ESC/VAL/REF patterns BEFORE any other cleanup removes the values
    text = re.sub(r'\b(VAL|ESC|REF):\s*(\d+)\b', r'\1_\2', text)
    text = re.sub(r'\b(VAL|ESC|REF)\s+(\d+)\b', r'\1_\2', text)
    
    # CRITICAL SECOND: Handle ATR pattern IMMEDIATELY after ESC/VAL/REF
    text = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
    
    # CRITICAL THIRD: Handle REJECTS patterns early to prevent "1" token isolation
    text = re.sub(r'REJECTS:(\d+)\*\([^)]*\)', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS:(\d+)', r'REJECTS_\1', text)
    text = re.sub(r'REJECTS\s+(\d+)', r'REJECTS_\1', text)
    
    # Remove EJ header patterns
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    
    # Handle other transaction elements
    text = re.sub(r'\bTRACK\s+\d+\s+DATA\b', 'TRACK_DATA', text)
    text = re.sub(r'\bT=(\d+)\b', r'T_\1', text)
    text = re.sub(r'\bSTEP\s+(\d+)\b', r'STEP_\1', text)
    
    # ENHANCED: Aggressively remove isolated numeric fragments
    # Protect meaningful patterns first
    text = re.sub(r'(AMOUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(COUNT)\s+(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(TOTAL)\s+(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(BALANCE)\s+(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(STEP)_(\d+)', r'PROTECTED_\1_\2', text)
    text = re.sub(r'(T)_(\d+)', r'PROTECTED_\1_\2', text)
    
    # AGGRESSIVE: Remove ALL isolated single digits that appear between words or at boundaries
    text = re.sub(r'\b\d\b', '', text)  # Remove any single isolated digit
    
    # Also remove isolated multi-digit fragments that are likely noise (2-4 digits)
    text = re.sub(r'\b\d{2,4}\b(?=\s+(?:[A-Z][A-Z_]+|[a-z]+)|\s*$)', '', text)
    
    # Restore protected meaningful numbers
    text = re.sub(r'PROTECTED_(AMOUNT|COUNT|TOTAL|BALANCE|STEP|T)_(\d+)', r'\1_\2', text)
    
    # Clean up spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def test_isolated_digit_elimination():
    """Test cases specifically for isolated digit elimination"""
    
    print("🧪 Testing Isolated Digit Elimination Fix")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Original Heatmap Sample',
            'input': 'CARD ESC: 000 STEP 1 VAL: 000 TRACK 1 DATA REF: 000 REJECTS:000*(1',
            'should_not_contain': ['1'],  # Should NOT contain isolated "1"
            'should_contain': ['ESC_000', 'VAL_000', 'REF_000', 'STEP_1', 'TRACK_DATA', 'REJECTS_000']
        },
        {
            'name': 'Multiple Isolated Digits',
            'input': 'CARD ESC: 000 some 1 noise 3 TRACK 1 DATA and 7 fragments',
            'should_not_contain': ['1', '3', '7'],  # Should NOT contain isolated digits
            'should_contain': ['ESC_000', 'TRACK_DATA']
        },
        {
            'name': 'Complex EJ Pattern',
            'input': '[020t*629*06/18/2025*00:46* CARD ESC: 000 STEP 1 VAL: 000 *1*1*(T=1,REJECTS:000*(1',
            'should_not_contain': ['1', '[020t*629*06/18/2025*00:46*'],
            'should_contain': ['ESC_000', 'VAL_000', 'STEP_1', 'T_1', 'REJECTS_000']
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['name']}")
        print(f"Input:  {test['input']}")
        
        result = enhanced_preprocess_text(test['input'])
        print(f"Output: {result}")
        
        # Check that unwanted patterns are removed
        unwanted_found = []
        for pattern in test['should_not_contain']:
            if f' {pattern} ' in f' {result} ':  # Check for isolated occurrence
                unwanted_found.append(pattern)
                print(f"  ❌ Found unwanted isolated pattern: '{pattern}'")
                all_passed = False
            else:
                print(f"  ✅ Successfully removed isolated: '{pattern}'")
        
        # Check that expected patterns exist
        missing_expected = []
        for pattern in test['should_contain']:
            if pattern in result:
                print(f"  ✅ Found expected pattern: '{pattern}'")
            else:
                missing_expected.append(pattern)
                print(f"  ❌ Missing expected pattern: '{pattern}'")
                all_passed = False
        
        print(f"  📊 Result: {len(test['should_contain']) - len(missing_expected)}/{len(test['should_contain'])} expected found, {len(test['should_not_contain']) - len(unwanted_found)}/{len(test['should_not_contain'])} unwanted removed")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL ISOLATED DIGIT ELIMINATION TESTS PASSED!")
        print("✅ The isolated '1' token issue has been resolved!")
    else:
        print("❌ SOME TESTS FAILED - May need further adjustment")
    
    return all_passed

if __name__ == '__main__':
    import sys
    success = test_isolated_digit_elimination()
    sys.exit(0 if success else 1)
