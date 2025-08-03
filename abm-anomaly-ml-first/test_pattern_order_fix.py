#!/usr/bin/env python3
"""
Test pattern execution order fix for ESC/VAL/REF and REJECTS patterns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'anomaly-detector'))

from bertviz_analyzer import BertVizAnalyzer

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
            'unwanted_patterns': ['ESC', 'VAL', 'REF', ': 000']
        },
        
        # Sample with REJECTS pattern that was creating isolated "1" tokens
        {
            'name': 'REJECTS Pattern',
            'text': 'Transaction REJECTS:000*(1\nS COMPLETED successfully',
            'expected_patterns': ['REJECTS_000'],
            'unwanted_patterns': ['REJECTS:000*(1', '1', 'S']
        },
        
        # Complex sample combining both issues
        {
            'name': 'Combined ESC/VAL/REF + REJECTS',
            'text': 'CARD ESC: 000 VAL: 000 failed REJECTS:000*(1\nS retry',
            'expected_patterns': ['ESC_000', 'VAL_000', 'REJECTS_000'],
            'unwanted_patterns': ['ESC', 'VAL', ': 000', 'REJECTS:000*(1', '1', 'S']
        },
        
        # Sample from your EJ log to test real-world scenario
        {
            'name': 'Real EJ Sample',
            'text': '[020t*629*06/18/2025*00:46* CARD ESC: 000 STEP 1 VAL: 000 TRACK 1 DATA REF: 000 *1*1*(T=1,REJECTS:000*(1\nS',
            'expected_patterns': ['ESC_000', 'VAL_000', 'REF_000', 'REJECTS_000', 'STEP_1', 'TRACK_DATA', 'T_1'],
            'unwanted_patterns': ['ESC', 'VAL', 'REF', ': 000', 'REJECTS:000*(1', '1\\nS']
        }
    ]
    
    # Initialize analyzer
    analyzer = BertVizAnalyzer()
    
    all_tests_passed = True
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n🧪 Test {i}: {sample['name']}")
        print(f"Input: {sample['text']}")
        
        # Preprocess the text
        preprocessed = analyzer._preprocess_text(sample['text'])
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
    success = test_pattern_order_fix()
    sys.exit(0 if success else 1)
