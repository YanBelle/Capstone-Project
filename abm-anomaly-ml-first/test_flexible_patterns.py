#!/usr/bin/env python3
"""
Test flexible Machine and R status pattern recognition
"""

import re

def test_flexible_patterns():
    """Test the updated flexible regex patterns for Machine and R status"""
    
    # Test cases with various Machine and R status patterns
    test_cases = [
        # Original patterns
        ("M-02, R-10011", "M_02 R_10011"),
        ("M-02,R-10011", "M_02 R_10011"),
        
        # Various Machine status patterns
        ("M-00", "M_00"),
        ("M-01", "M_01"),
        ("M-15", "M_15"),
        ("M-20", "M_20"),
        ("M-99", "M_99"),
        
        # Various R status patterns
        ("R-0000", "R_0000"),
        ("R-5005", "R_5005"),
        ("R-20001", "R_20001"),
        ("R-30015", "R_30015"),
        ("R-40000", "R_40000"),
        ("R-50000", "R_50000"),
        
        # Combined patterns in context
        ("Device status M-15, Error code R-5005", "Device status M_15 Error code R_5005"),
        ("Transaction failed M-99 R-20001", "Transaction failed M_99 R_20001"),
        ("Status: M-05, Reference: R-30015", "Status: M_05 Reference: R_30015"),
        
        # Edge cases
        ("M-123", "M_123"),  # 3-digit machine status
        ("R-99999", "R_99999"),  # 5-digit R status
        ("M-7", "M_7"),  # Single digit
        
        # Mixed with other content
        ("Card inserted M-02, processing R-10011 complete", "Card inserted M_02 processing R_10011 complete"),
        
        # Multiple occurrences
        ("M-01 initial, M-02 processing, M-99 error R-5005", "M_01 initial, M_02 processing, M_99 error R_5005"),
    ]
    
    def apply_patterns(text):
        """Apply the same patterns as in bertviz_analyzer.py"""
        # Machine status: M-02, M-15, etc. -> M_02, M_15, etc.
        text = re.sub(r'\bM-(\d+),?\s*', r'M_\1 ', text)
        # R status: R-10011, R-5005, etc. -> R_10011, R_5005, etc.
        text = re.sub(r'\bR-(\d+)\b', r'R_\1', text)
        # Clean up extra spaces
        text = ' '.join(text.split())
        return text
    
    print("Testing Flexible Machine and R Status Pattern Recognition")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = apply_patterns(input_text)
        
        if result == expected:
            print(f"✅ Test {i:2d}: PASSED")
            print(f"   Input:    '{input_text}'")
            print(f"   Output:   '{result}'")
            passed += 1
        else:
            print(f"❌ Test {i:2d}: FAILED")
            print(f"   Input:    '{input_text}'")
            print(f"   Expected: '{expected}'")
            print(f"   Got:      '{result}'")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Flexible patterns are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the regex patterns.")
    
    return failed == 0

if __name__ == "__main__":
    test_flexible_patterns()
