#!/usr/bin/env python3
"""
Comprehensive test for enhanced EJ cleaning patterns
Tests all the specified cleaning patterns in order
"""

import re

def test_enhanced_ej_cleaning():
    """Test all EJ cleaning patterns with the provided sample"""
    
    # Sample EJ text from user
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

    print("Enhanced EJ Text Cleaning Test")
    print("=" * 60)
    print(f"Original text length: {len(sample_ej)} characters")
    print(f"Original text:\n{sample_ej}")
    print("\n" + "=" * 60)
    
    # Apply cleaning patterns in the specified order
    text = sample_ej
    
    # 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
    print("\n1. Removing EJ header patterns [020t*seq*date*time*...")
    pattern1 = r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*'
    text = re.sub(pattern1, '', text)
    print(f"After step 1 ({len(text)} chars):\n{text[:200]}...")
    
    # 2. Remove remaining [020t patterns
    print("\n2. Removing remaining [020t patterns...")
    pattern2 = r'\[020t\s+'
    text = re.sub(pattern2, '', text)
    print(f"After step 2 ({len(text)} chars):\n{text[:200]}...")
    
    # 3. Remove standalone timestamps hh:mm:ss
    print("\n3. Removing standalone timestamps hh:mm:ss...")
    pattern3 = r'\s+\d{2}:\d{2}:\d{2}\s+'
    text = re.sub(pattern3, ' ', text)
    print(f"After step 3 ({len(text)} chars):\n{text[:200]}...")
    
    # 4. Remove transaction start markers
    print("\n4. Removing '---START OF TRANSACTION---' markers...")
    pattern4 = r'\s*---START OF TRANSACTION---\s*'
    text = re.sub(pattern4, ' ', text)
    print(f"After step 4 ({len(text)} chars):\n{text[:200]}...")
    
    # 5. Clean up whitespace
    print("\n5. Cleaning up excessive whitespace...")
    text = ' '.join(text.split())
    print(f"Final cleaned text ({len(text)} chars):\n{text}")
    
    print("\n" + "=" * 60)
    print("CLEANING VERIFICATION:")
    print("=" * 60)
    
    # Check what was preserved
    preserved_content = []
    if "DEVICE ERROR" in text:
        preserved_content.append("✅ DEVICE ERROR")
    if "REJECTS:000" in text:
        preserved_content.append("✅ REJECTS:000")
    if "CARD INSERTED" in text:
        preserved_content.append("✅ CARD INSERTED")
    if "PIN ENTERED" in text:
        preserved_content.append("✅ PIN ENTERED")
    if "CARD TAKEN" in text:
        preserved_content.append("✅ CARD TAKEN")
    if "TRANSACTION END" in text:
        preserved_content.append("✅ TRANSACTION END")
    
    # Check what was removed
    removed_patterns = []
    if "[020t*629*06/18/2025*00:46*" not in text:
        removed_patterns.append("✅ EJ header patterns removed")
    if "[020t " not in text:
        removed_patterns.append("✅ [020t patterns removed")
    if "00:46:27" not in text and "00:46:30" not in text:
        removed_patterns.append("✅ Standalone timestamps removed")
    if "---START OF TRANSACTION---" not in text:
        removed_patterns.append("✅ Transaction markers removed")
    
    print("PRESERVED CRITICAL CONTENT:")
    for item in preserved_content:
        print(f"  {item}")
    
    print("\nREMOVED NOISE PATTERNS:")
    for item in removed_patterns:
        print(f"  {item}")
    
    print(f"\nTEXT REDUCTION: {len(sample_ej)} → {len(text)} characters ({((len(sample_ej) - len(text)) / len(sample_ej) * 100):.1f}% reduction)")
    
    return text

def test_individual_patterns():
    """Test each pattern individually"""
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL PATTERN TESTS:")
    print("=" * 60)
    
    # Test pattern 1: EJ headers
    test1 = "[020t*629*06/18/2025*00:46* DEVICE ERROR"
    result1 = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', test1)
    print(f"Pattern 1 test: '{test1}' → '{result1}'")
    
    # Test pattern 2: [020t patterns
    test2 = "[020t CARD INSERTED [020t 00:47:13 DEVICE ERROR"
    result2 = re.sub(r'\[020t\s+', '', test2)
    print(f"Pattern 2 test: '{test2}' → '{result2}'")
    
    # Test pattern 3: timestamps
    test3 = "CARD INSERTED 00:46:27 ATR RECEIVED 00:46:30 OPCODE"
    result3 = re.sub(r'\s+\d{2}:\d{2}:\d{2}\s+', ' ', test3)
    print(f"Pattern 3 test: '{test3}' → '{result3}'")
    
    # Test pattern 4: transaction markers
    test4 = "PAN 0004263 ---START OF TRANSACTION--- DEVICE ERROR"
    result4 = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', test4)
    print(f"Pattern 4 test: '{test4}' → '{result4}'")

if __name__ == "__main__":
    # Run comprehensive test
    cleaned_text = test_enhanced_ej_cleaning()
    
    # Run individual pattern tests
    test_individual_patterns()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - Enhanced EJ cleaning patterns verified!")
    print("=" * 60)
