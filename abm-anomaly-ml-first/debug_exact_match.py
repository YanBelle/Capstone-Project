#!/usr/bin/env python3
"""
Debug exactly what pattern 1 produces for the first test case
"""
import re

# The failing test case
input_text = """ESC/VAL 000 16:07:37
1 16:07:37 > TRANSACTION_START
2 16:07:38 > WITHDRAWAL_500
ESC/VAL 001 16:07:38
1 16:07:38 > Your transaction receipt
    FIRST NATIONAL BANK
    ATM#: 12345
    DATE: 06/18/2025
    TIME: 16:07:38
    MACHINE: ABM250EJ
    
    WITHDRAWAL
    ACCOUNT: ****1234
    AMOUNT: $500.00
    
    THANK YOU
2 16:07:40 > TRANSACTION_END"""

expected = """ESC/VAL 000 16:07:37
1 16:07:37 > TRANSACTION_START
2 16:07:38 > WITHDRAWAL_500
ESC/VAL 001 16:07:38
1 16:07:38 > Your transaction receipt
 RECEIPT_PRINTED 
2 16:07:40 > TRANSACTION_END"""

# Pattern 1 from our bertviz_analyzer.py  
receipt_pattern1 = r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'

result = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)

print("=== Detailed Pattern 1 Analysis ===")
print(f"Input length: {len(input_text)}")
print(f"Expected length: {len(expected)}")
print(f"Result length: {len(result)}")
print()

print("Expected:")
print(repr(expected))
print()

print("Got:")
print(repr(result))
print()

print("Differences:")
if result == expected:
    print("✅ EXACT MATCH!")
else:
    print("❌ MISMATCH")
    
    # Character by character comparison
    for i, (c1, c2) in enumerate(zip(expected, result)):
        if c1 != c2:
            print(f"First difference at position {i}: expected {repr(c1)}, got {repr(c2)}")
            print(f"Context: ...{expected[max(0, i-10):i+10]}...")
            print(f"         ...{result[max(0, i-10):i+10]}...")
            break
    
    # Check if one is longer than the other
    if len(expected) != len(result):
        print(f"Length difference: expected {len(expected)}, got {len(result)}")
        if len(result) > len(expected):
            print(f"Extra content: {repr(result[len(expected):])}")
        else:
            print(f"Missing content: {repr(expected[len(result):])}")

# Let's also see what the pattern actually captures
match = re.search(receipt_pattern1, input_text, flags=re.DOTALL)
if match:
    print(f"\nPattern captured: {repr(match.group(1))}")
else:
    print("\nPattern did not match!")
