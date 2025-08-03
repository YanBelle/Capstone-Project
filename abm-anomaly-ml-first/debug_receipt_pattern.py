#!/usr/bin/env python3
"""
Debug the first receipt test case to understand why it's not matching
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

# Try each pattern individually
patterns = [
    ("Pattern 1", r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'),
    ("Pattern 2", r'(DATE:\s*[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'),
    ("Pattern 3", r'(\s+[A-Z][A-Z\s]+\n\s*(?:ATM|RECEIPT)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'),
    ("Custom", r'(FIRST NATIONAL BANK.*?THANK YOU)')
]

print("Testing patterns on the failing test case:")
print("=" * 50)

for name, pattern in patterns:
    result = re.sub(pattern, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
    if result != input_text:
        print(f"✅ {name}: MATCHED")
        print(f"   Result preview: {result[:100]}...")
    else:
        print(f"❌ {name}: NO MATCH")

# Let's look for what the receipt section actually looks like
print("\nReceipt section analysis:")
print("=" * 30)

# Find where the receipt starts and ends
lines = input_text.split('\n')
receipt_start = -1
receipt_end = -1

for i, line in enumerate(lines):
    if 'FIRST NATIONAL BANK' in line:
        receipt_start = i
    if 'THANK YOU' in line:
        receipt_end = i
        break

if receipt_start >= 0 and receipt_end >= 0:
    receipt_section = '\n'.join(lines[receipt_start:receipt_end+1])
    print(f"Receipt section (lines {receipt_start}-{receipt_end}):")
    print(repr(receipt_section))
    
    # Try a simpler pattern
    simple_pattern = r'(FIRST NATIONAL BANK.*?THANK YOU)'
    simple_result = re.sub(simple_pattern, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
    
    if simple_result != input_text:
        print("\n✅ Simple pattern works!")
        print(f"Result: {simple_result[:200]}...")
    else:
        print("\n❌ Even simple pattern doesn't work")
        
        # Let's try an even more specific pattern
        specific_pattern = r'(\s+FIRST NATIONAL BANK.*?THANK YOU)'
        specific_result = re.sub(specific_pattern, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
        
        if specific_result != input_text:
            print("✅ Specific pattern with whitespace works!")
        else:
            print("❌ Still not working - investigating further...")
            
            # Check what's actually between FIRST NATIONAL BANK and THANK YOU
            import re
            match = re.search(r'FIRST NATIONAL BANK(.*?)THANK YOU', input_text, re.DOTALL)
            if match:
                print(f"Content between FIRST NATIONAL BANK and THANK YOU:")
                print(repr(match.group(1)))
            else:
                print("No match found between FIRST NATIONAL BANK and THANK YOU")
