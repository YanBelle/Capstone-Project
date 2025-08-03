#!/usr/bin/env python3
"""
Debug ATR pattern processing step by step
"""

import re

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

text = ej_sample

print("STEP BY STEP DEBUG:")
print("=" * 60)

print("Original:")
print(repr(text))
print("\n" + "=" * 40)

# 1. Remove EJ header patterns: [020t*629*06/18/2025*00:46*
text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
print("After step 1a (remove [020t*...):")
print(repr(text))

text = re.sub(r'\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
print("After step 1b (remove *date* patterns):")
print(repr(text))

# 2. Remove remaining [020t patterns
text = re.sub(r'\[020t\s+', '', text)
print("After step 2 (remove [020t patterns):")
print(repr(text))

# 3. Remove timestamps
print("Before timestamp removal, checking for ATR:")
if "ATR RECEIVED T=0" in text:
    print("✅ ATR RECEIVED T=0 found!")
else:
    print("❌ ATR RECEIVED T=0 NOT found")
    if "ATR RECEIVED" in text:
        print("But ATR RECEIVED is present")

text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
print("After step 3a (remove hh:mm:ss):")
print(repr(text))

print("After timestamp removal, checking for ATR:")
if "ATR RECEIVED T=0" in text:
    print("✅ ATR RECEIVED T=0 still found!")
elif "ATR RECEIVED T=" in text:
    print("⚠️  ATR RECEIVED T= found (missing 0)")
elif "ATR RECEIVED" in text:
    print("⚠️  ATR RECEIVED found but T= part may be missing")
else:
    print("❌ ATR RECEIVED NOT found at all")

text = re.sub(r'\s*\d{2}:\d{2}\s+', ' ', text)
print("After step 3b (remove hh:mm):")
print(repr(text))

# Now apply ATR pattern
print("\nTesting ATR patterns:")
original_text = text

# Test pattern 1
test1 = re.sub(r'\bATR\s+RECEIVED\s+T=(\d+)\b', r'ATR_RECEIVED_T_\1', text)
print("Pattern 1 result:", test1 == text and "NO CHANGE" or "CHANGED")
if test1 != text:
    print("Changed to:", repr(test1))

# Test pattern 2 - more flexible
test2 = re.sub(r'\bATR\s+RECEIVED\s+T=(\d*)\b', r'ATR_RECEIVED_T_\1', text)
print("Pattern 2 result:", test2 == text and "NO CHANGE" or "CHANGED")
if test2 != text:
    print("Changed to:", repr(test2))

# Test pattern 3 - handle missing digit
test3 = re.sub(r'\bATR\s+RECEIVED\s+T=', 'ATR_RECEIVED_T_0', text)
print("Pattern 3 result:", test3 == text and "NO CHANGE" or "CHANGED")
if test3 != text:
    print("Changed to:", repr(test3))

print("\nFinal check - what's in the text around ATR?")
lines = text.split('\n')
for i, line in enumerate(lines):
    if 'ATR' in line:
        print(f"Line {i}: {repr(line)}")
