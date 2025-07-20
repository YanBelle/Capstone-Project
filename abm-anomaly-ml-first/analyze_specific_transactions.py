#!/usr/bin/env python3

"""
Test script to analyze specific transactions that should be anomalies
"""

import re
import json
from datetime import datetime

def analyze_transaction(txn_text, txn_name):
    """Analyze a specific transaction for anomaly patterns"""
    print(f"\n=== Analyzing {txn_name} ===")
    print(f"Transaction text:\n{txn_text}")
    
    # Anomaly indicators
    anomaly_indicators = {
        'no_dispense_success': False,
        'no_clear_completion': False,
        'short_session': False,
        'cryptic_codes': False,
        'card_taken_quickly': False,
        'pin_entered_no_result': False,
        'transaction_abandoned': False
    }
    
    # Check for various anomaly patterns
    lines = txn_text.split('\n')
    
    # Look for transaction flow
    has_transaction_start = any('TRANSACTION START' in line for line in lines)
    has_transaction_end = any('TRANSACTION END' in line for line in lines)
    has_dispense_success = any('DISPENSE SUCCESS' in line.upper() for line in lines)
    has_withdrawal = any('WITHDRAWAL' in line.upper() for line in lines)
    has_pin_entered = any('PIN ENTERED' in line for line in lines)
    has_card_taken = any('CARD TAKEN' in line for line in lines)
    
    # Check for cryptic codes
    cryptic_pattern = re.compile(r'\[000p\[040q\(I|\[020t')
    has_cryptic = any(cryptic_pattern.search(line) for line in lines)
    
    # Check timing (if available)
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}|\d{2}:\d{2})')
    times = []
    for line in lines:
        match = time_pattern.search(line)
        if match:
            times.append(match.group())
    
    # Analysis
    if has_transaction_start and has_transaction_end:
        print("✓ Transaction has clear start and end")
    else:
        print("⚠️ Transaction missing clear start/end")
        anomaly_indicators['no_clear_completion'] = True
    
    if not has_dispense_success and not has_withdrawal:
        print("⚠️ No dispense success or withdrawal indication")
        anomaly_indicators['no_dispense_success'] = True
    
    if has_pin_entered and not has_dispense_success:
        print("⚠️ PIN entered but no successful transaction")
        anomaly_indicators['pin_entered_no_result'] = True
    
    if has_cryptic:
        print("⚠️ Contains cryptic codes")
        anomaly_indicators['cryptic_codes'] = True
    
    if len(times) >= 2:
        print(f"⚠️ Transaction duration: {times[0]} to {times[-1]}")
        # This is a simple check - in real analysis we'd parse times properly
        if has_card_taken and not has_dispense_success:
            print("⚠️ Card taken quickly without successful transaction")
            anomaly_indicators['card_taken_quickly'] = True
    
    # Calculate anomaly score
    anomaly_score = sum(anomaly_indicators.values()) / len(anomaly_indicators)
    
    print(f"\n🔍 Anomaly Indicators:")
    for indicator, triggered in anomaly_indicators.items():
        status = "✓" if triggered else "○"
        print(f"  {status} {indicator}")
    
    print(f"\n📊 Anomaly Score: {anomaly_score:.2f}")
    
    if anomaly_score > 0.3:
        print("🚨 SHOULD BE FLAGGED AS ANOMALY")
    else:
        print("✅ Appears normal")
    
    return anomaly_score, anomaly_indicators

def main():
    print("=== Transaction Anomaly Analysis ===")
    
    # Transaction 1
    txn1 = """[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
[020t 13:39:56 CARD TAKEN
[000p[040q(I 75561D(10,M-090B0210B9,R-4S
[000p[040q(I 75561D(10,M-00,R-4S
[020t 13:39:56 TRANSACTION END
[020t15806/18/202513:39
PRIMARY CARD READER ACTIVATED"""
    
    # Transaction 2
    txn2 = """[020t*209*06/18/2025*14:23*
      *TRANSACTION START*
 [020t CARD INSERTED
  14:23:03 ATR RECEIVED T=0
 [020t 14:23:06 OPCODE = FI      
 
   PAN 0004263********6687
   ---START OF TRANSACTION---
  
 [020t 14:23:22 PIN ENTERED
 [020t 14:23:36 OPCODE = BC      
 
   PAN 0004263********6687
   ---START OF TRANSACTION---
  
 [020t 14:24:28 CARD TAKEN
 [020t 14:24:29 TRANSACTION END
 [020t*210*06/18/2025*14:24*
      *PRIMARY CARD READER ACTIVATED*"""
    
    # Analyze both transactions
    score1, indicators1 = analyze_transaction(txn1, "Transaction 1")
    score2, indicators2 = analyze_transaction(txn2, "Transaction 2")
    
    print(f"\n=== Summary ===")
    print(f"Transaction 1 Anomaly Score: {score1:.2f}")
    print(f"Transaction 2 Anomaly Score: {score2:.2f}")
    
    print(f"\n=== Why These Should Be Flagged ===")
    print("1. Both transactions show customer interaction (card insertion, PIN entry)")
    print("2. Neither shows successful dispense or withdrawal")
    print("3. Both contain cryptic codes that indicate potential issues")
    print("4. Transaction 1 has very quick card taken time")
    print("5. Transaction 2 has PIN entered but no successful outcome")
    print("6. Both represent failed customer interactions")
    
    print(f"\n=== Recommendations ===")
    print("1. Ensure ML model is trained on similar 'incomplete transaction' patterns")
    print("2. Add specific patterns for PIN entered + no dispense = anomaly")
    print("3. Flag transactions with cryptic codes as potential hardware issues")
    print("4. Consider transaction duration vs. expected outcome")

if __name__ == "__main__":
    main()
