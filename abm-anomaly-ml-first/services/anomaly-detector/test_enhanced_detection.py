#!/usr/bin/env python3
"""Test enhanced detection patterns against provided transaction examples"""

import re
from typing import List, Dict

def detect_incomplete_transactions(session_text: str) -> List[Dict]:
    """Enhanced detection for incomplete transactions"""
    anomalies = []
    text = session_text.upper()
    
    # Pattern 1: Card inserted/taken without PIN (like txn1)
    if (re.search(r'CARD INSERTED', text) and 
        re.search(r'CARD TAKEN', text) and 
        not re.search(r'PIN ENTERED', text) and 
        not re.search(r'OPCODE', text)):
        anomalies.append({
            "type": "incomplete_transaction",
            "pattern": "card_inserted_taken_no_pin",
            "confidence": 0.95,
            "severity": "high",
            "description": "Card inserted and taken without PIN entry or transaction processing"
        })
    
    # Pattern 2: PIN entered but no completion (like txn2)
    if (re.search(r'PIN ENTERED', text) and 
        re.search(r'OPCODE', text) and 
        re.search(r'CARD TAKEN', text) and 
        not any(re.search(pattern, text) for pattern in [
            r'NOTES PRESENTED', r'RECEIPT PRINTED', r'TRANSACTION COMPLETED',
            r'DISPENSE', r'WITHDRAWAL', r'BALANCE'
        ])):
        anomalies.append({
            "type": "incomplete_transaction",
            "pattern": "pin_entered_no_completion",
            "confidence": 0.90,
            "severity": "high",
            "description": "PIN entered and transaction initiated but not completed"
        })
    
    # Pattern 3: Transaction start/end without meaningful activity
    if (re.search(r'TRANSACTION START', text) and 
        re.search(r'TRANSACTION END', text) and
        not any(re.search(pattern, text) for pattern in [
            r'NOTES PRESENTED', r'RECEIPT PRINTED', r'BALANCE INQUIRY',
            r'WITHDRAWAL', r'DEPOSIT', r'TRANSFER'
        ])):
        anomalies.append({
            "type": "incomplete_transaction",
            "pattern": "transaction_no_completion",
            "confidence": 0.85,
            "severity": "medium",
            "description": "Transaction started and ended without meaningful activity"
        })
    
    return anomalies

# Test with the provided examples
test_txn1 = """[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
[020t 13:39:56 CARD TAKEN
[000p[040q(I 75561D(10,M-090B0210B9,R-4S
[000p[040q(I 75561D(10,M-00,R-4S
[020t 13:39:56 TRANSACTION END
[020t15806/18/202513:39
PRIMARY CARD READER ACTIVATED"""

test_txn2 = """[020t*209*06/18/2025*14:23*
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

if __name__ == "__main__":
    print("=== Enhanced Detection Test ===")
    print("\nTesting Transaction 1:")
    txn1_anomalies = detect_incomplete_transactions(test_txn1)
    for anomaly in txn1_anomalies:
        print(f"  ✓ {anomaly['type']}: {anomaly['description']} (confidence: {anomaly['confidence']})")
    
    print("\nTesting Transaction 2:")
    txn2_anomalies = detect_incomplete_transactions(test_txn2)
    for anomaly in txn2_anomalies:
        print(f"  ✓ {anomaly['type']}: {anomaly['description']} (confidence: {anomaly['confidence']})")
    
    if not txn1_anomalies:
        print("  ❌ No anomalies detected for Transaction 1")
    if not txn2_anomalies:
        print("  ❌ No anomalies detected for Transaction 2")
    
    print(f"\nTotal anomalies detected: {len(txn1_anomalies) + len(txn2_anomalies)}")
