#!/usr/bin/env python3
"""
Lightweight test for NCB EJ preprocessing patterns - no PyTorch dependencies
Tests regex patterns directly to validate preprocessing logic
"""

import re

def test_preprocessing_patterns():
    """Test the preprocessing patterns against NCB EJ sample"""
    
    # NCB EJ sample with the issues mentioned
    ncb_ej_sample = """[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
 04:48:38 ATR RECEIVED T=0
[020t 04:48:40 OPCODE = FI      

  PAN 0004263********2113
  ---START OF TRANSACTION---
 
[020t 04:48:55 PIN ENTERED
[020t 04:49:01 OPCODE = BBC     
 04:49:02 GENAC 1 : ARQC
 04:49:04 GENAC 2 : TC
[020t 04:49:11 NOTES STACKED
[020t 04:49:13 CARD TAKEN

  PAN 0004263********2113
  ---START OF TRANSACTION---
 
[020t 04:49:18 NOTES PRESENTED 1,0,0,0
[020t 04:49:18 NOTES TAKEN
[020t
CASH TOTAL       TYPE1 TYPE2 TYPE3 TYPE4
DENOMINATION      1000  2000  5000  5000
DISPENSED        00271 00243 00621 00540
REJECTED         00003 00001 00010 00003
REMAINING        01729 01757 01379 01460


       N.C.B. MIDAS
   NCB DUKE ST. BRANCH
     DATE        TIME
   2025/06/18   04:49:03
   SAV
   MACHINE       0250
   TRAN NO       227233
   AUTHORIZATION 044933
   ************2113
   WITHDRAWAL     1000.00
   ACCOUNT        7372.73
   FROM SAVINGS
         THANK YOU
[020t 04:49:30 TRANSACTION END
[020t*633*06/18/2025*04:49*
     *PRIMARY CARD READER ACTIVATED*"""

    print("🧪 Testing NCB EJ Preprocessing Patterns (Lightweight)")
    print("=" * 50)
    
    # Apply the preprocessing patterns step by step
    text = ncb_ej_sample
    
    print("📋 ORIGINAL TEXT LENGTH:", len(text))
    
    # Step 1: Replace Cash Dispensing Summary
    cash_summary_pattern = r'CASH\s+TOTAL\s+TYPE\d+.*?REMAINING\s+\d+(?:\s+\d+)*'
    text = re.sub(cash_summary_pattern, 'CASH_DISPENSED_SUMMARY', text, flags=re.DOTALL)
    print("✅ Applied cash summary pattern")
    
    # Step 2: Replace NCB Receipt
    receipt_pattern1 = r'N\.C\.B\.\s+MIDAS\s+NCB\s+[A-Z\s\.]+BRANCH.*?THANK YOU'
    before_receipt = text
    text = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', text, flags=re.DOTALL)
    receipt_replaced = len(before_receipt) != len(text)
    print(f"✅ Applied NCB receipt pattern - {'REPLACED' if receipt_replaced else 'NO MATCH'}")
    
    # Step 3: Handle NOTES patterns
    text = re.sub(r'\*PRIMARY CARD READER ACTIVATED\*', 'PRIMARY_CARD_READER_ACTIVATED', text)
    text = re.sub(r'\bNOTES\s+PRESENTED\s+[\d,\s]+', 'NOTES_PRESENTED', text)
    text = re.sub(r'\bNOTES\s+STACKED\b', 'NOTES_STACKED', text)
    text = re.sub(r'\bNOTES\s+TAKEN\b', 'NOTES_TAKEN', text)
    text = re.sub(r'\bOPCODE\s*=\s*(BBC)\b', r'OPCODE_\1', text)
    print("✅ Applied NOTES and OPCODE patterns")
    
    # Step 4: General cleanup
    text = re.sub(r'\[020t\*\d+\*\d{2}/\d{2}/\d{4}\*\d{2}:\d{2}\*', '', text)
    text = re.sub(r'\*TRANSACTION START\*', 'TRANSACTION_START', text)
    text = re.sub(r'PAN\s+\d{4}\d+\*+\d+', 'CardNumber', text)
    text = re.sub(r'\s*---START OF TRANSACTION---\s*', ' ', text)
    text = re.sub(r'\s*\d{2}:\d{2}:\d{2}\s+', ' ', text)
    text = ' '.join(text.split())  # Clean whitespace
    print("✅ Applied general cleanup patterns")
    
    print("\n" + "="*50)
    print("✨ PROCESSED TEXT:")
    print(text)
    print("\n" + "="*50)
    
    # Validation tests
    tests = [
        {
            'name': 'Receipt Section Replacement',
            'check': 'RECEIPT_PRINTED' in text,
            'fail_check': any(word in text.upper() for word in ['AUTHORIZATION', 'BRANCH', 'WITHDRAWAL', 'DUKE ST']),
            'description': 'NCB receipt should be replaced, removing noise words'
        },
        {
            'name': 'PRIMARY CARD READER ACTIVATED',
            'check': 'PRIMARY_CARD_READER_ACTIVATED' in text,
            'fail_check': '*PRIMARY CARD READER ACTIVATED*' in text,
            'description': 'Asterisks should be removed'
        },
        {
            'name': 'NOTES STACKED',
            'check': 'NOTES_STACKED' in text,
            'fail_check': 'NOTES STACKED' in text,
            'description': 'Should be compound token'
        },
        {
            'name': 'NOTES PRESENTED',
            'check': 'NOTES_PRESENTED' in text and '1,0,0,0' not in text,
            'fail_check': 'NOTES PRESENTED 1,0,0,0' in text,
            'description': 'Comma-separated numbers should be removed'
        },
        {
            'name': 'NOTES TAKEN',
            'check': 'NOTES_TAKEN' in text,
            'fail_check': 'NOTES TAKEN' in text,
            'description': 'Should be compound token'
        },
        {
            'name': 'Cash Dispensing Summary',
            'check': 'CASH_DISPENSED_SUMMARY' in text,
            'fail_check': 'TYPE1 TYPE2 TYPE3 TYPE4' in text,
            'description': 'Cash table should be replaced'
        },
        {
            'name': 'OPCODE BBC',
            'check': 'OPCODE_BBC' in text,
            'fail_check': 'OPCODE = BBC' in text,
            'description': 'Should be compound token'
        }
    ]
    
    print("🔍 PATTERN VALIDATION RESULTS:")
    print("-" * 50)
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        status = "✅ PASS" if test['check'] and not test['fail_check'] else "❌ FAIL"
        if test['check'] and not test['fail_check']:
            passed += 1
        
        print(f"{status} {test['name']}")
        print(f"   📝 {test['description']}")
        
        if not test['check']:
            print(f"   ⚠️  Missing expected pattern")
        if test['fail_check']:
            print(f"   ⚠️  Found unwanted noise pattern")
        print()
    
    print(f"🎯 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All NCB EJ preprocessing patterns working correctly!")
        return True
    else:
        print("⚠️  Some patterns need adjustment")
        
        # Debug failing patterns
        print("\n🔍 DEBUG INFO:")
        print(f"Text contains 'RECEIPT_PRINTED': {'RECEIPT_PRINTED' in text}")
        print(f"Text contains 'AUTHORIZATION': {'AUTHORIZATION' in text}")
        print(f"Text contains 'BRANCH': {'BRANCH' in text}")
        print(f"Text contains 'N.C.B. MIDAS': {'N.C.B. MIDAS' in text}")
        
        return False

if __name__ == "__main__":
    print("🚀 NCB EJ Pattern Testing Suite (Lightweight)")
    print("=" * 60)
    
    success = test_preprocessing_patterns()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Ready for BERT integration!")
    else:
        print("⚠️  TESTS FAILED - Pattern adjustments needed")
