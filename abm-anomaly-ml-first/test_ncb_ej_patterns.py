#!/usr/bin/env python3
"""
Test script to validate NCB EJ sample preprocessing patterns
Tests the specific issues mentioned:
1. Receipt section not fully replaced (authorization, branch, date, withdrawal noise)
2. *PRIMARY CARD READER ACTIVATED* asterisk removal
3. NOTES patterns with comma-separated numbers  
4. New vocabulary tokens
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'anomaly-detector'))

from bertviz_analyzer import BertVisualizationAnalyzer

def test_ncb_ej_preprocessing():
    """Test preprocessing of the NCB EJ sample"""
    
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

    print("🧪 Testing NCB EJ Preprocessing Patterns")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BertVisualizationAnalyzer()
    
    # Test preprocessing
    processed_text = analyzer._preprocess_text(ncb_ej_sample)
    
    print("📋 ORIGINAL TEXT (first 500 chars):")
    print(ncb_ej_sample[:500] + "..." if len(ncb_ej_sample) > 500 else ncb_ej_sample)
    print("\n" + "="*50)
    
    print("✨ PROCESSED TEXT:")
    print(processed_text)
    print("\n" + "="*50)
    
    # Test specific patterns
    tests = [
        {
            'name': 'Receipt Section Replacement',
            'check': 'RECEIPT_PRINTED' in processed_text,
            'fail_check': any(word in processed_text for word in ['authorization', 'AUTHORIZATION', 'BRANCH', 'WITHDRAWAL', 'DATE']),
            'description': 'NCB receipt should be replaced with RECEIPT_PRINTED, removing authorization/branch/date/withdrawal noise'
        },
        {
            'name': 'PRIMARY CARD READER ACTIVATED',
            'check': 'PRIMARY_CARD_READER_ACTIVATED' in processed_text,
            'fail_check': '*PRIMARY CARD READER ACTIVATED*' in processed_text,
            'description': 'Asterisks should be removed and converted to compound token'
        },
        {
            'name': 'NOTES STACKED',
            'check': 'NOTES_STACKED' in processed_text,
            'fail_check': 'NOTES STACKED' in processed_text,
            'description': 'NOTES STACKED should be converted to compound token'
        },
        {
            'name': 'NOTES PRESENTED',
            'check': 'NOTES_PRESENTED' in processed_text,
            'fail_check': any(pattern in processed_text for pattern in ['NOTES PRESENTED 1,0,0,0', '1,0,0,0', 'NOTES PRESENTED 1']),
            'description': 'NOTES PRESENTED with comma-separated numbers should be cleaned'
        },
        {
            'name': 'NOTES TAKEN',
            'check': 'NOTES_TAKEN' in processed_text,
            'fail_check': 'NOTES TAKEN' in processed_text,
            'description': 'NOTES TAKEN should be converted to compound token'
        },
        {
            'name': 'Cash Dispensing Summary',
            'check': 'CASH_DISPENSED_SUMMARY' in processed_text,
            'fail_check': 'TYPE1 TYPE2 TYPE3 TYPE4' in processed_text,
            'description': 'Cash dispensing table should be replaced with summary'
        },
        {
            'name': 'OPCODE BBC',
            'check': 'OPCODE_BBC' in processed_text,
            'fail_check': 'OPCODE = BBC' in processed_text,
            'description': 'OPCODE = BBC should be converted to compound token'
        },
        {
            'name': 'Transaction Markers',
            'check': 'TRANSACTION_START' in processed_text and 'TRANSACTION_END' in processed_text,
            'fail_check': '*TRANSACTION START*' in processed_text,
            'description': 'Transaction markers should be cleaned compound tokens'
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
        return False

def test_vocabulary_tokens():
    """Test that new tokens are in BERT vocabulary"""
    print("\n🔤 Testing BERT Vocabulary Integration")
    print("=" * 50)
    
    analyzer = BertVisualizationAnalyzer()
    
    # Test tokens that should be in vocabulary
    test_tokens = [
        'NOTES_STACKED',
        'NOTES_PRESENTED', 
        'NOTES_TAKEN',
        'PRIMARY_CARD_READER_ACTIVATED',
        'CASH_DISPENSED_SUMMARY',
        'OPCODE_BBC'
    ]
    
    vocab_tests_passed = 0
    for token in test_tokens:
        token_id = analyzer.tokenizer.convert_tokens_to_ids(token)
        is_in_vocab = token_id != analyzer.tokenizer.unk_token_id
        
        status = "✅ IN VOCAB" if is_in_vocab else "❌ NOT FOUND"
        print(f"{status} {token} (ID: {token_id})")
        
        if is_in_vocab:
            vocab_tests_passed += 1
    
    print(f"\n🎯 VOCABULARY: {vocab_tests_passed}/{len(test_tokens)} tokens in BERT vocabulary")
    
    return vocab_tests_passed == len(test_tokens)

if __name__ == "__main__":
    print("🚀 NCB EJ Pattern Testing Suite")
    print("=" * 60)
    
    # Run preprocessing tests
    preprocessing_success = test_ncb_ej_preprocessing()
    
    # Run vocabulary tests  
    vocab_success = test_vocabulary_tokens()
    
    print("\n" + "=" * 60)
    if preprocessing_success and vocab_success:
        print("🎉 ALL TESTS PASSED - NCB EJ patterns working correctly!")
        print("✨ Ready for enhanced BERT attention analysis")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Review patterns and vocabulary")
        sys.exit(1)
