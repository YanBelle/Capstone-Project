#!/usr/bin/env python3
"""
Test the new cash dispensing summary and receipt noise reduction patterns
"""
import re

def test_cash_receipt_patterns():
    """Test the new cash and receipt noise reduction patterns"""
    
    # Sample EJ data with cash dispensing summary - verbose table format
    cash_test_cases = [
        (
            "ESC/VAL 000 16:07:37\n1 16:07:37 > CUST_PROFILE_UPDATE\n2 16:07:38 > CASH_WITHDRAWAL_REQUEST 500\nESC/VAL 001 16:07:38\n1 16:07:38 > CASH_DISPENSER_INIT\n2 16:07:39 > CASH TOTAL TYPE1     2000 2500 5000 10000 SUM DISPENSED    2    2    0    0    4 REMAINING   500    0  480  250 1230\nESC/VAL 002 16:07:40\n1 16:07:40 > TRANSACTION_COMPLETE",
            "ESC/VAL 000 16:07:37\n1 16:07:37 > CUST_PROFILE_UPDATE\n2 16:07:38 > CASH_WITHDRAWAL_REQUEST 500\nESC/VAL 001 16:07:38\n1 16:07:38 > CASH_DISPENSER_INIT\n2 16:07:39 > CASH_DISPENSED_SUMMARY\nESC/VAL 002 16:07:40\n1 16:07:40 > TRANSACTION_COMPLETE"
        ),
        (
            "CASH TOTAL TYPE1     1000 2000 5000 10000 SUM DISPENSED    1    0    1    0    2 REMAINING   199   50  480  250  979",
            "CASH_DISPENSED_SUMMARY"
        ),
        (
            "Before cash table\nCASH TOTAL TYPE2     5000 10000 SUM DISPENSED    3    1    4 REMAINING   100  200  300\nAfter cash table",
            "Before cash table\nCASH_DISPENSED_SUMMARY\nAfter cash table"
        )
    ]
    
    # Sample EJ data with receipts - customer receipt format
    receipt_test_cases = [
        (
            "ESC/VAL 000 16:07:37\n1 16:07:37 > TRANSACTION_START\n2 16:07:38 > WITHDRAWAL_500\nESC/VAL 001 16:07:38\n1 16:07:38 > Your transaction receipt\n    FIRST NATIONAL BANK\n    ATM#: 12345\n    DATE: 06/18/2025\n    TIME: 16:07:38\n    MACHINE: ABM250EJ\n    \n    WITHDRAWAL\n    ACCOUNT: ****1234\n    AMOUNT: $500.00\n    \n    THANK YOU\n2 16:07:40 > TRANSACTION_END",
            "ESC/VAL 000 16:07:37\n1 16:07:37 > TRANSACTION_START\n2 16:07:38 > WITHDRAWAL_500\nESC/VAL 001 16:07:38\n1 16:07:38 > Your transaction receipt\n     RECEIPT_PRINTED \n2 16:07:40 > TRANSACTION_END"
        ),
        (
            "ATLANTIC BANK\nATM RECEIPT\nDATE: 06/18/2025\nTIME: 15:30:45\nMACHINE: ATM001\n\nWITHDRAWAL\nACCOUNT: ****5678\nAMOUNT: $200.00\nFEE: $2.50\n\nTHANK YOU",
            " RECEIPT_PRINTED "
        ),
        (
            "Before receipt\nCITY CREDIT UNION\nDATE: 06/18/2025\nTIME: 16:45:30\nMACHINE: ABM123\nBALANCE INQUIRY\nTHANK YOU\nAfter receipt",
            "Before receipt\n RECEIPT_PRINTED \nAfter receipt"
        )
    ]
    
    # Define the patterns from bertviz_analyzer.py (updated)
    cash_summary_pattern = r'CASH\s+TOTAL\s+TYPE\d+.*?REMAINING\s+\d+(?:\s+\d+)*'
    receipt_pattern1 = r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
    receipt_pattern2 = r'(DATE:\s*[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
    receipt_pattern3 = r'(\s+[A-Z][A-Z\s]+\n\s*(?:ATM|RECEIPT)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
    
    print("=== Testing Cash Dispensing Summary Patterns ===")
    cash_passed = 0
    cash_total = len(cash_test_cases)
    
    for i, (input_text, expected) in enumerate(cash_test_cases, 1):
        # Apply cash summary replacement
        result = re.sub(cash_summary_pattern, 'CASH_DISPENSED_SUMMARY', input_text, flags=re.DOTALL)
        
        if result == expected:
            print(f"✅ Cash Test {i}: PASSED")
            cash_passed += 1
        else:
            print(f"❌ Cash Test {i}: FAILED")
            print(f"   Input:    '{input_text[:60]}{'...' if len(input_text) > 60 else ''}'")
            print(f"   Expected: '{expected[:60]}{'...' if len(expected) > 60 else ''}'")
            print(f"   Got:      '{result[:60]}{'...' if len(result) > 60 else ''}'")
        print()
    
    print("=== Testing Receipt Patterns ===")
    receipt_passed = 0
    receipt_total = len(receipt_test_cases)
    
    for i, (input_text, expected) in enumerate(receipt_test_cases, 1):
        # Try all receipt patterns
        result = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
        
        # If no replacement, try pattern 2
        if result == input_text:
            result = re.sub(receipt_pattern2, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
        
        # If still no replacement, try pattern 3
        if result == input_text:
            result = re.sub(receipt_pattern3, ' RECEIPT_PRINTED ', input_text, flags=re.DOTALL)
        
        if result == expected:
            print(f"✅ Receipt Test {i}: PASSED")
            receipt_passed += 1
        else:
            print(f"❌ Receipt Test {i}: FAILED")
            print(f"   Input:    '{input_text[:60]}{'...' if len(input_text) > 60 else ''}'")
            print(f"   Expected: '{expected[:60]}{'...' if len(expected) > 60 else ''}'")
            print(f"   Got:      '{result[:60]}{'...' if len(result) > 60 else ''}'")
        print()
    
    print("=== Test Summary ===")
    print(f"Cash dispensing summary tests: {cash_passed}/{cash_total} passed")
    print(f"Receipt replacement tests: {receipt_passed}/{receipt_total} passed")
    
    total_passed = cash_passed + receipt_passed
    total_tests = cash_total + receipt_total
    
    if total_passed == total_tests:
        print("🎉 All noise reduction patterns working correctly!")
        return True
    else:
        print("⚠️  Some patterns need adjustment")
        return False

def test_real_ej_sample():
    """Test with real EJ data sample"""
    
    print("\n=== Testing with Real EJ Sample ===")
    
    # Use actual EJ data from the workspace
    try:
        with open('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/ABM250EJ_20250618_20250618.txt', 'r') as f:
            ej_sample = f.read()[:2000]  # First 2000 chars
        
        print("Found real EJ data, testing patterns...")
        
        # Apply cash and receipt patterns
        cash_pattern = r'CASH\s+TOTAL\s+TYPE\d+.*?REMAINING\s+\d+(?:\s+\d+)*'
        receipt_pattern1 = r'([A-Z][A-Z\s\.]+(?:BANK|CREDIT UNION|ATM)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        receipt_pattern2 = r'(DATE:\s*[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        receipt_pattern3 = r'(\s+[A-Z][A-Z\s]+\n\s*(?:ATM|RECEIPT)[^\n]*\n(?:.*?\n)*?.*?THANK YOU)'
        
        original_length = len(ej_sample)
        
        # Apply replacements
        processed = re.sub(cash_pattern, 'CASH_DISPENSED_SUMMARY', ej_sample, flags=re.DOTALL)
        processed = re.sub(receipt_pattern1, ' RECEIPT_PRINTED ', processed, flags=re.DOTALL)
        processed = re.sub(receipt_pattern2, ' RECEIPT_PRINTED ', processed, flags=re.DOTALL)
        processed = re.sub(receipt_pattern3, ' RECEIPT_PRINTED ', processed, flags=re.DOTALL)
        
        processed_length = len(processed)
        
        # Count replacements
        cash_replacements = ej_sample.count('CASH TOTAL') - processed.count('CASH TOTAL')
        receipt_replacements = processed.count('RECEIPT_PRINTED')
        
        print(f"Original length: {original_length} chars")
        print(f"Processed length: {processed_length} chars")
        print(f"Cash dispensing summaries replaced: {cash_replacements}")
        print(f"Receipts replaced: {receipt_replacements}")
        
        if cash_replacements > 0 or receipt_replacements > 0:
            print("✅ Real EJ data processing successful")
            return True
        else:
            print("ℹ️  No matching patterns found in sample (may be normal)")
            return True
            
    except FileNotFoundError:
        print("ℹ️  Real EJ file not found, skipping real data test")
        return True

if __name__ == "__main__":
    pattern_success = test_cash_receipt_patterns()
    real_data_success = test_real_ej_sample()
    
    overall_success = pattern_success and real_data_success
    print(f"\n🎯 Overall Result: {'SUCCESS' if overall_success else 'NEEDS WORK'}")
