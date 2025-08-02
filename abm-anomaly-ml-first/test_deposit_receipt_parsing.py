#!/usr/bin/env python3
"""
Test the enhanced receipt parsing for cash deposits
"""

import sys
import os
sys.path.append('services/anomaly-detector')

from ej_contextual_labeler import EJLogLabeler

def test_cash_deposit_receipt():
    """Test parsing of the cash deposit receipt example"""
    
    # Your successful cash deposit receipt example
    deposit_receipt_log = """
       N.C.B. MIDAS
   NCB DUKE ST. BRANCH
     DATE        TIME
   2025/06/18   09:28:33
   MACHINE       0250
   TRAN NO       227329
   AUTHORIZATION 092629
   ************6595
   NO.OF BILLS DEPOSITED
            00  X $ 50
            00  X $ 100
            00  X $ 500
            00  X $ 1000
            00  X $ 2000
            02  X $ 5000
 DEPOSIT AC #   VALUE JMD
                10,000.00
 AVAILABLE      67,744.47
 PLEASE CONTACT OUR CARE
  CENTRE 1-888-622-3477
"""
    
    print("🧪 Testing Cash Deposit Receipt Parsing\n")
    
    labeler = EJLogLabeler()
    labels = labeler.label_log(deposit_receipt_log)
    
    # Find the receipt label
    receipt_labels = [label for label in labels if label.event_type.value == 'receipt_print']
    
    if receipt_labels:
        receipt_label = receipt_labels[0]
        print("✅ Receipt detected and parsed!")
        print(f"📄 Receipt spans lines {receipt_label.metadata.get('receipt_start_line', 'N/A')} to {receipt_label.line_number}")
        print(f"🏦 Bank: {receipt_label.metadata.get('bank', 'N/A')}")
        print(f"🏢 Branch: {receipt_label.metadata.get('branch', 'N/A')}")
        print(f"🏧 Machine ID: {receipt_label.metadata.get('machine_id', 'N/A')}")
        print(f"📋 Transaction Number: {receipt_label.metadata.get('transaction_number', 'N/A')}")
        print(f"🔐 Authorization: {receipt_label.metadata.get('authorization_code', 'N/A')}")
        print(f"💳 Masked Card: {receipt_label.metadata.get('masked_card', 'N/A')}")
        
        # Deposit-specific information
        if 'deposit_bills' in receipt_label.metadata:
            print("\n💰 Deposit Bill Breakdown:")
            total_bills = 0
            for denom, count in receipt_label.metadata['deposit_bills'].items():
                denomination = denom.replace('JMD_', 'JMD$')
                print(f"   {count:02d} x {denomination}")
                total_bills += count
            print(f"   Total Bills: {total_bills}")
        
        deposit_value = receipt_label.metadata.get('deposit_value') or receipt_label.metadata.get('calculated_deposit_value')
        if deposit_value:
            print(f"💵 Deposit Value: JMD${deposit_value:,.2f}")
        
        if 'available_balance' in receipt_label.metadata:
            print(f"🏦 Available Balance: JMD${receipt_label.metadata['available_balance']:,.2f}")
        
        if 'contact_number' in receipt_label.metadata:
            print(f"📞 Support Contact: {receipt_label.metadata['contact_number']}")
        
        # Receipt classification
        receipt_type = receipt_label.metadata.get('receipt_type', 'UNKNOWN')
        print(f"\n🏷️ Receipt Type: {receipt_type}")
        
        if receipt_type == 'CASH_DEPOSIT':
            print("✅ Correctly identified as cash deposit receipt!")
        else:
            print("❌ Receipt type not correctly identified")
        
        # Verify ending detection
        receipt_end_type = receipt_label.metadata.get('receipt_end_type', 'N/A')
        print(f"🔚 Receipt End Type: {receipt_end_type}")
        
    else:
        print("❌ No receipt label found in the parsed output")
        print(f"Found {len(labels)} labels total:")
        for label in labels:
            print(f"  - Line {label.line_number}: {label.event_type.value}")

def test_receipt_ending_patterns():
    """Test different receipt ending patterns"""
    
    print("\n🔬 Testing Receipt Ending Patterns\n")
    
    labeler = EJLogLabeler()
    
    # Test 1: Traditional "THANK YOU" ending
    thank_you_receipt = """
   N.C.B. MIDAS
   TRANSACTION APPROVED
   THANK YOU
"""
    
    print("Test 1: THANK YOU ending")
    labels = labeler.label_log(thank_you_receipt)
    receipt_labels = [l for l in labels if l.event_type.value == 'receipt_print']
    if receipt_labels:
        end_type = receipt_labels[0].metadata.get('receipt_end_type', 'UNKNOWN')
        print(f"✅ Detected ending: {end_type}")
    else:
        print("❌ Receipt not detected")
    
    # Test 2: Contact centre ending
    contact_receipt = """
   N.C.B. MIDAS
   DEPOSIT COMPLETED
   PLEASE CONTACT OUR CARE
  CENTRE 1-888-622-3477
"""
    
    print("\nTest 2: Contact centre ending")
    labels = labeler.label_log(contact_receipt)
    receipt_labels = [l for l in labels if l.event_type.value == 'receipt_print']
    if receipt_labels:
        end_type = receipt_labels[0].metadata.get('receipt_end_type', 'UNKNOWN')
        print(f"✅ Detected ending: {end_type}")
    else:
        print("❌ Receipt not detected")

if __name__ == "__main__":
    try:
        test_cash_deposit_receipt()
        test_receipt_ending_patterns()
        print("\n🎉 All receipt parsing tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
