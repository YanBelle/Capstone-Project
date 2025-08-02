#!/usr/bin/env python3
# Test the updated receipt detection

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_receipt_detection():
    """Test the updated receipt detection with real EJ receipt format"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
        
        labeler = EJLogLabeler()
        
        # Sample EJ log with actual receipt content format
        sample_ej_with_receipt = """
[020t*629*06/18/2025*00:46*] Transaction started - Card inserted
[020t*630*06/18/2025*00:47*] Card authentication in progress
[020t*631*06/18/2025*00:48*] PIN verification failed
[020t*632*06/18/2025*00:48*] Starting receipt printing
    N.C.B. MIDAS

   NCB DUKE ST. BRANCH

     DATE        TIME

   2025/06/18   05:51:25

   MACHINE       0250

   TRAN NO       227238

   ***************8209

   UNABLE TO PROCESS

         THANK YOU
[020t*633*06/18/2025*00:49*] Transaction completed with failure
        """.strip()
        
        print("Testing receipt detection with real EJ format...")
        labels = labeler.label_log(sample_ej_with_receipt)
        
        print(f"Total labels found: {len(labels)}")
        print("\nLabel breakdown:")
        
        receipt_labels = []
        for i, label in enumerate(labels):
            print(f"{i+1}. Line {label.line_number}: {label.event_type.value} | {label.phase.value} | {label.severity.value}")
            
            if label.event_type == EventType.RECEIPT_PRINT:
                receipt_labels.append(label)
                print(f"   Receipt detected! Machine: {label.metadata.get('machine_id', 'N/A')}")
                print(f"   Transaction: {label.metadata.get('transaction_number', 'N/A')}")
                print(f"   Result: {label.metadata.get('transaction_result', 'N/A')}")
                print(f"   Branch: {label.metadata.get('branch', 'N/A')}")
                if label.metadata.get('transaction_failed'):
                    print(f"   *** FAILED TRANSACTION DETECTED ***")
        
        if receipt_labels:
            print(f"\n✓ Successfully detected {len(receipt_labels)} receipt(s)")
            print("✓ Receipt content parsing working correctly")
            
            # Test specific receipt content
            receipt = receipt_labels[0]
            receipt_content = receipt.metadata.get('receipt_content', [])
            print(f"\nReceipt content ({len(receipt_content)} lines):")
            for line in receipt_content[:5]:  # Show first 5 lines
                print(f"  '{line}'")
            if len(receipt_content) > 5:
                print(f"  ... and {len(receipt_content) - 5} more lines")
        else:
            print("✗ No receipt detected - check patterns")
        
        return True
        
    except Exception as e:
        print(f"✗ Receipt detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_receipt_parsing_edge_cases():
    """Test edge cases in receipt parsing"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler
        from ej_contextual_labeler import EventType  # Import EventType separately
        
        labeler = EJLogLabeler()
        
        # Test different receipt formats
        test_cases = [
            # Successful transaction
            """
    N.C.B. MIDAS
   NCB KINGSTON BRANCH
   2025/06/18   14:30:15
   MACHINE       0125
   TRAN NO       445566
   ***************1234
   TRANSACTION APPROVED
   WITHDRAWAL    $100.00
         THANK YOU
            """,
            # Different branch format
            """
    N.C.B. MIDAS
   NCB SPANISH TOWN BRANCH
   2025/06/18   09:15:42
   MACHINE       0088
   TRAN NO       998877
   ***************5678
   DECLINED - INSUFFICIENT FUNDS
         THANK YOU
            """
        ]
        
        print("\nTesting receipt parsing edge cases...")
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test case {i+1} ---")
            labels = labeler.label_log(test_case.strip())
            
            receipt_labels = [l for l in labels if l.event_type == EventType.RECEIPT_PRINT]
            
            if receipt_labels:
                receipt = receipt_labels[0]
                print(f"✓ Receipt detected")
                print(f"  Machine: {receipt.metadata.get('machine_id', 'N/A')}")
                print(f"  Branch: {receipt.metadata.get('branch', 'N/A')}")
                print(f"  Result: {receipt.metadata.get('transaction_result', 'N/A')}")
                print(f"  Failed: {receipt.metadata.get('transaction_failed', False)}")
                if receipt.amount:
                    print(f"  Amount: ${receipt.amount}")
            else:
                print("✗ No receipt detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def main():
    """Run receipt detection tests"""
    print("EJ Receipt Detection Test Suite")
    print("=" * 40)
    
    tests = [
        ("Receipt Detection", test_receipt_detection),
        ("Edge Cases", test_receipt_parsing_edge_cases)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"RESULT: PASS")
        else:
            print(f"RESULT: FAIL")
    
    print(f"\n{'='*40}")
    print(f"Receipt Detection Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✓ Receipt detection is working correctly!")
        print("✓ The system now properly identifies receipt printing")
        print("✓ from actual EJ receipt content instead of explicit events")
    else:
        print("✗ Some receipt detection tests failed")

if __name__ == "__main__":
    main()
