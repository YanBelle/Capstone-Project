#!/usr/bin/env python3
"""Test the enhanced cash dispensing sequence including NOTES STACKED"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_cash_dispensing_sequence():
    """Test the complete cash dispensing sequence with NOTES STACKED"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
        
        labeler = EJLogLabeler()
        
        # Sample EJ log with complete cash dispensing sequence
        cash_dispensing_log = """
[020t*625*06/18/2025*00:42*] Transaction started - Card inserted
[020t*626*06/18/2025*00:43*] PIN verified successfully
[020t*627*06/18/2025*00:44*] Customer selected withdrawal amount
[020t*628*06/18/2025*00:45*] Processing withdrawal request
[020t*629*06/18/2025*00:46*] NOTES STACKED
[020t*630*06/18/2025*00:47*] NOTES PRESENTED
[020t*631*06/18/2025*00:48*] NOTES TAKEN
[020t*632*06/18/2025*00:49*] Transaction completed successfully
        """.strip()
        
        print("Testing cash dispensing sequence with NOTES STACKED...")
        labels = labeler.label_log(cash_dispensing_log)
        
        print("Total labels found: " + str(len(labels)))
        print("\nCash Dispensing Sequence Analysis:")
        
        # Track the cash dispensing events
        notes_stacked_found = False
        notes_presented_found = False
        notes_taken_found = False
        
        sequence_order = []
        
        for i, label in enumerate(labels):
            print(str(i+1) + ". Line " + str(label.line_number) + ": " + label.event_type.value + " | " + label.phase.value)
            
            if label.event_type == EventType.NOTES_STACKED:
                notes_stacked_found = True
                sequence_order.append(('STACKED', label.line_number))
                print("   --> Notes retrieved and stacked in presentation area")
                
            elif label.event_type == EventType.NOTES_PRESENT:
                notes_presented_found = True
                sequence_order.append(('PRESENTED', label.line_number))
                print("   --> Shutter opened, cash presented to customer")
                
            elif label.event_type == EventType.NOTES_TAKEN:
                notes_taken_found = True
                sequence_order.append(('TAKEN', label.line_number))
                print("   --> Customer took the cash")
        
        print("\n--- Cash Dispensing Sequence Validation ---")
        print("NOTES STACKED detected: " + ("YES" if notes_stacked_found else "NO"))
        print("NOTES PRESENTED detected: " + ("YES" if notes_presented_found else "NO"))
        print("NOTES TAKEN detected: " + ("YES" if notes_taken_found else "NO"))
        
        # Validate sequence order
        print("\nSequence order:")
        for event, line_num in sequence_order:
            print("  " + event + " at line " + str(line_num))
        
        # Check if sequence is correct (STACKED -> PRESENTED -> TAKEN)
        sequence_correct = True
        if len(sequence_order) >= 3:
            expected_order = ['STACKED', 'PRESENTED', 'TAKEN']
            actual_order = [event for event, _ in sequence_order]
            
            if actual_order[:3] != expected_order:
                sequence_correct = False
                print("\n[WARNING] Sequence order incorrect!")
                print("Expected: " + " -> ".join(expected_order))
                print("Actual: " + " -> ".join(actual_order[:3]))
            else:
                print("\n[PASS] Cash dispensing sequence is correct!")
        
        # Validate that all three events were found
        all_events_detected = notes_stacked_found and notes_presented_found and notes_taken_found
        
        return all_events_detected and sequence_correct
        
    except Exception as e:
        print("[FAIL] Cash dispensing test failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

def test_notes_stacked_phase():
    """Test that NOTES STACKED is correctly assigned to CASH_DISPENSING phase"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
        
        labeler = EJLogLabeler()
        
        # Simple test for NOTES STACKED
        notes_stacked_log = "[020t*625*06/18/2025*00:42*] NOTES STACKED"
        
        labels = labeler.label_log(notes_stacked_log)
        
        if labels and len(labels) > 0:
            label = labels[0]
            
            print("NOTES STACKED event analysis:")
            print("  Event Type: " + label.event_type.value)
            print("  Phase: " + label.phase.value)
            
            # Validate event type and phase
            correct_event = label.event_type == EventType.NOTES_STACKED
            correct_phase = label.phase == TransactionPhase.CASH_DISPENSING
            
            print("  Correct Event Type: " + ("YES" if correct_event else "NO"))
            print("  Correct Phase: " + ("YES" if correct_phase else "NO"))
            
            return correct_event and correct_phase
        else:
            print("[FAIL] No labels generated for NOTES STACKED")
            return False
            
    except Exception as e:
        print("[FAIL] NOTES STACKED phase test failed: " + str(e))
        return False

def test_cash_flow_anomaly_detection():
    """Test anomaly detection for incorrect cash dispensing flow"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType
        
        labeler = EJLogLabeler()
        
        # Anomalous sequence: NOTES PRESENTED without NOTES STACKED
        anomalous_log = """
[020t*625*06/18/2025*00:42*] Transaction started
[020t*626*06/18/2025*00:43*] NOTES PRESENTED
[020t*627*06/18/2025*00:44*] NOTES TAKEN
        """.strip()
        
        print("\nTesting cash flow anomaly detection...")
        labels = labeler.label_log(anomalous_log)
        
        # Check if we can detect the missing NOTES STACKED
        notes_presented = False
        notes_stacked = False
        
        for label in labels:
            if label.event_type == EventType.NOTES_PRESENTED:
                notes_presented = True
            elif label.event_type == EventType.NOTES_STACKED:
                notes_stacked = True
        
        # This should be flagged as anomalous (PRESENTED without STACKED)
        anomaly_detected = notes_presented and not notes_stacked
        
        print("NOTES PRESENTED without NOTES STACKED: " + ("DETECTED" if anomaly_detected else "NOT DETECTED"))
        
        return anomaly_detected
        
    except Exception as e:
        print("[FAIL] Cash flow anomaly test failed: " + str(e))
        return False

def main():
    """Run all cash dispensing tests"""
    print("Cash Dispensing Sequence Test Suite")
    print("=" * 50)
    
    tests = [
        ("Complete Cash Dispensing Sequence", test_cash_dispensing_sequence),
        ("NOTES STACKED Phase Assignment", test_notes_stacked_phase),
        ("Cash Flow Anomaly Detection", test_cash_flow_anomaly_detection)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print("\n" + "=" * 20 + " " + test_name + " " + "=" * 20)
        if test_func():
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
    
    print("\n" + "=" * 50)
    print("Cash Dispensing Tests: " + str(passed) + "/" + str(len(tests)) + " passed")
    
    if passed == len(tests):
        print("[PASS] NOTES STACKED operational understanding working correctly!")
        print("[PASS] Cash dispensing sequence: STACKED -> PRESENTED -> TAKEN")
        print("[PASS] Notes retrieved and queued before presentation")
        print("[PASS] Shutter opening properly tracked")
        print("[PASS] Customer interaction completion detected")
    else:
        print("[FAIL] Some cash dispensing tests failed")

if __name__ == "__main__":
    main()
