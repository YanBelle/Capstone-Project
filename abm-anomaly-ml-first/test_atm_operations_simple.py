#!/usr/bin/env python3
"""Test the enhanced ATM operational patterns detection"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_atm_operational_patterns():
    """Test ATM operational patterns including cash totals and service status"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, OperationalMode
        
        labeler = EJLogLabeler()
        
        # Sample EJ log with ATM operational patterns
        sample_ej_operational = """
[020t*625*06/18/2025*00:42*] ATM STARTUP SEQUENCE INITIATED
[020t*626*06/18/2025*00:43*] PRIMARY CARD READER ACTIVATED
[020t*627*06/18/2025*00:44*] ATM IN SERVICE
[020t*628*06/18/2025*00:45*] Transaction started - Card inserted
[020t*629*06/18/2025*00:46*] NOTES PRESENTED
[020t*630*06/18/2025*00:47*] NOTES TAKEN
[020t*631*06/18/2025*00:48*] Transaction completed successfully
CASH TOTAL       TYPE1 TYPE2 TYPE3 TYPE4
DENOMINATION      1000  2000  5000  5000
DISPENSED        00271 00243 00621 00540
REJECTED         00003 00001 00010 00003
REMAINING        01729 01757 01379 01460
[020t*632*06/18/2025*00:49*] PRIMARY CARD READER ACTIVATED
        """.strip()
        
        print("Testing ATM operational patterns...")
        labels = labeler.label_log(sample_ej_operational)
        
        print("Total labels found: " + str(len(labels)))
        print("\nOperational Event Analysis:")
        
        # Analyze each label
        card_reader_activations = 0
        in_service_events = 0
        notes_events = 0
        cash_total_events = 0
        
        for i, label in enumerate(labels):
            print(str(i+1) + ". Line " + str(label.line_number) + ": " + label.event_type.value + " | " + label.operational_mode.value)
            
            if label.event_type == EventType.CARD_READER_ACTIVATED:
                card_reader_activations += 1
                print("   --> ATM ready for customers")
            elif label.event_type == EventType.ATM_IN_SERVICE:
                in_service_events += 1
                print("   --> ATM service mode active")
            elif label.event_type == EventType.NOTES_PRESENT:
                notes_events += 1
                print("   --> Cash issued to customer")
            elif label.event_type == EventType.NOTES_TAKEN:
                notes_events += 1
                print("   --> Customer took cash")
            elif label.event_type == EventType.CASH_TOTAL_REPORT or label.denomination_data:
                cash_total_events += 1
                if label.denomination_data:
                    data_type = list(label.denomination_data.keys())[0]
                    print("   --> Cash data: " + data_type)
                    
                    # Show cash analysis if available
                    if 'cash_analysis' in label.metadata:
                        analysis = label.metadata['cash_analysis']
                        print("       Health Score: " + str(round(analysis.get('cash_health_score', 0), 2)))
                        if analysis.get('insights'):
                            for insight in analysis['insights']:
                                print("       Insight: " + insight)
        
        print("\n--- Summary ---")
        print("Card Reader Activations: " + str(card_reader_activations))
        print("In Service Events: " + str(in_service_events))
        print("Notes Events: " + str(notes_events))
        print("Cash Total Events: " + str(cash_total_events))
        
        # Validate expected detections
        expected_patterns = {
            'card_reader_activated': card_reader_activations >= 2,
            'in_service_detected': in_service_events >= 1,
            'notes_handling': notes_events >= 2,
            'cash_reconciliation': cash_total_events >= 4
        }
        
        all_passed = True
        for pattern, detected in expected_patterns.items():
            status = "[PASS]" if detected else "[FAIL]"
            result = 'DETECTED' if detected else 'MISSING'
            print(status + " " + pattern + ": " + result)
            if not detected:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print("[FAIL] ATM operational test failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run ATM operational pattern test"""
    print("ATM Operational Patterns Test")
    print("=" * 40)
    
    if test_atm_operational_patterns():
        print("\nRESULT: PASS")
        print("[PASS] ATM operational pattern detection working!")
        print("[PASS] PRIMARY CARD READER ACTIVATED properly detected")
        print("[PASS] NOTES PRESENTED/TAKEN cash handling tracked")
        print("[PASS] Cash total reports analyzed")
    else:
        print("\nRESULT: FAIL")
        print("[FAIL] ATM operational tests failed")

if __name__ == "__main__":
    main()
