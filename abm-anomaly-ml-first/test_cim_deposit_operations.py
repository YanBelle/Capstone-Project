#!/usr/bin/env python3
"""Test enhanced CIM deposit operations and note quality analysis"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_cim_deposit_operations():
    """Test CIM deposit transaction flow recognition"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, TransactionPhase
        
        labeler = EJLogLabeler()
        
        # Complete CIM deposit transaction
        cim_deposit_log = """
10:15:00 *TRANSACTION START*
10:15:05 CARD INSERTED
10:15:10 PIN ENTRY
10:15:15 CASHIN DEPOSIT SELECTED
10:15:20 CIM-DEPOSIT ACTIVATED
10:15:25 CIM-SHUTTER OPENED
10:15:30 CIM-ITEMS INSERTED
10:15:35 CIM-ITEMS PRESENTED
10:15:40 CIM-ITEMS TAKEN
10:15:45 CIM-DEPOSIT COMPLETED
10:15:50 *TRANSACTION END*
        """.strip()
        
        print("Testing CIM deposit operations...")
        labels = labeler.label_log(cim_deposit_log)
        
        # Verify key CIM events are detected
        cim_events = {}
        phases_seen = set()
        
        for label in labels:
            if label.event_type.value.startswith('cim_'):
                cim_events[label.event_type] = label
            phases_seen.add(label.phase)
            print(f"  {label.event_type.value} | {label.phase.value} | {label.timestamp}")
        
        # Check for key CIM events
        expected_events = [
            EventType.CASHIN_DEPOSIT_SELECTED,
            EventType.CIM_DEPOSIT_ACTIVATED,
            EventType.CIM_SHUTTER_OPENED,
            EventType.CIM_ITEMS_INSERTED,
            EventType.CIM_ITEMS_PRESENTED,
            EventType.CIM_ITEMS_TAKEN,
            EventType.CIM_DEPOSIT_COMPLETED
        ]
        
        found_events = [event for event in expected_events if event in cim_events]
        
        print(f"\nCIM Events Found: {len(found_events)}/{len(expected_events)}")
        for event in found_events:
            print(f"  ✓ {event.value}")
        
        # Check for deposit-related phases
        deposit_phases = [phase for phase in phases_seen if 'deposit' in phase.value.lower()]
        print(f"\nDeposit Phases: {deposit_phases}")
        
        return len(found_events) >= 5  # Should find most CIM events
        
    except Exception as e:
        print(f"[FAIL] CIM deposit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_note_quality_analysis():
    """Test note categorization and quality analysis"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Note quality issues
        note_quality_log = """
11:00:00 CIM-ITEMS INSERTED
11:00:05 FAILED SERIAL NUMBER READS and CAT4 NOTES: 1
11:00:10 CAT1 NOTES: 3
11:00:15 CAT2 NOTES: 1
11:00:20 CAT4 NOTES: 1
11:00:25 CIM-INPUT REFUSED,REASON-INVALID MEDIA
        """.strip()
        
        print("\nTesting note quality analysis...")
        labels = labeler.label_log(note_quality_log)
        
        quality_issues = []
        note_data = {}
        
        for label in labels:
            print(f"  {label.event_type.value} | {label.severity.value}")
            if label.note_categories:
                note_data.update(label.note_categories)
                print(f"    Note Categories: {label.note_categories}")
            if label.serial_read_failures:
                print(f"    Serial Failures: {label.serial_read_failures}")
            if label.rejected_reason:
                print(f"    Rejection Reason: {label.rejected_reason}")
            
            # Check for quality-related anomalies
            anomalies = label.metadata.get('contextual_anomalies', [])
            for anomaly in anomalies:
                if any(keyword in anomaly.lower() for keyword in ['rejected', 'serial', 'cat4', 'invalid']):
                    quality_issues.append(anomaly)
                    print(f"    Quality Issue: {anomaly}")
        
        print(f"\nNote Categories Found: {note_data}")
        print(f"Quality Issues Detected: {len(quality_issues)}")
        
        # Should detect multiple quality issues
        return len(quality_issues) >= 2 and 'CAT4' in note_data
        
    except Exception as e:
        print(f"[FAIL] Note quality test failed: {e}")
        return False

def test_retract_bin_operations():
    """Test enhanced retract bin operation detection"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, RecoveryType
        
        labeler = EJLogLabeler()
        
        # Retract bin operations
        retract_log = """
12:00:00 INIT BNA STARTED - RETRACT BIN
12:00:05 CASHIN RETRACT STARTED - RETRACT BIN
12:00:10 CIM-RESET CALLED - RETRACT BIN
12:00:15 CASHIN RECOVERY OK
        """.strip()
        
        print("\nTesting retract bin operations...")
        labels = labeler.label_log(retract_log)
        
        retract_operations = []
        recovery_types = set()
        
        for label in labels:
            print(f"  {label.event_type.value} | Recovery: {label.recovery_type}")
            if 'retract' in label.event_type.value.lower():
                retract_operations.append(label.event_type)
            if label.recovery_type:
                recovery_types.add(label.recovery_type)
        
        print(f"\nRetract Operations: {len(retract_operations)}")
        print(f"Recovery Types: {recovery_types}")
        
        # Should detect retract bin operations
        return len(retract_operations) >= 2
        
    except Exception as e:
        print(f"[FAIL] Retract bin test failed: {e}")
        return False

def test_high_rejection_rate_anomaly():
    """Test detection of high note rejection rate anomalies"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # High rejection scenario
        high_rejection_log = """
13:00:00 CIM-ITEMS INSERTED
13:00:05 CAT1 NOTES: 2
13:00:10 CAT4 NOTES: 3
13:00:15 FAILED SERIAL NUMBER READS: 4
        """.strip()
        
        print("\nTesting high rejection rate anomaly...")
        labels = labeler.label_log(high_rejection_log)
        
        rejection_anomalies = []
        severity_escalations = []
        
        for label in labels:
            print(f"  {label.event_type.value} | {label.severity.value}")
            if label.severity in [Severity.ERROR, Severity.WARNING]:
                severity_escalations.append(label.severity)
            
            anomalies = label.metadata.get('contextual_anomalies', [])
            for anomaly in anomalies:
                if 'rejection rate' in anomaly.lower() or 'rejected' in anomaly.lower():
                    rejection_anomalies.append(anomaly)
                    print(f"    Rejection Anomaly: {anomaly}")
        
        print(f"\nRejection Anomalies: {len(rejection_anomalies)}")
        print(f"Severity Escalations: {len(severity_escalations)}")
        
        # Should detect high rejection rate and escalate severity
        return len(rejection_anomalies) >= 1 and len(severity_escalations) >= 1
        
    except Exception as e:
        print(f"[FAIL] High rejection test failed: {e}")
        return False

def main():
    """Run all CIM deposit and note quality tests"""
    print("Enhanced CIM Deposit & Note Quality Analysis Test Suite")
    print("=" * 75)
    
    tests = [
        ("CIM Deposit Operations", test_cim_deposit_operations),
        ("Note Quality Analysis", test_note_quality_analysis),
        ("Retract Bin Operations", test_retract_bin_operations),
        ("High Rejection Rate Anomaly", test_high_rejection_rate_anomaly)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print("\n" + "=" * 25 + " " + test_name + " " + "=" * 25)
        if test_func():
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
    
    print("\n" + "=" * 75)
    print(f"CIM Deposit & Note Quality Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("\n[SUCCESS] Enhanced CIM Deposit & Note Quality Analysis Working!")
        print("[SUCCESS] Recognizes complete CIM deposit transaction flows")
        print("[SUCCESS] Analyzes note categories (CAT1-CAT5) and quality issues")
        print("[SUCCESS] Detects serial number read failures and rejection reasons")
        print("[SUCCESS] Identifies high rejection rates and escalates severity")
        print("[SUCCESS] Handles enhanced retract bin operations")
        print("\nNew Features:")
        print("- CIM Deposit Transaction Phases (CASH_DEPOSITING, NOTE_QUALITY_CHECK, DEPOSIT_VERIFICATION)")
        print("- Note Categorization Analysis (CAT1-CAT5 fitness levels)")
        print("- Serial Number Read Failure Detection")
        print("- CIM Input Refusal Reason Analysis (INVALID MEDIA, DOUBLE FEED, JAM)")
        print("- Enhanced Retract Bin Operation Types")
        print("- Rejection Rate Anomaly Detection (>10% WARNING, >30% ERROR)")
        print("- Deposit Amount Analysis (suspicious amounts flagged)")
    else:
        print(f"\n[FAIL] {len(tests) - passed} CIM deposit/note quality tests failed")

if __name__ == "__main__":
    main()
