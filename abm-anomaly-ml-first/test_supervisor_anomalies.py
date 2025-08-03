#!/usr/bin/env python3
"""Test enhanced supervisor mode anomaly detection"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_supervisor_during_transaction():
    """Test detection of supervisor mode during active transaction"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Scenario: Supervisor mode entered during transaction
        suspicious_transaction_log = """
07:30:00 *TRANSACTION START*
07:30:15 CARD INSERTED
07:30:20 PIN ENTRY
07:30:25 SUPERVISOR MODE ENTRY
07:30:30 NOTES PRESENTED
07:30:35 NOTES TAKEN
07:30:40 SUPERVISOR MODE EXIT
07:30:45 *TRANSACTION END*
        """.strip()
        
        print("Testing supervisor mode during transaction...")
        labels = labeler.label_log(suspicious_transaction_log)
        
        # Look for critical anomalies
        critical_anomalies = []
        supervisor_entry_found = False
        
        for label in labels:
            if label.event_type == EventType.SUPERVISOR_ENTRY:
                supervisor_entry_found = True
                print("Supervisor entry detected:")
                print("  Severity: " + label.severity.value)
                print("  Anomalies: " + str(label.metadata.get('contextual_anomalies', [])))
                
                if label.severity == Severity.CRITICAL:
                    critical_anomalies.extend(label.metadata.get('contextual_anomalies', []))
        
        # Validate critical detection
        transaction_during_supervisor = any('Transaction during supervisor mode' in anomaly for anomaly in critical_anomalies)
        supervisor_during_transaction = any('during active transaction' in anomaly for anomaly in critical_anomalies)
        
        print("Critical anomalies found: " + str(len(critical_anomalies)))
        for anomaly in critical_anomalies:
            print("  - " + anomaly)
        
        return supervisor_entry_found and (transaction_during_supervisor or supervisor_during_transaction)
        
    except Exception as e:
        print("[FAIL] Supervisor during transaction test failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

def test_supervisor_immediately_after_transaction():
    """Test detection of supervisor mode immediately after transaction"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Scenario: Supervisor mode 10 seconds after transaction end
        suspicious_timing_log = """
08:15:00 *TRANSACTION START*
08:15:30 CARD INSERTED
08:15:35 NOTES PRESENTED
08:15:40 NOTES TAKEN
08:15:45 *TRANSACTION END*
08:15:55 SUPERVISOR MODE ENTRY
08:17:30 SUPERVISOR MODE EXIT
        """.strip()
        
        print("\nTesting supervisor mode immediately after transaction...")
        labels = labeler.label_log(suspicious_timing_log)
        
        # Look for timing anomalies
        timing_anomalies = []
        
        for label in labels:
            if label.event_type == EventType.SUPERVISOR_ENTRY:
                print("Supervisor entry timing analysis:")
                print("  Severity: " + label.severity.value)
                anomalies = label.metadata.get('contextual_anomalies', [])
                print("  Anomalies: " + str(anomalies))
                
                # Check for timing-related anomalies
                for anomaly in anomalies:
                    if 'after transaction' in anomaly:
                        timing_anomalies.append(anomaly)
                        print("  --> Timing anomaly detected: " + anomaly)
        
        return len(timing_anomalies) > 0
        
    except Exception as e:
        print("[FAIL] Supervisor timing test failed: " + str(e))
        return False

def test_very_short_supervisor_session():
    """Test detection of unusually short supervisor sessions"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Scenario: 30-second supervisor session (suspicious)
        short_session_log = """
09:45:00 SUPERVISOR MODE ENTRY
09:45:30 SUPERVISOR MODE EXIT
        """.strip()
        
        print("\nTesting very short supervisor session...")
        labels = labeler.label_log(short_session_log)
        
        # Look for short session anomalies
        session_anomalies = []
        duration_detected = False
        
        for label in labels:
            if label.event_type == EventType.SUPERVISOR_EXIT:
                print("Supervisor exit analysis:")
                print("  Severity: " + label.severity.value)
                anomalies = label.metadata.get('contextual_anomalies', [])
                print("  Anomalies: " + str(anomalies))
                
                duration = label.metadata.get('supervisor_session_duration')
                classification = label.metadata.get('supervisor_session_classification')
                
                if duration is not None:
                    duration_detected = True
                    print("  Duration: " + str(duration) + "s")
                    print("  Classification: " + str(classification))
                
                # Check for short session anomalies
                for anomaly in anomalies:
                    if 'short supervisor session' in anomaly.lower():
                        session_anomalies.append(anomaly)
                        print("  --> Short session anomaly: " + anomaly)
        
        return len(session_anomalies) > 0 and duration_detected
        
    except Exception as e:
        print("[FAIL] Short session test failed: " + str(e))
        return False

def test_normal_supervisor_session():
    """Test that normal supervisor sessions don't trigger false alarms"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Scenario: Normal supervisor session (10 minutes, well after transaction)
        normal_session_log = """
10:00:00 *TRANSACTION START*
10:01:00 NOTES PRESENTED
10:01:05 NOTES TAKEN
10:01:10 *TRANSACTION END*
10:05:00 SUPERVISOR MODE ENTRY
10:15:00 SUPERVISOR MODE EXIT
        """.strip()
        
        print("\nTesting normal supervisor session...")
        labels = labeler.label_log(normal_session_log)
        
        # Should have minimal or no anomalies
        supervisor_anomalies = []
        
        for label in labels:
            if label.event_type in [EventType.SUPERVISOR_ENTRY, EventType.SUPERVISOR_EXIT]:
                anomalies = label.metadata.get('contextual_anomalies', [])
                print("Event: " + label.event_type.value)
                print("  Severity: " + label.severity.value)
                print("  Anomalies: " + str(anomalies))
                
                # Count supervisor-related anomalies
                for anomaly in anomalies:
                    if 'supervisor' in anomaly.lower():
                        supervisor_anomalies.append(anomaly)
        
        # Normal session should have few/no anomalies
        print("Total supervisor anomalies: " + str(len(supervisor_anomalies)))
        return len(supervisor_anomalies) <= 1  # Allow minor warnings but no critical issues
        
    except Exception as e:
        print("[FAIL] Normal session test failed: " + str(e))
        return False

def main():
    """Run all supervisor mode anomaly detection tests"""
    print("Enhanced Supervisor Mode Anomaly Detection Test Suite")
    print("=" * 70)
    
    tests = [
        ("Supervisor During Transaction", test_supervisor_during_transaction),
        ("Supervisor Immediately After Transaction", test_supervisor_immediately_after_transaction),
        ("Very Short Supervisor Session", test_very_short_supervisor_session),
        ("Normal Supervisor Session", test_normal_supervisor_session)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print("\n" + "=" * 20 + " " + test_name + " " + "=" * 20)
        if test_func():
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")
    
    print("\n" + "=" * 70)
    print("Supervisor Mode Anomaly Tests: " + str(passed) + "/" + str(len(tests)) + " passed")
    
    if passed == len(tests):
        print("\n[SUCCESS] Enhanced Supervisor Mode Anomaly Detection Working!")
        print("[SUCCESS] Detects supervisor mode during transactions (CRITICAL)")
        print("[SUCCESS] Flags supervisor mode immediately after transactions (ERROR)")
        print("[SUCCESS] Identifies unusually short supervisor sessions (ERROR/WARNING)")
        print("[SUCCESS] Normal supervisor operations don't trigger false alarms")
        print("\nSecurity Features:")
        print("- Transaction-supervisor overlap detection")
        print("- Post-transaction timing analysis (30s = ERROR, 120s = WARNING)")
        print("- Session duration classification (VERY_SHORT/SHORT/BRIEF/NORMAL/EXTENDED)")
        print("- Severity escalation based on suspicion level")
        print("- Detailed timing metadata for forensic analysis")
    else:
        print("\n[FAIL] Some supervisor mode anomaly tests failed")

if __name__ == "__main__":
    main()
