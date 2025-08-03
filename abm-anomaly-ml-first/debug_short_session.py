#!/usr/bin/env python3
"""Debug supervisor mode short session detection"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def debug_short_session():
    """Debug very short supervisor session detection"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, Severity
        
        labeler = EJLogLabeler()
        
        # Simple 30-second supervisor session
        short_session_log = """
09:45:00 SUPERVISOR MODE ENTRY
09:45:30 SUPERVISOR MODE EXIT
        """.strip()
        
        print("Testing very short supervisor session (30s)...")
        labels = labeler.label_log(short_session_log)
        
        print("All labels:")
        for i, label in enumerate(labels):
            print(f"  {i}: {label.event_type.value} | {label.severity.value} | {label.timestamp}")
            print(f"     Metadata: {label.metadata}")
            print()
        
        # Check supervisor mode tracking state
        print("Labeler state:")
        print(f"  supervisor_mode_start_time: {labeler.supervisor_mode_start_time}")
        print(f"  current_supervisor_mode: {labeler.current_supervisor_mode}")
        
        # Look specifically at exit event
        exit_labels = [l for l in labels if l.event_type == EventType.SUPERVISOR_EXIT]
        if exit_labels:
            exit_label = exit_labels[0]
            print("Exit event analysis:")
            print(f"  Severity: {exit_label.severity.value}")
            print(f"  Anomalies: {exit_label.metadata.get('contextual_anomalies', [])}")
            print(f"  Duration: {exit_label.metadata.get('supervisor_session_duration')}")
            print(f"  Classification: {exit_label.metadata.get('supervisor_session_classification')}")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_short_session()
