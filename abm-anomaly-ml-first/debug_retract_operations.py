#!/usr/bin/env python3
"""Debug retract bin operation detection"""

import sys
import os

# Add the anomaly-detector path
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def debug_retract_operations():
    """Debug retract bin operation pattern matching"""
    
    try:
        from ej_contextual_labeler import EJLogLabeler, EventType, RecoveryType
        
        labeler = EJLogLabeler()
        
        # Test individual retract operations
        test_lines = [
            "INIT BNA STARTED - RETRACT BIN",
            "CASHIN RETRACT STARTED - RETRACT BIN", 
            "CIM-RESET CALLED - RETRACT BIN"
        ]
        
        print("Testing individual retract bin patterns...")
        
        for line in test_lines:
            print(f"\nTesting: '{line}'")
            labels = labeler.label_log(line)
            
            if labels:
                label = labels[0]
                print(f"  Event Type: {label.event_type.value}")
                print(f"  Recovery Type: {label.recovery_type}")
                print(f"  Severity: {label.severity.value}")
            else:
                print("  No labels generated!")
        
        # Test as a sequence
        print("\n" + "="*50)
        print("Testing complete sequence...")
        
        complete_log = """
INIT BNA STARTED - RETRACT BIN
CASHIN RETRACT STARTED - RETRACT BIN
CIM-RESET CALLED - RETRACT BIN
        """.strip()
        
        labels = labeler.label_log(complete_log)
        
        for i, label in enumerate(labels):
            print(f"{i}: {label.event_type.value} | Recovery: {label.recovery_type}")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_retract_operations()
