#!/usr/bin/env python3

"""
Debug script to identify where add_anomaly is being called incorrectly
"""

import sys
import os
sys.path.append('/app')

from services.anomaly_detector.ml_analyzer import TransactionSession

def test_add_anomaly():
    """Test add_anomaly calls to find the issue"""
    
    print("Testing add_anomaly method...")
    
    # Create a test session
    session = TransactionSession(
        session_id="test_session",
        raw_text="test data",
        timestamp_start="2025-01-01 00:00:00",
        timestamp_end="2025-01-01 00:01:00"
    )
    
    # Test correct call
    try:
        session.add_anomaly(
            anomaly_type="test_anomaly",
            confidence=0.8,
            detection_method="test_method",
            description="Test description"
        )
        print("✓ Correct add_anomaly call works")
    except Exception as e:
        print(f"✗ Correct add_anomaly call failed: {e}")
    
    # Test incorrect call that might be happening
    try:
        session.add_anomaly("test_anomaly_only")
        print("✗ Incorrect add_anomaly call should have failed but didn't")
    except Exception as e:
        print(f"✓ Incorrect add_anomaly call failed as expected: {e}")

if __name__ == "__main__":
    test_add_anomaly()
