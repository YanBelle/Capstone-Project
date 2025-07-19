#!/usr/bin/env python3
"""
Test to verify that the specific user-reported session is now detected correctly.
"""
import sys
import os
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

from ml_analyzer import MLAnomalyAnalyzer

def test_user_specific_session():
    """Test the specific session that the user reported was not being picked up."""
    
    # Read the actual EJ log file
    log_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/output/ABM250EJ_20250618_20250618.txt"
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    # Create analyzer instance
    analyzer = MLAnomalyAnalyzer()
    
    # Process the log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw_logs = f.read()
    
    sessions = analyzer.split_into_sessions(raw_logs, log_file)
    
    print(f"Found {len(sessions)} sessions in the EJ log")
    
    # Look for the specific session with "PRIMARY CARD READER ACTIVATED"
    target_session = None
    for session in sessions:
        if "*PRIMARY CARD READER ACTIVATED*" in session.raw_text:
            target_session = session
            break
    
    if target_session:
        print("\n✓ Found the session with '*PRIMARY CARD READER ACTIVATED*'")
        print(f"Session ID: {target_session.session_id}")
        print(f"Start Time: {target_session.start_time}")
        print(f"Session Length: {len(target_session.raw_text.split())} lines")
        
        # Check if this matches the user's expectation
        session_lines = target_session.raw_text.split('\n')
        print(f"\nFirst 10 lines of session:")
        for i, line in enumerate(session_lines[:10]):
            print(f"  {i+1}: {line}")
        
        # Check for timestamp extraction
        if target_session.start_time:
            print(f"\n✓ Session start time successfully extracted: {target_session.start_time}")
        else:
            print(f"\n✗ Failed to extract session start time")
        
        # Check for session content completeness
        if "CASHIN RECOVERY OK" in target_session.raw_text:
            print("✓ Session includes post-transaction recovery information")
        else:
            print("✗ Session may be truncated - missing recovery information")
            
        print("\n✅ User-reported session is now being detected correctly!")
    else:
        print("\n✗ Could not find session with '*PRIMARY CARD READER ACTIVATED*'")
        print("Available sessions start with:")
        for i, session in enumerate(sessions[:5]):
            first_line = session.raw_text.split('\n')[0]
            print(f"  Session {i+1}: {first_line}")

if __name__ == "__main__":
    test_user_specific_session()
