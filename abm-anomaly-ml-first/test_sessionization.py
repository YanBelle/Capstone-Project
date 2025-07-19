#!/usr/bin/env python3
"""
Test script to verify the updated sessionization logic.
This script specifically tests that session start times are captured 
from the line immediately above the TRANSACTION START marker.
"""
import sys
import os
sys.path.append('/app')

from ml_analyzer import MLAnomalyAnalyzer

def test_sessionization_with_timestamp():
    """Test that session start times are captured from the line above TRANSACTION START"""
    
    # Sample log content that mimics the structure from the user's example
    sample_log = """[020t*629*06/18/2025*00:46*
     *TRANSACTION START*
[020t CARD INSERTED
 00:46:27 ATR RECEIVED T=0
[020t 00:46:30 OPCODE = FI      
  PAN 0004263********1897
  ---START OF TRANSACTION---
[020t 00:46:42 PIN ENTERED
[020t 00:46:47 OPCODE = IB      
  PAN 0004263********1897
  ---START OF TRANSACTION---
*630*06/18/2025*00:46*
*7231*1*(Iw(1*3, M-02, R-10011
A/C 
DEVICE ERROR
ESC: 000
VAL: 000
REF: 000
REJECTS:000*(1
S
[020t 00:47:13 CARD TAKEN
[020t 00:47:15 TRANSACTION END
[020t*631*06/18/2025*00:47*
     *PRIMARY CARD READER ACTIVATED*
[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
 04:48:38 ATR RECEIVED T=0
[020t 04:48:40 OPCODE = FI      
  PAN 0004263********2113
  ---START OF TRANSACTION---
[020t 04:48:55 PIN ENTERED
[020t 04:49:01 OPCODE = BBC     
 04:49:02 GENAC 1 : ARQC
 04:49:04 GENAC 2 : TC
[020t 04:49:11 NOTES STACKED
[020t 04:49:13 CARD TAKEN
  PAN 0004263********2113
  ---START OF TRANSACTION---
[020t 04:49:18 NOTES PRESENTED 1,0,0,0
[020t 04:49:18 NOTES TAKEN"""

    print("Testing sessionization with timestamp extraction...")
    
    # Initialize analyzer
    analyzer = MLAnomalyAnalyzer()
    
    # Split into sessions
    sessions = analyzer.split_into_sessions(sample_log, "test_ABM250EJ_20250618_20250618.txt")
    
    print(f"Found {len(sessions)} sessions")
    
    for i, session in enumerate(sessions):
        print(f"\nSession {i+1}:")
        print(f"  ID: {session.session_id}")
        print(f"  Start time: {session.start_time}")
        print(f"  End time: {session.end_time}")
        print(f"  Text preview: {session.raw_text[:100]}...")
        
        # Check if we have device errors
        if "DEVICE ERROR" in session.raw_text:
            print("  ✓ Contains DEVICE ERROR")
        if "ESC: 000" in session.raw_text:
            print("  ✓ Contains ESC error code")
        if "VAL: 000" in session.raw_text:
            print("  ✓ Contains VAL error code")
        if "REF: 000" in session.raw_text:
            print("  ✓ Contains REF error code")
        if "REJECTS:000" in session.raw_text:
            print("  ✓ Contains REJECTS error code")
            
    # Verify the specific session mentioned by the user
    expected_session = None
    for session in sessions:
        if session.start_time and session.start_time.hour == 4 and session.start_time.minute == 48:
            expected_session = session
            break
    
    if expected_session:
        print(f"\n✓ Found the expected session starting at 04:48")
        print(f"  Session ID: {expected_session.session_id}")
        print(f"  Start time: {expected_session.start_time}")
        print(f"  Contains 'PRIMARY CARD READER ACTIVATED': {'PRIMARY CARD READER ACTIVATED' in expected_session.raw_text}")
    else:
        print("\n✗ Expected session starting at 04:48 not found")
    
    return sessions

if __name__ == "__main__":
    test_sessionization_with_timestamp()
