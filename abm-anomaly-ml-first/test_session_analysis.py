#!/usr/bin/env python3
import sys
import os
sys.path.append('services/anomaly-detector')

from deeplog_analyzer import DeepLogAnalyzer

def test_session_events(session_id):
    """Test event extraction for a specific session"""
    file_path = f'data/sessions/SESSION_{session_id}.txt'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    analyzer = DeepLogAnalyzer()
    events = analyzer.extract_log_events(content)
    
    print(f"\n=== SESSION_{session_id} Analysis ===")
    print(f"Extracted Events: {events}")
    
    # Test completeness check
    try:
        is_complete, score, missing = analyzer.check_transaction_completeness(events)
        print(f"Is Complete: {is_complete}")
        print(f"Completeness Score: {score:.3f}")
        print(f"Missing/Issues: {missing}")
    except Exception as e:
        print(f"Error in completeness check: {e}")
    
    # Show some raw content for debugging
    print(f"\nFirst 500 chars of session content:")
    print(content[:500])
    print("...")

if __name__ == "__main__":
    # Test the problematic sessions
    for session_id in ['282', '305', '357']:
        test_session_events(session_id)
