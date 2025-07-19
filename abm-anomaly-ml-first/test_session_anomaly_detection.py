#!/usr/bin/env python3
"""
Test script to verify advanced anomaly detection on the specific session
that was not properly flagged as an anomaly.
"""

import re
import sys
import os

# Add the path to import our analyzer
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_session_detection():
    """Test the anomaly detection on the specific session file"""
    session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_172012.txt"
    
    print("🔍 Testing Advanced Anomaly Detection")
    print("="*50)
    
    # Read the session file
    try:
        with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
            session_text = f.read()
    except FileNotFoundError:
        print(f"❌ Session file not found: {session_file}")
        return
    
    print(f"📄 Session file: {session_file}")
    print(f"📏 Session length: {len(session_text)} characters, {len(session_text.split('\n'))} lines")
    
    # Test 1: Supervisor Mode Detection
    print("\n🔒 Testing Supervisor Mode Detection:")
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', session_text, re.IGNORECASE))
    supervisor_exits = len(re.findall(r'SUPERVISOR MODE EXIT', session_text, re.IGNORECASE))
    print(f"   - Supervisor Mode Entries: {supervisor_entries}")
    print(f"   - Supervisor Mode Exits: {supervisor_exits}")
    
    if supervisor_entries > 5:
        confidence = min(0.95, 0.5 + (supervisor_entries / 20.0))
        severity = "high" if supervisor_entries > 10 else "medium"
        print(f"   ✅ ANOMALY DETECTED: Excessive supervisor mode (confidence: {confidence:.2f}, severity: {severity})")
    else:
        print(f"   ❌ No anomaly detected (threshold: 5)")
    
    # Test 2: Diagnostic Pattern Detection
    print("\n🔧 Testing Diagnostic Pattern Detection:")
    diagnostic_patterns = len(re.findall(r'\[000p\[040q\(I.*?R-\d+S', session_text, re.IGNORECASE))
    print(f"   - Diagnostic Patterns Found: {diagnostic_patterns}")
    
    if diagnostic_patterns > 50:
        confidence = min(0.95, 0.5 + (diagnostic_patterns / 100.0))
        severity = "high" if diagnostic_patterns > 100 else "medium"
        print(f"   ✅ ANOMALY DETECTED: Excessive diagnostics (confidence: {confidence:.2f}, severity: {severity})")
    else:
        print(f"   ❌ No anomaly detected (threshold: 50)")
    
    # Test 3: Repetitive Pattern Analysis
    print("\n🔄 Testing Repetitive Pattern Analysis:")
    lines = session_text.split('\n')
    non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
    
    if non_timestamp_lines:
        unique_lines = len(set(non_timestamp_lines))
        total_lines = len(non_timestamp_lines)
        repetition_ratio = (total_lines - unique_lines) / total_lines
        
        print(f"   - Total Lines: {total_lines}")
        print(f"   - Unique Lines: {unique_lines}")
        print(f"   - Repetition Ratio: {repetition_ratio:.2f}")
        
        if repetition_ratio > 0.8:
            confidence = min(0.95, repetition_ratio)
            print(f"   ✅ ANOMALY DETECTED: Repetitive patterns (confidence: {confidence:.2f}, severity: high)")
        else:
            print(f"   ❌ No anomaly detected (threshold: 0.8)")
    
    # Test 4: Session Duration/Size Analysis
    print("\n⏰ Testing Session Size Analysis:")
    session_lines = len(session_text.split('\n'))
    if session_lines > 500:
        print(f"   ✅ ANOMALY DETECTED: Unusually large session ({session_lines} lines)")
    else:
        print(f"   ❌ Normal session size ({session_lines} lines)")
    
    # Test 5: Card Insertion Pattern
    print("\n💳 Testing Transaction Pattern:")
    has_card_inserted = 'CARD INSERTED' in session_text
    has_card_taken = 'CARD TAKEN' in session_text
    has_transaction_start = 'TRANSACTION START' in session_text
    has_transaction_end = 'TRANSACTION END' in session_text
    
    print(f"   - Card Inserted: {has_card_inserted}")
    print(f"   - Card Taken: {has_card_taken}")
    print(f"   - Transaction Start: {has_transaction_start}")
    print(f"   - Transaction End: {has_transaction_end}")
    
    # Check for incomplete transaction pattern
    if has_card_inserted and has_card_taken and not any(indicator in session_text.upper() for indicator in ['PIN ENTERED', 'AUTHORIZATION', 'NOTES PRESENTED']):
        print(f"   ✅ ANOMALY DETECTED: Card inserted and removed without proper transaction flow")
    else:
        print(f"   ❌ Normal transaction pattern detected")
    
    print("\n" + "="*50)
    print("🎯 SUMMARY:")
    
    anomaly_count = 0
    if supervisor_entries > 5:
        anomaly_count += 1
        print(f"   ✅ Excessive supervisor mode entries: {supervisor_entries}")
    
    if diagnostic_patterns > 50:
        anomaly_count += 1
        print(f"   ✅ Excessive diagnostic patterns: {diagnostic_patterns}")
    
    if len(non_timestamp_lines) > 0:
        repetition_ratio = (len(non_timestamp_lines) - len(set(non_timestamp_lines))) / len(non_timestamp_lines)
        if repetition_ratio > 0.8:
            anomaly_count += 1
            print(f"   ✅ High repetition ratio: {repetition_ratio:.2f}")
    
    if session_lines > 500:
        anomaly_count += 1
        print(f"   ✅ Unusually large session: {session_lines} lines")
    
    if anomaly_count > 0:
        print(f"\n🚨 RESULT: {anomaly_count} anomalies detected - This session SHOULD be flagged as anomalous!")
    else:
        print(f"\n✅ RESULT: No anomalies detected - Session appears normal")

if __name__ == "__main__":
    test_session_detection()
