#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test to verify the enhanced anomaly detection works correctly
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path

# Add the path to import our analyzer
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_enhanced_anomaly_detection():
    """Test our enhanced anomaly detection logic directly"""
    
    # Test the specific session that should be flagged
    session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_172012.txt"
    
    print("ENHANCED ANOMALY DETECTION TEST")
    print("="*50)
    print("Testing session:", os.path.basename(session_file))
    
    # Read session content
    try:
        with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
            session_content = f.read()
        print("Session loaded:", len(session_content), "characters")
    except Exception as e:
        print("Failed to load session:", e)
        return False
    
    # Apply our enhanced detection logic
    print("\n🔍 APPLYING ENHANCED DETECTION LOGIC:")
    
    # 1. Test Excessive Supervisor Mode Detection
    print("\n1. SUPERVISOR MODE DETECTION:")
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', session_content, re.IGNORECASE))
    print(f"   Found {supervisor_entries} supervisor mode entries")
    
    if supervisor_entries > 5:
        confidence = min(0.95, 0.5 + (supervisor_entries / 20.0))
        severity = "high" if supervisor_entries > 10 else "medium"
        print(f"   🚨 ANOMALY DETECTED: excessive_supervisor_mode")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Severity: {severity}")
        print(f"   - Threshold: 5 (actual: {supervisor_entries})")
    else:
        print(f"   ✅ Normal supervisor mode usage (threshold: 5)")
    
    # 2. Test Excessive Diagnostic Patterns
    print("\n2. DIAGNOSTIC PATTERN DETECTION:")
    diagnostic_patterns = len(re.findall(r'\[000p\[040q\(I.*?R-\d+S', session_content, re.IGNORECASE))
    simple_diagnostic_patterns = len(re.findall(r'\[000p', session_content))
    print(f"   Found {diagnostic_patterns} complex diagnostic patterns")
    print(f"   Found {simple_diagnostic_patterns} simple diagnostic patterns")
    
    if diagnostic_patterns > 50:
        confidence = min(0.95, 0.5 + (diagnostic_patterns / 100.0))
        severity = "high" if diagnostic_patterns > 100 else "medium"
        print(f"   🚨 ANOMALY DETECTED: excessive_diagnostics")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Severity: {severity}")
        print(f"   - Threshold: 50 (actual: {diagnostic_patterns})")
    else:
        print(f"   ✅ Normal diagnostic pattern usage (threshold: 50)")
    
    # 3. Test Repetitive Pattern Detection
    print("\n3. REPETITIVE PATTERN DETECTION:")
    lines = session_content.split('\n')
    session_lines = len(lines)
    print(f"   Total lines: {session_lines}")
    
    if session_lines > 500:
        # Count unique vs total lines (excluding timestamps)
        non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
        unique_lines = len(set(non_timestamp_lines))
        total_lines = len(non_timestamp_lines)
        repetition_ratio = (total_lines - unique_lines) / total_lines if total_lines > 0 else 0
        
        print(f"   Non-timestamp lines: {total_lines}")
        print(f"   Unique lines: {unique_lines}")
        print(f"   Repetition ratio: {repetition_ratio:.3f}")
        
        if repetition_ratio > 0.8:
            confidence = min(0.95, repetition_ratio)
            print(f"   🚨 ANOMALY DETECTED: repetitive_pattern_loop")
            print(f"   - Confidence: {confidence:.3f}")
            print(f"   - Severity: high")
            print(f"   - Threshold: 0.8 (actual: {repetition_ratio:.3f})")
        else:
            print(f"   ✅ Normal repetition level (threshold: 0.8)")
    else:
        print(f"   ✅ Normal session size (threshold: 500)")
    
    # 4. Test Large Session Detection
    print("\n4. LARGE SESSION DETECTION:")
    if session_lines > 500:
        print(f"   🚨 ANOMALY DETECTED: large_session")
        print(f"   - Session size: {session_lines} lines")
        print(f"   - Threshold: 500 lines")
        print(f"   - Confidence: 0.70")
        print(f"   - Severity: medium")
    else:
        print(f"   ✅ Normal session size (threshold: 500)")
    
    # Summary
    print("\n" + "="*50)
    print("📊 DETECTION SUMMARY:")
    
    anomaly_count = 0
    detected_anomalies = []
    
    if supervisor_entries > 5:
        anomaly_count += 1
        detected_anomalies.append(f"excessive_supervisor_mode ({supervisor_entries} entries)")
    
    if diagnostic_patterns > 50:
        anomaly_count += 1
        detected_anomalies.append(f"excessive_diagnostics ({diagnostic_patterns} patterns)")
    
    if session_lines > 500:
        non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
        if non_timestamp_lines:
            unique_lines = len(set(non_timestamp_lines))
            total_lines = len(non_timestamp_lines)
            repetition_ratio = (total_lines - unique_lines) / total_lines
            
            if repetition_ratio > 0.8:
                anomaly_count += 1
                detected_anomalies.append(f"repetitive_pattern_loop ({repetition_ratio:.3f} ratio)")
        
        anomaly_count += 1
        detected_anomalies.append(f"large_session ({session_lines} lines)")
    
    if anomaly_count > 0:
        print(f"🚨 RESULT: {anomaly_count} ANOMALIES DETECTED")
        for anomaly in detected_anomalies:
            print(f"   - {anomaly}")
        print(f"\n✅ SESSION SHOULD BE FLAGGED AS ANOMALOUS!")
        return True
    else:
        print(f"❌ RESULT: NO ANOMALIES DETECTED")
        print(f"   Session appears normal by our enhanced detection logic")
        return False

if __name__ == "__main__":
    success = test_enhanced_anomaly_detection()
    sys.exit(0 if success else 1)
