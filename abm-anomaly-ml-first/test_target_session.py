#!/usr/bin/env python3
"""
Direct test of the specific session that should be flagged as anomalous
"""

import sys
import os
import re
from pathlib import Path

# Add paths
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def test_specific_session():
    """Test the specific session that should be anomalous"""
    
    session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_181730.txt"
    
    print("=" * 60)
    print("🔍 TESTING SPECIFIC SESSION FOR ANOMALY DETECTION")
    print("=" * 60)
    print(f"Session: {os.path.basename(session_file)}")
    
    # Read the session
    try:
        with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"✅ Session loaded: {len(content)} characters, {len(content.split('\n'))} lines")
    except Exception as e:
        print(f"❌ Error reading session: {e}")
        return False
    
    # Manual anomaly detection using our logic
    print("\n" + "=" * 60)
    print("🔍 MANUAL ANOMALY DETECTION")
    print("=" * 60)
    
    anomalies_found = []
    
    # 1. Supervisor Mode Check
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', content, re.IGNORECASE))
    print(f"1. Supervisor Mode Entries: {supervisor_entries} (threshold: 5)")
    if supervisor_entries > 5:
        confidence = min(0.95, 0.5 + (supervisor_entries / 20.0))
        severity = "high" if supervisor_entries > 10 else "medium"
        anomalies_found.append({
            'type': 'excessive_supervisor_mode',
            'count': supervisor_entries,
            'confidence': confidence,
            'severity': severity
        })
        print(f"   🚨 ANOMALY: excessive_supervisor_mode (confidence: {confidence:.3f}, severity: {severity})")
    
    # 2. Diagnostic Pattern Check
    diagnostic_patterns = len(re.findall(r'\[000p\[040q\(I.*?R-\d+S', content, re.IGNORECASE))
    simple_diagnostic = len(re.findall(r'\[000p', content))
    print(f"2. Diagnostic Patterns: {diagnostic_patterns} complex, {simple_diagnostic} simple (threshold: 50)")
    if diagnostic_patterns > 50:
        confidence = min(0.95, 0.5 + (diagnostic_patterns / 100.0))
        severity = "high" if diagnostic_patterns > 100 else "medium"
        anomalies_found.append({
            'type': 'excessive_diagnostics',
            'count': diagnostic_patterns,
            'confidence': confidence,
            'severity': severity
        })
        print(f"   🚨 ANOMALY: excessive_diagnostics (confidence: {confidence:.3f}, severity: {severity})")
    
    # 3. Large Session Check
    lines = content.split('\n')
    line_count = len(lines)
    print(f"3. Session Size: {line_count} lines (threshold: 500)")
    if line_count > 500:
        anomalies_found.append({
            'type': 'large_session',
            'count': line_count,
            'confidence': 0.70,
            'severity': 'medium'
        })
        print(f"   🚨 ANOMALY: large_session (confidence: 0.70, severity: medium)")
    
    # 4. Repetitive Pattern Check
    non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
    if non_timestamp_lines:
        unique_lines = len(set(non_timestamp_lines))
        total_lines = len(non_timestamp_lines)
        repetition_ratio = (total_lines - unique_lines) / total_lines
        print(f"4. Repetitive Patterns: {repetition_ratio:.3f} ratio (threshold: 0.8)")
        if repetition_ratio > 0.8:
            confidence = min(0.95, repetition_ratio)
            anomalies_found.append({
                'type': 'repetitive_pattern_loop',
                'ratio': repetition_ratio,
                'confidence': confidence,
                'severity': 'high'
            })
            print(f"   🚨 ANOMALY: repetitive_pattern_loop (confidence: {confidence:.3f}, severity: high)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    
    if anomalies_found:
        print(f"🚨 {len(anomalies_found)} ANOMALIES DETECTED:")
        for i, anomaly in enumerate(anomalies_found, 1):
            print(f"   {i}. {anomaly['type']}")
            if 'count' in anomaly:
                print(f"      Count: {anomaly['count']}")
            if 'ratio' in anomaly:
                print(f"      Ratio: {anomaly['ratio']:.3f}")
            print(f"      Confidence: {anomaly['confidence']:.3f}")
            print(f"      Severity: {anomaly['severity']}")
        
        print(f"\n✅ SESSION SHOULD BE FLAGGED AS ANOMALOUS!")
        return True
    else:
        print("❌ NO ANOMALIES DETECTED")
        print("   This suggests our detection logic may not be working correctly")
        return False

if __name__ == "__main__":
    success = test_specific_session()
    if success:
        print("\n🎯 TEST PASSED: Session correctly identified as anomalous")
    else:
        print("\n❌ TEST FAILED: Session not identified as anomalous")
    
    # Now test if the actual ML analyzer works
    print("\n" + "=" * 60)
    print("🔧 TESTING ACTUAL ML ANALYZER")
    print("=" * 60)
    
    try:
        from ml_analyzer import MLAnalyzer
        print("✅ MLAnalyzer imported successfully")
        
        # Test the actual analyzer
        analyzer = MLAnalyzer()
        print("✅ MLAnalyzer initialized successfully")
        
        # Read the session file
        session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_181730.txt"
        with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
            session_content = f.read()
        
        # Run the analyzer
        result = analyzer.analyze_session(session_content)
        
        if result and 'anomalies' in result:
            print(f"✅ ML Analyzer found {len(result['anomalies'])} anomalies:")
            for anomaly in result['anomalies']:
                print(f"   - {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        else:
            print("❌ ML Analyzer found no anomalies")
            print("   This indicates the ML analyzer is not detecting the anomalies our manual logic found")
            
    except Exception as e:
        print(f"❌ Error testing ML Analyzer: {e}")
        import traceback
        traceback.print_exc()
