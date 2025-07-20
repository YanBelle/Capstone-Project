#!/usr/bin/env python3
"""
Direct session analysis to force anomaly detection on our target session
"""

import sys
import os
import json
from datetime import datetime

# Test our specific session directly
session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_185210.txt"

def analyze_session_directly():
    """Analyze the session directly with our enhanced logic"""
    
    print("🔍 DIRECT SESSION ANALYSIS")
    print("="*60)
    
    # Read session
    with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Session: {os.path.basename(session_file)}")
    print(f"Size: {len(content)} characters, {len(content.split('\n'))} lines")
    
    # Apply enhanced detection logic
    import re
    
    anomalies = []
    
    # 1. Supervisor Mode Detection
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', content, re.IGNORECASE))
    if supervisor_entries > 5:
        confidence = min(0.95, 0.5 + (supervisor_entries / 20.0))
        severity = "high" if supervisor_entries > 10 else "medium"
        anomalies.append({
            'type': 'excessive_supervisor_mode',
            'description': f'Excessive supervisor mode entries: {supervisor_entries} times',
            'confidence': confidence,
            'severity': severity,
            'detection_method': 'expert_rule',
            'details': {'supervisor_entries': supervisor_entries, 'threshold': 5}
        })
    
    # 2. Diagnostic Pattern Detection (fixed regex)
    diagnostic_patterns = len(re.findall(r'\[000p', content, re.IGNORECASE))
    if diagnostic_patterns > 50:
        confidence = min(0.95, 0.5 + (diagnostic_patterns / 100.0))
        severity = "high" if diagnostic_patterns > 100 else "medium"
        anomalies.append({
            'type': 'excessive_diagnostics',
            'description': f'Excessive diagnostic messages: {diagnostic_patterns} patterns',
            'confidence': confidence,
            'severity': severity,
            'detection_method': 'expert_rule',
            'details': {'diagnostic_patterns': diagnostic_patterns, 'threshold': 50}
        })
    
    # 3. Large Session Detection
    lines = content.split('\n')
    session_lines = len(lines)
    if session_lines > 500:
        anomalies.append({
            'type': 'large_session',
            'description': f'Unusually large session: {session_lines} lines',
            'confidence': 0.70,
            'severity': 'medium',
            'detection_method': 'expert_rule',
            'details': {'session_lines': session_lines, 'threshold': 500}
        })
    
    # 4. Repetitive Pattern Detection
    if session_lines > 500:
        non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
        if non_timestamp_lines:
            unique_lines = len(set(non_timestamp_lines))
            total_lines = len(non_timestamp_lines)
            repetition_ratio = (total_lines - unique_lines) / total_lines
            
            if repetition_ratio > 0.8:
                confidence = min(0.95, repetition_ratio)
                anomalies.append({
                    'type': 'repetitive_pattern_loop',
                    'description': f'High repetition ratio: {repetition_ratio:.2f} ({total_lines} lines, {unique_lines} unique)',
                    'confidence': confidence,
                    'severity': 'high',
                    'detection_method': 'expert_rule',
                    'details': {
                        'repetition_ratio': repetition_ratio,
                        'total_lines': total_lines,
                        'unique_lines': unique_lines
                    }
                })
    
    # Create anomaly report
    report = {
        'session_id': os.path.basename(session_file).replace('.txt', ''),
        'timestamp': datetime.now().isoformat(),
        'total_anomalies': len(anomalies),
        'anomalies': anomalies,
        'session_stats': {
            'total_characters': len(content),
            'total_lines': session_lines,
            'supervisor_entries': supervisor_entries,
            'diagnostic_patterns': diagnostic_patterns
        }
    }
    
    # Save report
    report_file = f"/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/output/manual_anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   Total anomalies detected: {len(anomalies)}")
    
    for i, anomaly in enumerate(anomalies, 1):
        print(f"   {i}. {anomaly['type']}")
        print(f"      - {anomaly['description']}")
        print(f"      - Confidence: {anomaly['confidence']:.3f}")
        print(f"      - Severity: {anomaly['severity']}")
    
    if anomalies:
        print(f"\n✅ SESSION IS ANOMALOUS!")
        print(f"📄 Report saved: {report_file}")
    else:
        print(f"\n❌ NO ANOMALIES DETECTED")
    
    return len(anomalies) > 0

if __name__ == "__main__":
    success = analyze_session_directly()
    print(f"\n🎯 Analysis {'PASSED' if success else 'FAILED'}")
