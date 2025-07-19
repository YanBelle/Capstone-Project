#!/usr/bin/env python3
"""
Batch anomaly detection test to check all sessions
"""

import os
import sys
import json
import re
from pathlib import Path

# Add the path to import our analyzer
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

def simple_anomaly_check(session_content):
    """Simple anomaly checking using our enhanced rules"""
    anomalies = []
    
    # 1. Check for excessive supervisor mode entries
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', session_content, re.IGNORECASE))
    if supervisor_entries > 5:
        confidence = min(0.95, 0.5 + (supervisor_entries / 20.0))
        severity = "high" if supervisor_entries > 10 else "medium"
        anomalies.append({
            'type': 'excessive_supervisor_mode',
            'description': f'Excessive supervisor mode entries: {supervisor_entries}',
            'confidence': confidence,
            'severity': severity
        })
    
    # 2. Check for excessive diagnostic patterns
    diagnostic_patterns = len(re.findall(r'\[000p', session_content))
    if diagnostic_patterns > 50:
        confidence = min(0.95, 0.5 + (diagnostic_patterns / 100.0))
        severity = "high" if diagnostic_patterns > 100 else "medium"
        anomalies.append({
            'type': 'excessive_diagnostics',
            'description': f'Excessive diagnostic patterns: {diagnostic_patterns}',
            'confidence': confidence,
            'severity': severity
        })
    
    # 3. Check for repetitive patterns
    lines = session_content.split('\n')
    non_timestamp_lines = [line for line in lines if not re.match(r'^\*\d+\*\d{2}/\d{2}/\d{4}\*', line)]
    
    if non_timestamp_lines:
        unique_lines = len(set(non_timestamp_lines))
        total_lines = len(non_timestamp_lines)
        repetition_ratio = (total_lines - unique_lines) / total_lines
        
        if repetition_ratio > 0.8:
            confidence = min(0.95, repetition_ratio)
            anomalies.append({
                'type': 'repetitive_patterns',
                'description': f'High repetition ratio: {repetition_ratio:.2f}',
                'confidence': confidence,
                'severity': 'high'
            })
    
    # 4. Check for unusually large sessions
    session_lines = len(session_content.split('\n'))
    if session_lines > 500:
        anomalies.append({
            'type': 'large_session',
            'description': f'Unusually large session: {session_lines} lines',
            'confidence': 0.7,
            'severity': 'medium'
        })
    
    return anomalies

def test_all_sessions():
    """Test all sessions in the AB directory"""
    sessions_dir = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB"
    
    if not os.path.exists(sessions_dir):
        print(f"❌ Sessions directory not found: {sessions_dir}")
        return
    
    session_files = [f for f in os.listdir(sessions_dir) if f.endswith('.txt')]
    print(f"🔍 Found {len(session_files)} session files")
    
    anomalous_sessions = []
    normal_sessions = []
    
    for session_file in session_files:
        session_path = os.path.join(sessions_dir, session_file)
        
        try:
            with open(session_path, 'r', encoding='utf-8', errors='ignore') as f:
                session_content = f.read()
            
            anomalies = simple_anomaly_check(session_content)
            
            if anomalies:
                anomalous_sessions.append({
                    'file': session_file,
                    'anomalies': anomalies
                })
                print(f"🚨 {session_file}: {len(anomalies)} anomalies")
            else:
                normal_sessions.append(session_file)
                
        except Exception as e:
            print(f"❌ Error processing {session_file}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total sessions: {len(session_files)}")
    print(f"   Anomalous sessions: {len(anomalous_sessions)}")
    print(f"   Normal sessions: {len(normal_sessions)}")
    
    # Check if our target session is flagged
    target_session = "ABM250_20250618_SESSION_357_e7d058b7_20250705_172012.txt"
    target_found = False
    
    for anomalous_session in anomalous_sessions:
        if anomalous_session['file'] == target_session:
            target_found = True
            print(f"\n✅ TARGET SESSION DETECTED AS ANOMALOUS:")
            print(f"   File: {target_session}")
            for anomaly in anomalous_session['anomalies']:
                print(f"   - {anomaly['type']}: {anomaly['description']} (confidence: {anomaly['confidence']:.2f})")
            break
    
    if not target_found:
        print(f"\n❌ TARGET SESSION NOT DETECTED AS ANOMALOUS: {target_session}")
    
    # Show top 10 anomalous sessions
    print(f"\n🔍 TOP 10 ANOMALOUS SESSIONS:")
    for i, session in enumerate(anomalous_sessions[:10]):
        print(f"   {i+1}. {session['file']} - {len(session['anomalies'])} anomalies")

if __name__ == "__main__":
    test_all_sessions()
