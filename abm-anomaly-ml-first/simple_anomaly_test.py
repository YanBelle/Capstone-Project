#!/usr/bin/env python3
"""
Simple test to verify anomaly detection on the problematic session
"""

import sys
import os
import re

# Add the path to import our analyzer
sys.path.append('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/services/anomaly-detector')

try:
    from ml_analyzer import MLAnalyzer
    print("✅ Successfully imported MLAnalyzer")
except Exception as e:
    print(f"❌ Failed to import MLAnalyzer: {e}")
    sys.exit(1)

def test_single_session():
    """Test anomaly detection on a single session"""
    session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_172012.txt"
    
    print(f"🔍 Testing session: {session_file}")
    
    # Read the session file
    try:
        with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
            session_content = f.read()
        print(f"📄 Session loaded: {len(session_content)} characters")
    except Exception as e:
        print(f"❌ Failed to read session: {e}")
        return
    
    # Count supervisor mode entries manually
    supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', session_content, re.IGNORECASE))
    print(f"🔒 Supervisor mode entries: {supervisor_entries}")
    
    # Count diagnostic patterns
    diagnostic_patterns = len(re.findall(r'\[000p', session_content))
    print(f"🔧 Diagnostic patterns: {diagnostic_patterns}")
    
    # Initialize analyzer
    try:
        analyzer = MLAnalyzer()
        print("✅ MLAnalyzer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MLAnalyzer: {e}")
        return
    
    # Test the session
    try:
        result = analyzer.analyze_session(session_content)
        print(f"📊 Analysis result: {result}")
        
        if result and 'anomalies' in result:
            anomaly_count = len(result['anomalies'])
            print(f"🚨 Found {anomaly_count} anomalies:")
            for anomaly in result['anomalies']:
                print(f"   - {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        else:
            print("❌ No anomalies detected!")
            
    except Exception as e:
        print(f"❌ Failed to analyze session: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_session()
