#!/usr/bin/env python3
import re

# Test session content
session_file = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/sessions/AB/ABM250_20250618_SESSION_357_e7d058b7_20250705_172012.txt"

with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Test our detection logic
supervisor_entries = len(re.findall(r'SUPERVISOR MODE ENTRY', content, re.IGNORECASE))
diagnostic_patterns = len(re.findall(r'\[000p', content))
lines = len(content.split('\n'))

print(f"Supervisor entries: {supervisor_entries}")
print(f"Diagnostic patterns: {diagnostic_patterns}")
print(f"Total lines: {lines}")

# Test thresholds
if supervisor_entries > 5:
    print("ANOMALY: Excessive supervisor mode")
if diagnostic_patterns > 50:
    print("ANOMALY: Excessive diagnostics")
if lines > 500:
    print("ANOMALY: Large session")

print("Test completed")
