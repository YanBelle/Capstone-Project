#!/usr/bin/env python3

import os
import sys

def test_basic():
    print("Python test script is running")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if the file exists
    file_path = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/ABM250EJ_20250618_20250618.txt"
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
        # Read first few lines
        with open(file_path, 'r') as f:
            lines = f.readlines()[:5]
            print(f"First 5 lines of file:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line.strip()}")
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    test_basic()
