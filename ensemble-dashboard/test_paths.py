#!/usr/bin/env python3
"""
Simple test to check current working directory and paths
"""

import os
import glob

def test_paths():
    print("Current working directory:", os.getcwd())
    print()
    
    possible_data_dirs = [
        "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/processed",
        "/data/processed",
        "../../abm-anomaly-ml-first/data/processed",
        "../../../abm-anomaly-ml-first/data/processed", 
        "./data/processed"
    ]
    
    print("Testing data directory paths:")
    for i, path in enumerate(possible_data_dirs, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"{i}. {path}")
        print(f"   -> {abs_path}")
        print(f"   -> EXISTS: {exists}")
        
        if exists:
            pattern = os.path.join(path, "normal_sessions_full_*.json")
            files = glob.glob(pattern)
            print(f"   -> Files matching pattern: {len(files)}")
            for file in files[:2]:  # Show first 2
                print(f"      {os.path.basename(file)}")
        print()

if __name__ == "__main__":
    test_paths()
