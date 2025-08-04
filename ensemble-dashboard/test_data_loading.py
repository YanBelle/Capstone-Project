#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to debug data loading for the ensemble dashboard
"""

import os
import json
import glob
import base64

def test_data_loading():
    print("Testing Data Loading for Ensemble Dashboard")
    print("=" * 60)
    
    # Check possible data directories
    possible_data_dirs = [
        "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/processed",
        "/data/processed",
        "../abm-anomaly-ml-first/data/processed", 
        "./data/processed"
    ]
    
    print("Checking data directories:")
    data_dir = None
    for i, dir_path in enumerate(possible_data_dirs, 1):
        exists = os.path.exists(dir_path)
        status = 'EXISTS' if exists else 'NOT FOUND'
        print("  {}. {} - {}".format(i, dir_path, status))
        if exists and data_dir is None:
            data_dir = dir_path
    
    if not data_dir:
        print("\nNo data directory found!")
        return False
    
    print(f"\nUsing data directory: {data_dir}")
    
    # Check for normal session files
    normal_pattern = os.path.join(data_dir, "normal_sessions_full_*.json")
    normal_files = glob.glob(normal_pattern)
    
    print(f"\nLooking for pattern: {normal_pattern}")
    print(f"Found {len(normal_files)} normal session files:")
    
    for file in normal_files:
        size = os.path.getsize(file) / 1024  # KB
        print(f"  {os.path.basename(file)} ({size:.1f} KB)")
    
    if not normal_files:
        print("No normal session files found!")
        
        # List all files in directory for debugging
        print(f"\nAll files in {data_dir}:")
        try:
            all_files = os.listdir(data_dir)
            for file in sorted(all_files):
                print(f"  {file}")
        except Exception as e:
            print(f"  Error listing files: {e}")
        return False
    
    # Try to load the latest file
    latest_normal_file = max(normal_files, key=os.path.getctime)
    print(f"\nLoading latest file: {os.path.basename(latest_normal_file)}")
    
    try:
        with open(latest_normal_file, 'r', encoding='utf-8') as f:
            normal_sessions = json.load(f)
        
        print(f"Successfully loaded {len(normal_sessions)} sessions")
        
        # Test decoding a few sessions
        decoded_sessions = []
        for i, session in enumerate(normal_sessions[:3]):  # Test first 3
            try:
                raw_text_b64 = session.get('raw_text_base64', '')
                if raw_text_b64:
                    session_text = base64.b64decode(raw_text_b64).decode('utf-8')
                    decoded_sessions.append(session_text)
                    print(f"  Session {i+1}: {len(session_text)} characters")
                    # Show first few lines
                    lines = session_text.strip().split('\n')[:3]
                    for line in lines:
                        print(f"    {line[:80]}...")
                    print()
                else:
                    print(f"  Session {i+1}: No raw_text_base64 field")
            except Exception as e:
                print(f"  Session {i+1}: Error decoding - {e}")
        
        print(f"Successfully decoded {len(decoded_sessions)} test sessions")
        
        # Check for error sessions too
        error_pattern = os.path.join(data_dir, "error_sessions_full_*.json")
        error_files = glob.glob(error_pattern)
        print(f"\nFound {len(error_files)} error session files")
        
        return True
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    print(f"\n{'DATA LOADING TEST PASSED' if success else 'DATA LOADING TEST FAILED'}")
