#!/usr/bin/env python3

"""
Test script to verify the labeled anomalies database connection and retraining functionality
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

def check_services():
    """Check if Docker services are running"""
    try:
        result = subprocess.run(['docker', 'compose', 'ps'], capture_output=True, text=True)
        print("Docker services status:")
        print(result.stdout)
        return "abm-anomaly-ml-first-api" in result.stdout
    except Exception as e:
        print(f"Error checking services: {e}")
        return False

def test_database_connection():
    """Test database connection and check for labeled anomalies"""
    try:
        result = subprocess.run([
            'docker', 'compose', 'exec', 'postgres', 'psql', 
            '-U', 'abm_user', '-d', 'abm_db', '-c', 
            'SELECT COUNT(*) FROM labeled_anomalies;'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Database query result: {result.stdout}")
            return True
        else:
            print(f"Database query failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

def test_retraining_api():
    """Test the retraining API endpoint"""
    try:
        result = subprocess.run([
            'curl', '-X', 'POST', 
            'http://localhost:8000/api/v1/continuous-learning/trigger-retraining',
            '-H', 'Content-Type: application/json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Retraining API response: {result.stdout}")
            return True
        else:
            print(f"Retraining API failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"API test error: {e}")
        return False

def test_continuous_learning_status():
    """Test the continuous learning status endpoint"""
    try:
        result = subprocess.run([
            'curl', '-X', 'GET', 
            'http://localhost:8000/api/v1/continuous-learning/status',
            '-H', 'Content-Type: application/json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Continuous learning status: {result.stdout}")
            return True
        else:
            print(f"Status API failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Status API error: {e}")
        return False

def main():
    print("=== Testing Updated Retraining System ===")
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Change to the project directory
    os.chdir('/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first')
    
    # 1. Check if services are running
    print("1. Checking Docker services...")
    if not check_services():
        print("   Services not running, starting them...")
        try:
            subprocess.run(['docker', 'compose', 'up', '-d'], check=True)
            print("   Services started, waiting 30 seconds for startup...")
            time.sleep(30)
        except Exception as e:
            print(f"   Failed to start services: {e}")
            return
    else:
        print("   Services are running!")
    print()
    
    # 2. Test database connection
    print("2. Testing database connection...")
    if test_database_connection():
        print("   Database connection successful!")
    else:
        print("   Database connection failed!")
    print()
    
    # 3. Test continuous learning status
    print("3. Testing continuous learning status...")
    if test_continuous_learning_status():
        print("   Status API working!")
    else:
        print("   Status API failed!")
    print()
    
    # 4. Test retraining API
    print("4. Testing retraining API...")
    if test_retraining_api():
        print("   Retraining API working!")
    else:
        print("   Retraining API failed!")
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
