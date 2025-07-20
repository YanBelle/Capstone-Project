#!/usr/bin/env python3

"""
Test script to verify database schema fixes and retraining functionality
"""

import subprocess
import time
import json

def run_sql_query(query):
    """Execute a SQL query in the database"""
    try:
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'postgres', 'psql', 
            '-U', 'abm_user', '-d', 'abm_db', '-c', query
        ], capture_output=True, text=True, cwd='/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first')
        
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def test_database_schema():
    """Test if database schema issues are fixed"""
    print("=== Testing Database Schema Fixes ===")
    
    # Test 1: Check if last_activity column exists
    print("\n1. Checking if last_activity column exists...")
    stdout, stderr, code = run_sql_query("SELECT column_name FROM information_schema.columns WHERE table_name = 'ml_sessions' AND column_name = 'last_activity';")
    if code == 0 and 'last_activity' in stdout:
        print("   ✅ last_activity column exists")
    else:
        print("   ❌ last_activity column missing")
        print(f"   Error: {stderr}")
    
    # Test 2: Check if anomaly_sessions table exists
    print("\n2. Checking if anomaly_sessions table exists...")
    stdout, stderr, code = run_sql_query("SELECT table_name FROM information_schema.tables WHERE table_name = 'anomaly_sessions';")
    if code == 0 and 'anomaly_sessions' in stdout:
        print("   ✅ anomaly_sessions table exists")
    else:
        print("   ❌ anomaly_sessions table missing")
        print(f"   Error: {stderr}")
    
    # Test 3: Check if expert_feedback table exists
    print("\n3. Checking if expert_feedback table exists...")
    stdout, stderr, code = run_sql_query("SELECT table_name FROM information_schema.tables WHERE table_name = 'expert_feedback';")
    if code == 0 and 'expert_feedback' in stdout:
        print("   ✅ expert_feedback table exists")
    else:
        print("   ❌ expert_feedback table missing")
        print(f"   Error: {stderr}")
    
    # Test 4: Check labeled_anomalies count
    print("\n4. Checking labeled_anomalies count...")
    stdout, stderr, code = run_sql_query("SELECT COUNT(*) FROM labeled_anomalies;")
    if code == 0:
        print(f"   ✅ labeled_anomalies count: {stdout.strip()}")
    else:
        print(f"   ❌ Error checking labeled_anomalies: {stderr}")
    
    # Test 5: Check ml_sessions count
    print("\n5. Checking ml_sessions count...")
    stdout, stderr, code = run_sql_query("SELECT COUNT(*) FROM ml_sessions;")
    if code == 0:
        print(f"   ✅ ml_sessions count: {stdout.strip()}")
    else:
        print(f"   ❌ Error checking ml_sessions: {stderr}")

def test_api_endpoints():
    """Test if API endpoints are working"""
    print("\n=== Testing API Endpoints ===")
    
    # Test continuous learning status
    print("\n1. Testing continuous learning status...")
    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'GET', 
            'http://localhost:8000/api/v1/continuous-learning/status'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                status = json.loads(result.stdout)
                print(f"   ✅ Status API working: {json.dumps(status, indent=2)}")
            except json.JSONDecodeError:
                print(f"   ⚠️  Status API responded but not JSON: {result.stdout}")
        else:
            print(f"   ❌ Status API failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error testing status API: {e}")
    
    # Test retraining trigger
    print("\n2. Testing retraining trigger...")
    try:
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', 
            'http://localhost:8000/api/v1/continuous-learning/trigger-retraining'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                print(f"   ✅ Retraining API working: {json.dumps(response, indent=2)}")
            except json.JSONDecodeError:
                print(f"   ⚠️  Retraining API responded but not JSON: {result.stdout}")
        else:
            print(f"   ❌ Retraining API failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error testing retraining API: {e}")

def main():
    print("=== Database Schema and Retraining Fix Test ===")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test database schema
    test_database_schema()
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n=== Test Complete ===")
    print("\nNext steps:")
    print("1. If schema issues persist, manually run the migration:")
    print("   docker compose exec postgres psql -U abm_user -d abm_db")
    print("   Then paste the contents of database/migrations/003_fix_missing_schema.sql")
    print("2. If APIs work, rebuild services:")
    print("   docker compose build && docker compose up -d")
    print("3. Test the retraining button in the web UI")

if __name__ == "__main__":
    main()
