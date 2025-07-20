#!/usr/bin/env python3

"""
Test script to verify retraining improvements with specific transaction examples
"""

import requests
import json
import time
import tempfile
import os

def create_test_file_with_problematic_transactions():
    """Create a test EJ file with the transactions that should be anomalies"""
    
    test_content = """
[020t15706/18/202513:39
TRANSACTION START
[020t CARD INSERTED
[020t 13:39:56 CARD TAKEN
[000p[040q(I 75561D(10,M-090B0210B9,R-4S
[000p[040q(I 75561D(10,M-00,R-4S
[020t 13:39:56 TRANSACTION END
[020t15806/18/202513:39
PRIMARY CARD READER ACTIVATED

[020t*209*06/18/2025*14:23*
      *TRANSACTION START*
[020t CARD INSERTED
14:23:03 ATR RECEIVED T=0
[020t 14:23:06 OPCODE = FI      
PAN 0004263********6687
---START OF TRANSACTION---
[020t 14:23:22 PIN ENTERED
[020t 14:23:36 OPCODE = BC      
PAN 0004263********6687
---START OF TRANSACTION---
[020t 14:24:28 CARD TAKEN
[020t 14:24:29 TRANSACTION END
[020t*210*06/18/2025*14:24*
      *PRIMARY CARD READER ACTIVATED*

[020t*211*06/18/2025*15:30*
      *TRANSACTION START*
[020t CARD INSERTED
15:30:05 ATR RECEIVED T=0
[020t 15:30:08 OPCODE = FI
PAN 0004263********1234
---START OF TRANSACTION---
[020t 15:30:15 PIN ENTERED
[020t 15:30:18 OPCODE = BC
[020t 15:30:25 DISPENSE SUCCESS
[020t 15:30:28 CASH DISPENSED: $100
[020t 15:30:30 CARD TAKEN
[020t 15:30:31 TRANSACTION END
[020t*212*06/18/2025*15:30*
      *PRIMARY CARD READER ACTIVATED*
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        return f.name

def test_retraining_before_and_after():
    """Test the system before and after retraining"""
    
    print("=== Testing Retraining Improvements ===")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create test file with problematic transactions
    test_file = create_test_file_with_problematic_transactions()
    print(f"\nCreated test file: {test_file}")
    
    try:
        # Step 1: Check current status
        print("\n1. Checking current continuous learning status...")
        response = requests.get("http://localhost:8000/api/v1/continuous-learning/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Current feedback buffer size: {status.get('feedback_buffer_size', 0)}")
            print(f"   Database labeled count: {status.get('feedback_database_count', 0)}")
            print(f"   Retraining cycles: {status.get('retraining_cycles', 0)}")
        else:
            print(f"   Status check failed: {response.status_code}")
        
        # Step 2: Upload test file (before retraining)
        print("\n2. Processing test file BEFORE retraining...")
        with open(test_file, 'rb') as f:
            files = {'file': ('test_transactions.txt', f, 'text/plain')}
            response = requests.post("http://localhost:8000/api/v1/upload", files=files)
            if response.status_code == 200:
                print("   ✅ Test file uploaded successfully")
                upload_result = response.json()
                print(f"   Response: {upload_result}")
            else:
                print(f"   ❌ Upload failed: {response.status_code}")
        
        # Wait for processing
        time.sleep(5)
        
        # Step 3: Check for anomalies (before retraining)
        print("\n3. Checking anomaly detection BEFORE retraining...")
        response = requests.get("http://localhost:8000/api/v1/sessions/recent")
        if response.status_code == 200:
            sessions = response.json()
            print(f"   Found {len(sessions)} sessions")
            
            anomalies_before = [s for s in sessions if s.get('is_anomaly', False)]
            print(f"   Anomalies detected BEFORE retraining: {len(anomalies_before)}")
            
            for anomaly in anomalies_before[:3]:  # Show first 3
                print(f"     - {anomaly.get('session_id', 'unknown')}: {anomaly.get('anomaly_type', 'unknown')} (score: {anomaly.get('anomaly_score', 0):.2f})")
        else:
            print(f"   Sessions check failed: {response.status_code}")
            anomalies_before = []
        
        # Step 4: Trigger retraining
        print("\n4. Triggering retraining...")
        response = requests.post("http://localhost:8000/api/v1/continuous-learning/trigger-retraining")
        if response.status_code == 200:
            retrain_result = response.json()
            print(f"   ✅ Retraining triggered: {retrain_result}")
        else:
            print(f"   ❌ Retraining failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        # Wait for retraining to complete
        print("\n5. Waiting for retraining to complete...")
        time.sleep(30)
        
        # Step 6: Check status after retraining
        print("\n6. Checking status AFTER retraining...")
        response = requests.get("http://localhost:8000/api/v1/continuous-learning/status")
        if response.status_code == 200:
            status_after = response.json()
            print(f"   Feedback buffer size: {status_after.get('feedback_buffer_size', 0)}")
            print(f"   Retraining cycles: {status_after.get('retraining_cycles', 0)}")
            print(f"   Performance improvement: {status_after.get('last_performance_improvement', 0):.3f}")
        
        # Step 7: Re-upload test file (after retraining)
        print("\n7. Re-processing test file AFTER retraining...")
        with open(test_file, 'rb') as f:
            files = {'file': ('test_transactions_after.txt', f, 'text/plain')}
            response = requests.post("http://localhost:8000/api/v1/upload", files=files)
            if response.status_code == 200:
                print("   ✅ Test file re-uploaded successfully")
            else:
                print(f"   ❌ Re-upload failed: {response.status_code}")
        
        # Wait for processing
        time.sleep(5)
        
        # Step 8: Check for anomalies (after retraining)
        print("\n8. Checking anomaly detection AFTER retraining...")
        response = requests.get("http://localhost:8000/api/v1/sessions/recent")
        if response.status_code == 200:
            sessions = response.json()
            
            # Filter for sessions processed after retraining
            recent_sessions = [s for s in sessions if 'test_transactions_after' in s.get('session_id', '')]
            anomalies_after = [s for s in recent_sessions if s.get('is_anomaly', False)]
            
            print(f"   Anomalies detected AFTER retraining: {len(anomalies_after)}")
            
            for anomaly in anomalies_after[:3]:  # Show first 3
                print(f"     - {anomaly.get('session_id', 'unknown')}: {anomaly.get('anomaly_type', 'unknown')} (score: {anomaly.get('anomaly_score', 0):.2f})")
        else:
            print(f"   Sessions check failed: {response.status_code}")
            anomalies_after = []
        
        # Step 9: Compare results
        print("\n9. Comparing results...")
        improvement = len(anomalies_after) - len(anomalies_before)
        if improvement > 0:
            print(f"   🎉 IMPROVEMENT: {improvement} more anomalies detected after retraining!")
            print("   ✅ Your problematic transactions should now be flagged")
        elif improvement == 0:
            print("   ⚠️  Same number of anomalies detected - may need more training data")
        else:
            print("   ❌ Fewer anomalies detected - possible overfitting")
        
        print("\n=== Expected Improvements ===")
        print("After retraining, you should see:")
        print("- Transaction 1 (card inserted/taken quickly) → FLAGGED AS ANOMALY")
        print("- Transaction 2 (PIN entered, no dispense) → FLAGGED AS ANOMALY")
        print("- Transaction 3 (successful transaction) → NOT FLAGGED")
        print("- Better anomaly type classification")
        print("- More business-relevant anomaly categories")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    test_retraining_before_and_after()
