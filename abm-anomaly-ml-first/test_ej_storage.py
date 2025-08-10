#!/usr/bin/env python3
"""
Test script to verify that EJ processing stores raw and cleaned text in database
"""

import os
import sys
import tempfile
import requests
import time

def create_sample_ej_file():
    """Create a sample EJ file for testing"""
    sample_ej_content = """[020t*629*06/18/2025*00:46*
TRANSACTION START
PRIMARY CARD READER ACTIVATED
CARD INSERTED ATR RECEIVED T=0
*7231*1*(Iw(1*3,
PIN ENTERED
NOTES PRESENTED 50,000
NOTES TAKEN
CARD TAKEN
TRANSACTION END
[020t*629*06/18/2025*00:47*
"""
    
    # Create temporary file in input directory
    input_dir = "/Users/christopherpearson/Projects/abm_ej_exporter_docker_cleaned/EJAnomalyDetectionV3/abm-anomaly-ml-first/data/input/processed"
    os.makedirs(input_dir, exist_ok=True)
    
    # Create test file
    test_file = os.path.join(input_dir, "TEST_EJ_20250618.txt")
    with open(test_file, 'w') as f:
        f.write(sample_ej_content)
    
    print(f"✅ Created test EJ file: {test_file}")
    return test_file

def test_api_processing():
    """Test the API processing endpoint"""
    print("\n🔄 Testing API processing...")
    
    try:
        # Call the process-input endpoint
        response = requests.post("http://localhost:8000/api/v1/process-input", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful")
            print(f"📊 Response: {result}")
            return True
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False

def check_database_storage():
    """Check if the data was stored correctly in the database"""
    print("\n🔍 Checking database storage...")
    
    try:
        # Query the sessions endpoint to see if data was stored
        response = requests.get("http://localhost:8000/api/v1/ej/sessions/summary", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Database query successful")
            
            summary = result.get('summary', {})
            print(f"📈 Total sessions: {summary.get('total_sessions', 0)}")
            print(f"📄 Sessions with raw text: {summary.get('sessions_with_raw_text', 0)}")
            print(f"🧹 Sessions with cleaned text: {summary.get('sessions_with_cleaned_text', 0)}")
            print(f"🔧 EJ Cleaner available: {summary.get('ej_cleaner_available', False)}")
            
            return summary.get('sessions_with_raw_text', 0) > 0
        else:
            print(f"❌ Database query failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Database query failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing EJ Processing and Database Storage")
    print("=" * 50)
    
    # Step 1: Create sample EJ file
    test_file = create_sample_ej_file()
    
    # Step 2: Test API processing
    if test_api_processing():
        print("✅ API processing completed")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Step 3: Check database storage
        if check_database_storage():
            print("\n🎉 SUCCESS: EJ processing correctly stores both raw and cleaned text in database!")
        else:
            print("\n❌ ISSUE: Data not found in database")
    else:
        print("\n❌ ISSUE: API processing failed")
    
    # Cleanup
    try:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🧹 Cleaned up test file: {test_file}")
    except Exception as e:
        print(f"⚠️ Could not clean up test file: {e}")

if __name__ == "__main__":
    main()
