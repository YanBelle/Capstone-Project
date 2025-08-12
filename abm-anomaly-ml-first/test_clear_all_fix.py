#!/usr/bin/env python3
"""
Test script to verify the Clear All functionality works properly
and prevents anomalies from showing in the dashboard
"""

import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:8000"  # Adjust if your API runs on a different port

def test_clear_all_functionality():
    """Test the clear all functionality end-to-end"""
    
    print("🧪 Testing Clear All Functionality")
    print("=" * 50)
    
    # Step 1: Check initial state
    print("\n1️⃣ Checking initial dashboard state...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/dashboard/stats")
        if response.status_code == 200:
            initial_stats = response.json()
            print(f"   Initial anomalies: {initial_stats.get('total_anomalies', 0)}")
            print(f"   Initial transactions: {initial_stats.get('total_transactions', 0)}")
        else:
            print(f"   ❌ Failed to get initial stats: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting initial stats: {e}")
    
    # Step 2: Check initial anomalies list
    print("\n2️⃣ Checking initial anomalies list...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/anomalies?limit=10")
        if response.status_code == 200:
            anomalies_data = response.json()
            print(f"   Initial anomalies count: {len(anomalies_data.get('anomalies', []))}")
        else:
            print(f"   ❌ Failed to get initial anomalies: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting initial anomalies: {e}")
    
    # Step 3: Perform clear all operation
    print("\n3️⃣ Performing Clear All operation...")
    try:
        response = requests.delete(f"{API_BASE}/api/v1/data/clear-all?confirm=true")
        if response.status_code == 200:
            clear_result = response.json()
            print(f"   ✅ Clear operation successful")
            print(f"   Redis cleared: {clear_result.get('redis_cleared', False)}")
            print(f"   Cache prevention enabled: {clear_result.get('cache_prevention_enabled', False)}")
            print(f"   Total records deleted: {clear_result.get('total_records_deleted', 0)}")
            
            # Show deleted counts
            deleted_counts = clear_result.get('deleted_counts', {})
            for table, count in deleted_counts.items():
                if isinstance(count, int) and count > 0:
                    print(f"   - {table}: {count} records")
        else:
            print(f"   ❌ Clear operation failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error during clear operation: {e}")
        return False
    
    # Step 4: Wait a moment for changes to propagate
    print("\n4️⃣ Waiting for changes to propagate...")
    time.sleep(2)
    
    # Step 5: Check dashboard stats after clearing
    print("\n5️⃣ Checking dashboard stats after clearing...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/dashboard/stats")
        if response.status_code == 200:
            post_clear_stats = response.json()
            print(f"   Post-clear anomalies: {post_clear_stats.get('total_anomalies', 0)}")
            print(f"   Post-clear transactions: {post_clear_stats.get('total_transactions', 0)}")
            
            # Verify stats are zero
            if post_clear_stats.get('total_anomalies', 0) == 0 and post_clear_stats.get('total_transactions', 0) == 0:
                print("   ✅ Dashboard stats correctly show zero values")
            else:
                print("   ❌ Dashboard stats still showing non-zero values")
                return False
        else:
            print(f"   ❌ Failed to get post-clear stats: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting post-clear stats: {e}")
        return False
    
    # Step 6: Check anomalies list after clearing
    print("\n6️⃣ Checking anomalies list after clearing...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/anomalies?limit=10")
        if response.status_code == 200:
            post_clear_anomalies = response.json()
            anomalies_count = len(post_clear_anomalies.get('anomalies', []))
            print(f"   Post-clear anomalies count: {anomalies_count}")
            
            if anomalies_count == 0:
                print("   ✅ Anomalies list correctly empty")
            else:
                print("   ❌ Anomalies list still contains data")
                return False
        else:
            print(f"   ❌ Failed to get post-clear anomalies: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting post-clear anomalies: {e}")
        return False
    
    # Step 7: Test cache prevention
    print("\n7️⃣ Testing cache prevention (waiting 10 seconds)...")
    time.sleep(10)
    
    try:
        response = requests.get(f"{API_BASE}/api/v1/dashboard/stats")
        if response.status_code == 200:
            stats_after_wait = response.json()
            print(f"   Stats after wait - anomalies: {stats_after_wait.get('total_anomalies', 0)}")
            
            if stats_after_wait.get('total_anomalies', 0) == 0:
                print("   ✅ Cache prevention working - stats remain zero")
            else:
                print("   ❌ Cache was repopulated despite prevention flag")
                return False
        else:
            print(f"   ❌ Failed to get stats after wait: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting stats after wait: {e}")
        return False
    
    print("\n🎉 All tests passed! Clear All functionality is working correctly.")
    return True

def print_instructions():
    """Print instructions for manual testing"""
    print("\n📋 Manual Testing Instructions:")
    print("=" * 30)
    print("1. Open your dashboard in a web browser")
    print("2. Note any existing anomaly counts")
    print("3. Click the 'Clear All' button")
    print("4. Confirm the operation")
    print("5. Verify that:")
    print("   - All anomaly counters show 0")
    print("   - Anomaly lists are empty")
    print("   - Dashboard charts show no data")
    print("   - Stats remain at 0 after refreshing")
    print("6. Wait 10+ minutes and verify stats remain at 0")
    print("\nIf any of these fail, the issue persists.")

if __name__ == "__main__":
    print("🚀 Clear All Functionality Test")
    print(f"Testing API at: {API_BASE}")
    print("Make sure your API server is running!")
    
    # Test the functionality
    success = test_clear_all_functionality()
    
    if success:
        print("\n✅ Automated tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    # Print manual testing instructions
    print_instructions()
