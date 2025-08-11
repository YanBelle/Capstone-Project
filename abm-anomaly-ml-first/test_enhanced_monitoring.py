#!/usr/bin/env python3
"""
Simple Monitoring Test Script
Tests the enhanced monitoring interface by simulating some activity
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"

def test_monitoring_endpoint():
    """Test the monitoring status endpoint"""
    print("🔍 Testing monitoring endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/monitoring/status", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Monitoring endpoint is working!")
            
            # Display current stats
            parsing = data.get('parsing', {})
            ml_training = data.get('ml_training', {})
            system = data.get('system', {})
            
            print(f"\n📊 Current Status:")
            print(f"   Parsing: {parsing.get('status', 'unknown')} - {parsing.get('processed', 0)} files processed")
            print(f"   ML Training: {ml_training.get('status', 'unknown')} - {ml_training.get('accuracy', 0):.3f} accuracy")
            print(f"   System: CPU {system.get('cpu_usage', 0):.1f}%, Memory {system.get('memory_usage', 0):.1f}%")
            
            return True
        else:
            print(f"❌ Endpoint returned status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing monitoring endpoint: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection for real-time updates"""
    print("\n🔌 Testing WebSocket connection...")
    
    try:
        import websocket
        
        def on_message(ws, message):
            data = json.loads(message)
            print(f"📨 WebSocket message received: {len(message)} chars")
            
        def on_error(ws, error):
            print(f"❌ WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print("🔌 WebSocket connection closed")
            
        def on_open(ws):
            print("✅ WebSocket connection opened")
            
        ws_url = f"ws://localhost:8000/ws/monitoring"
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run for 10 seconds
        import threading
        def run_websocket():
            ws.run_forever()
            
        ws_thread = threading.Thread(target=run_websocket)
        ws_thread.daemon = True
        ws_thread.start()
        
        time.sleep(10)
        ws.close()
        return True
        
    except ImportError:
        print("⚠️ websocket-client not available, skipping WebSocket test")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_process_input_endpoint():
    """Test the process input endpoint to trigger some activity"""
    print("\n🔄 Testing process input endpoint...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/process-input", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Process input successful: {data.get('message', 'No message')}")
            return True
        else:
            print(f"⚠️ Process input returned {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing process input: {e}")
        return False

def monitor_for_changes(duration=15):
    """Monitor the system for changes over a period"""
    print(f"\n📊 Monitoring system for {duration} seconds...")
    
    start_time = time.time()
    last_stats = None
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/monitoring/status", timeout=5)
            if response.status_code == 200:
                current_stats = response.json()
                
                if last_stats:
                    # Check for changes
                    parsing_change = current_stats.get('parsing', {}).get('processed', 0) - last_stats.get('parsing', {}).get('processed', 0)
                    training_change = current_stats.get('ml_training', {}).get('accuracy', 0) - last_stats.get('ml_training', {}).get('accuracy', 0)
                    
                    if parsing_change > 0:
                        print(f"📈 Parsing progress: +{parsing_change} files processed")
                    
                    if abs(training_change) > 0.001:
                        print(f"🧠 Training update: accuracy changed by {training_change:.3f}")
                
                # Display current system load
                system = current_stats.get('system', {})
                cpu = system.get('cpu_usage', 0)
                memory = system.get('memory_usage', 0)
                
                if time.time() % 5 < 1:  # Every 5 seconds
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] 💻 System: CPU {cpu:.1f}%, Memory {memory:.1f}%")
                
                last_stats = current_stats
            
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            time.sleep(2)

def main():
    """Main test function"""
    print("🧪 Enhanced Monitoring Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic monitoring endpoint
    total_tests += 1
    if test_monitoring_endpoint():
        tests_passed += 1
    
    # Test 2: Process input (to generate some activity)
    total_tests += 1
    if test_process_input_endpoint():
        tests_passed += 1
    
    # Test 3: Monitor for changes
    monitor_for_changes(10)
    
    # Test 4: WebSocket connection
    total_tests += 1
    if test_websocket_connection():
        tests_passed += 1
    
    print(f"\n📋 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Enhanced monitoring is working.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n🌐 Enhanced monitoring interface: http://localhost/dashboard/realtime")
    print("\n📈 Features available:")
    print("   • Real-time EJ file processing progress with progress bars")
    print("   • Model training progress with accuracy tracking")
    print("   • System resource monitoring")
    print("   • WebSocket-based live updates")
    print("   • Enhanced error tracking and ETA calculations")

if __name__ == "__main__":
    main()
