#!/usr/bin/env python3
"""
Test BERT-Enhanced DeepLog on DigitalOcean
Run this script to verify the deployment is working correctly
"""

import requests
import time
import json

DIGITALOCEAN_IP = "64.227.16.180"
BASE_URL = f"http://{DIGITALOCEAN_IP}"

def test_digitalocean_deployment():
    """Test the BERT-Enhanced DeepLog deployment on DigitalOcean"""
    
    print("🧪 Testing BERT-Enhanced DeepLog on DigitalOcean")
    print("=" * 60)
    print(f"Server: {DIGITALOCEAN_IP}")
    print("=" * 60)
    
    tests = [
        ("🏥 Health Check", "/health"),
        ("🏠 Dashboard Home", "/"),
        ("🧠 DeepLog Dashboard", "/dashboard/deeplog"),
        ("📊 Model Info API", "/api/v1/bert-deeplog/model-info"),
        ("📁 Load EJ Sessions", "/api/v1/bert-deeplog/load-ej-sessions?limit=1"),
    ]
    
    results = []
    
    for test_name, endpoint in tests:
        print(f"\n{test_name}:")
        print(f"  URL: {BASE_URL}{endpoint}")
        
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ Status: {response.status_code}")
                
                # Special handling for specific endpoints
                if endpoint == "/health":
                    print(f"  📝 Response: {response.text.strip()}")
                elif endpoint.endswith("model-info"):
                    try:
                        data = response.json()
                        if 'model_stats' in data:
                            model_info = data['model_stats']['model_info']
                            print(f"  📊 Model Parameters: {model_info.get('parameters', 'N/A'):,}")
                            print(f"  💻 Device: {model_info.get('device', 'N/A')}")
                            print(f"  🎯 Trained: {model_info.get('trained', False)}")
                    except:
                        print(f"  📄 JSON Response received")
                elif endpoint.endswith("load-ej-sessions?limit=1"):
                    try:
                        data = response.json()
                        sessions = data.get('sessions', [])
                        print(f"  📁 Sessions loaded: {len(sessions)}")
                        if sessions:
                            print(f"  🎯 Sample session ID: {sessions[0].get('session_id', 'N/A')[:50]}...")
                    except:
                        print(f"  📄 JSON Response received")
                elif endpoint in ["/", "/dashboard/deeplog"]:
                    if "html" in response.headers.get('content-type', '').lower():
                        print(f"  📄 HTML page loaded successfully")
                        # Check if it contains React app content
                        if 'id="root"' in response.text:
                            print(f"  ⚛️  React app detected")
                    else:
                        print(f"  📄 Content loaded")
                
                results.append((test_name, True, response.status_code))
            else:
                print(f"  ❌ Status: {response.status_code}")
                print(f"  📄 Response: {response.text[:200]}...")
                results.append((test_name, False, response.status_code))
                
        except requests.exceptions.ConnectTimeout:
            print(f"  ⏰ Connection timeout")
            results.append((test_name, False, "Timeout"))
        except requests.exceptions.ConnectionError:
            print(f"  🔌 Connection refused - service not running")
            results.append((test_name, False, "Connection Error"))
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, status in results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS! BERT-Enhanced DeepLog is fully deployed and working!")
        print(f"🔗 Access your dashboard at: {BASE_URL}/dashboard/deeplog")
    elif passed > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: Some services are working")
        print(f"🎯 Try accessing: {BASE_URL}/dashboard/deeplog")
    else:
        print(f"\n❌ DEPLOYMENT FAILED: No services are responding")
        print(f"🛠️  Check if Docker services are running on {DIGITALOCEAN_IP}")
    
    return passed == total

if __name__ == "__main__":
    success = test_digitalocean_deployment()
    
    if not success:
        print(f"\n🔧 TROUBLESHOOTING STEPS:")
        print(f"1. SSH to your server: ssh root@{DIGITALOCEAN_IP}")
        print(f"2. Check if Docker is running: docker ps")
        print(f"3. Navigate to project: cd /root/Capstone-Project/abm-anomaly-ml-first")
        print(f"4. Check service status: docker-compose ps")
        print(f"5. View logs: docker-compose logs")
        print(f"6. Restart services: docker-compose down && docker-compose up -d")
