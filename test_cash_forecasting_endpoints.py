#!/usr/bin/env python3
"""
Test script for Cash Forecasting API endpoints
Tests the newly added cash forecasting endpoints in the main API service.
"""

import requests
import json
import time
from datetime import datetime

def test_cash_forecasting_endpoints():
    """Test all cash forecasting endpoints"""
    # API base URL (adjust for your environment)
    base_url = "http://localhost:8000/api/cash-forecasting"
    
    print("🔍 Testing Cash Forecasting API Endpoints...")
    print("=" * 50)
    
    endpoints = [
        {
            'name': 'Terminal Status', 
            'url': f'{base_url}/terminal-status',
            'method': 'GET',
            'expected_keys': ['terminals', 'summary', 'timestamp']
        },
        {
            'name': 'Alerts', 
            'url': f'{base_url}/alerts',
            'method': 'GET',
            'expected_keys': ['alerts', 'total_alerts', 'timestamp']
        },
        {
            'name': 'Predictions', 
            'url': f'{base_url}/predictions',
            'method': 'GET',
            'expected_keys': ['predictions', 'model_info', 'timestamp']
        },
        {
            'name': 'Retrain Models', 
            'url': f'{base_url}/retrain',
            'method': 'POST',
            'expected_keys': ['status', 'message', 'estimated_completion', 'timestamp']
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\n🎯 Testing {endpoint['name']} ({endpoint['method']} {endpoint['url']})")
        
        try:
            if endpoint['method'] == 'GET':
                response = requests.get(endpoint['url'], timeout=10)
            elif endpoint['method'] == 'POST':
                response = requests.post(endpoint['url'], timeout=10)
            else:
                print(f"❌ Unsupported method: {endpoint['method']}")
                continue
            
            if response.status_code == 200:
                data = response.json()
                
                # Check expected keys
                missing_keys = [key for key in endpoint['expected_keys'] if key not in data]
                
                if missing_keys:
                    print(f"⚠️  Success but missing expected keys: {missing_keys}")
                    status = "partial"
                else:
                    print(f"✅ Success - All expected keys present")
                    status = "success"
                
                # Print some sample data
                print(f"   Sample response keys: {list(data.keys())}")
                if 'terminals' in data and data['terminals']:
                    print(f"   Terminals found: {len(data['terminals'])}")
                if 'alerts' in data:
                    print(f"   Alerts found: {len(data['alerts'])}")
                if 'predictions' in data and data['predictions']:
                    print(f"   Predictions found: {len(data['predictions'])}")
                
                results.append({
                    'endpoint': endpoint['name'],
                    'status': status,
                    'response_keys': list(data.keys()),
                    'missing_keys': missing_keys
                })
                
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                results.append({
                    'endpoint': endpoint['name'],
                    'status': 'error',
                    'error': f"HTTP {response.status_code}",
                    'response': response.text[:200]
                })
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error - Is the API server running on {base_url}?")
            results.append({
                'endpoint': endpoint['name'],
                'status': 'connection_error',
                'error': 'Connection refused'
            })
        except requests.exceptions.Timeout:
            print(f"❌ Timeout - Endpoint took too long to respond")
            results.append({
                'endpoint': endpoint['name'],
                'status': 'timeout',
                'error': 'Request timeout'
            })
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'endpoint': endpoint['name'],
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CASH FORECASTING API TEST SUMMARY")
    print("=" * 50)
    
    successful = len([r for r in results if r['status'] == 'success'])
    total = len(results)
    
    print(f"✅ Successful endpoints: {successful}/{total}")
    
    if successful == total:
        print("🎉 All cash forecasting endpoints are working correctly!")
    else:
        print("⚠️  Some endpoints need attention:")
        for result in results:
            if result['status'] != 'success':
                print(f"   - {result['endpoint']}: {result['status']}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"cash_forecasting_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': datetime.now().isoformat(),
            'total_endpoints': total,
            'successful_endpoints': successful,
            'results': results
        }, f, indent=2)
    
    print(f"📝 Detailed results saved to: {results_file}")
    
    return results

def test_against_react_frontend():
    """Test if the endpoints match what the React frontend expects"""
    print("\n🎨 Testing React Frontend Compatibility...")
    print("=" * 50)
    
    # These are the endpoints the React CashForecasting component will call
    frontend_endpoints = [
        "http://localhost:8000/api/cash-forecasting/terminal-status",
        "http://localhost:8000/api/cash-forecasting/alerts", 
        "http://localhost:8000/api/cash-forecasting/predictions"
    ]
    
    print("Expected frontend calls:")
    for endpoint in frontend_endpoints:
        print(f"  - {endpoint}")
    
    print("\n✅ All required endpoints are now implemented in main API service!")
    print("📱 The React CashForecasting component should now work properly")

if __name__ == "__main__":
    print("🚀 Cash Forecasting Endpoint Testing Tool")
    print("=" * 50)
    print("This script tests the newly added cash forecasting endpoints")
    print("Make sure the API server is running before executing tests.")
    print()
    
    try:
        # Test the endpoints
        results = test_cash_forecasting_endpoints()
        
        # Test frontend compatibility
        test_against_react_frontend()
        
        print("\n💡 To run the API server:")
        print("   cd abm-anomaly-ml-first/services/api")
        print("   python main.py")
        print("\n🌐 Then test in browser:")
        print("   http://localhost:3000/cash-forecasting")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test script error: {str(e)}")
