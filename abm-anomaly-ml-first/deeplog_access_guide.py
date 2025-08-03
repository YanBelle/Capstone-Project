#!/usr/bin/env python3
"""
DeepLog Dashboard Access Helper
Provides correct URLs for accessing the BERT-Enhanced DeepLog system on DigitalOcean
"""

import requests
import json

DIGITALOCEAN_IP = "64.227.16.180"
BASE_URL = f"http://{DIGITALOCEAN_IP}"

def check_server_status():
    """Check if the DigitalOcean server is accessible"""
    print("🌐 BERT-Enhanced DeepLog Server Access Guide")
    print("=" * 60)
    
    # Test basic connectivity
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ DigitalOcean Server Status: ONLINE")
            print(f"   Server IP: {DIGITALOCEAN_IP}")
            print(f"   Health Status: {response.text.strip()}")
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to DigitalOcean server: {e}")
        return False
    
    return True

def show_access_urls():
    """Display the correct URLs for accessing the system"""
    print(f"\n📋 Correct Access URLs:")
    print(f"=" * 60)
    
    print(f"🎯 Main Dashboard:")
    print(f"   {BASE_URL}/")
    
    print(f"\n🧠 BERT-Enhanced DeepLog Dashboard:")
    print(f"   {BASE_URL}/dashboard/deeplog")
    
    print(f"\n🔧 API Endpoints:")
    print(f"   Model Info:     {BASE_URL}/api/v1/bert-deeplog/model-info")
    print(f"   Load Sessions:  {BASE_URL}/api/v1/bert-deeplog/load-ej-sessions")
    print(f"   Training:       {BASE_URL}/api/v1/bert-deeplog/train")
    print(f"   Prediction:     {BASE_URL}/api/v1/bert-deeplog/predict")
    
    print(f"\n📊 Monitoring:")
    print(f"   Grafana:        {BASE_URL}:3001")
    print(f"   Prometheus:     {BASE_URL}:9090")

def test_deeplog_api():
    """Test if BERT-DeepLog API is accessible"""
    print(f"\n🧪 Testing BERT-DeepLog API:")
    print(f"=" * 60)
    
    endpoints = [
        ("Model Info", "/api/v1/bert-deeplog/model-info"),
        ("Load Sessions", "/api/v1/bert-deeplog/load-ej-sessions?limit=1"),
    ]
    
    for name, endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: Working")
                if endpoint.endswith("model-info"):
                    data = response.json()
                    model_stats = data.get('model_stats', {}).get('model_info', {})
                    print(f"   Parameters: {model_stats.get('parameters', 'N/A'):,}")
                    print(f"   Device: {model_stats.get('device', 'N/A')}")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")

def main():
    if check_server_status():
        show_access_urls()
        test_deeplog_api()
        
        print(f"\n" + "=" * 60)
        print(f"🎯 Quick Access Commands:")
        print(f"   Open Dashboard: firefox {BASE_URL}/dashboard/deeplog")
        print(f"   Test API:       curl {BASE_URL}/api/v1/bert-deeplog/model-info")
        print(f"=" * 60)
        
        # Show warning about local access
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"   • Do NOT use localhost URLs")
        print(f"   • Always use {DIGITALOCEAN_IP} for production access")
        print(f"   • Local services are now stopped to prevent confusion")
    else:
        print(f"\n❌ Cannot access DigitalOcean server")
        print(f"   Please check if services are running on the server")

if __name__ == "__main__":
    main()
