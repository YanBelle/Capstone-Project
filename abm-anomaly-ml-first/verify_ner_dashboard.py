#!/usr/bin/env python3
"""
NER Dashboard Verification Script
================================

Quick verification to check if NER endpoints and dashboard updates are working.
"""

import requests
import json
import time
from datetime import datetime

def test_ner_endpoints():
    """Test if NER endpoints are accessible"""
    base_url = "http://localhost:8001"  # From your status output
    
    print("🧪 Testing NER Endpoints")
    print("=" * 40)
    
    # Test NER training status
    try:
        response = requests.get(f"{base_url}/api/v1/ner-training/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ NER Training Status endpoint working")
            print(f"   Model Accuracy: {data.get('modelAccuracy', 0)*100:.1f}%")
            print(f"   Training Status: {'Active' if data.get('isTraining') else 'Ready'}")
        else:
            print(f"❌ NER Training Status: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ NER Training Status failed: {e}")
    
    # Test NER stats
    try:
        response = requests.get(f"{base_url}/api/v1/ner-training/stats")
        if response.status_code == 200:
            data = response.json()
            print("✅ NER Stats endpoint working")
            print(f"   Training Data: {data.get('totalTrainingData', 0)} samples")
            print(f"   Entity Types: {len(data.get('entityTypes', []))} categories")
        else:
            print(f"❌ NER Stats: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ NER Stats failed: {e}")
    
    # Test fine-tuned sessionization
    try:
        test_data = {
            "text": """[020t*632*06/18/2025*04:48*
     *TRANSACTION START*
[020t CARD INSERTED
  PAN 0004263********2113
DEVICE ERROR
ESC: 000"""
        }
        
        response = requests.post(
            f"{base_url}/api/v1/sessionize-fine-tuned",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Fine-tuned Sessionization working")
            if 'sessions' in result:
                print(f"   Sessions extracted: {len(result['sessions'])}")
                if 'analytics' in result:
                    print(f"   Entities found: {result['analytics'].get('total_entities_found', 0)}")
        else:
            print(f"❌ Fine-tuned Sessionization: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Fine-tuned Sessionization failed: {e}")

def test_dashboard_access():
    """Test dashboard accessibility"""
    dashboard_url = "http://localhost:3000"  # Your dashboard port
    
    print("\n🌐 Testing Dashboard Access")
    print("=" * 40)
    
    try:
        response = requests.get(dashboard_url, timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard accessible")
            print(f"   URL: {dashboard_url}")
            print("   Note: NER tab should appear after rebuild completes")
        else:
            print(f"❌ Dashboard: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard access failed: {e}")
        print("   Dashboard may still be rebuilding...")

def show_next_steps():
    """Show next steps to see NER in dashboard"""
    print("\n🔧 Next Steps to See NER Tab")
    print("=" * 40)
    print("1. Wait for dashboard rebuild to complete")
    print("2. Restart dashboard service:")
    print("   docker restart dashboard_dev")
    print("3. Access dashboard with new NER tab:")
    print("   http://localhost:3000/dashboard/ner-training")
    print("4. If still not visible, clear browser cache")
    print("\n📋 NER Tab Features:")
    print("   • Real-time training progress")
    print("   • ABM entity recognition (9 types)")
    print("   • Performance comparison charts")
    print("   • Model export/deployment tools")
    print("   • Training logs and statistics")

def main():
    print("🧠 NER Dashboard Verification")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test backend endpoints
    test_ner_endpoints()
    
    # Test dashboard access
    test_dashboard_access()
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "=" * 50)
    print("🎯 Verification Complete!")
    print("\nIf NER tab is not visible yet:")
    print("1. Dashboard is likely still rebuilding")
    print("2. Run: docker restart dashboard_dev")
    print("3. Navigate to: /dashboard/ner-training")

if __name__ == "__main__":
    main()
